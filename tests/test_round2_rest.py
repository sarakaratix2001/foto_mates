"""Round 2, Tasks 12 to 16: enhance only the whole photo, prompt v3, displays on their
side, the repeated-digit line, example photos, and the gmm enhancer for glare only.
Every one is behind a setting that is off by default.

    python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detect_quality  # noqa: E402
import extract_readings  # noqa: E402
from extract_readings import (_read_photo, box_turned, crop_rect, extract_decimal_digits,  # noqa: E402
                              load_fewshots, settings_note)

BOX = (250, 300, 750, 540)    # a display lying across a 240x400 photo
TALL = (400, 200, 600, 800)   # a display on its side: 48 px wide, 240 px tall


def answer(text):
    return NS(model_extra=None, usage=None, choices=[NS(message=NS(content=text), logprobs=None)])


def send(box=BOX, crop=True, answers=None, **settings):
    """What _read_photo sends for a grey 240x400 photo: the pixels' shape, the labels, the messages."""
    photo = np.full((400, 240, 3), 128, np.uint8)
    seen, aids, replies = dict(shapes=[]), [], list(answers or [])

    def model(*args, **kwargs):
        seen['messages'] = kwargs['messages']
        if not replies:
            raise RuntimeError('stop before the model')
        return answer(replies.pop(0))

    def png(pixels):
        seen['shapes'].append(pixels.shape[:2])
        return 'data:image/png;base64,x'

    flag = extract_readings._ENHANCE_WHEN[0]
    with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
            mock.patch.object(extract_readings, 'assess', return_value={flag: True, 'contrast': 200.0}), \
            mock.patch.object(extract_readings, 'enhance', side_effect=lambda pixels, report: pixels), \
            mock.patch.object(extract_readings, 'png_data_url', side_effect=png), \
            mock.patch.object(extract_readings, 'complete', side_effect=model), \
            mock.patch.multiple(extract_readings, **(settings or dict(ROTATED=extract_readings.ROTATED))):
        try:
            seen['out'] = _read_photo(Path('x.jpg'), lambda label, work: work(), aids, [], [],
                                      extract_readings.DEFAULT_BACKEND, 3.0, crop, box)
        except RuntimeError:
            pass
    return seen, aids


class EnhanceWholeTest(unittest.TestCase):  # Task 12
    def enhanced(self, mode, crop):
        _, aids = send(crop=crop, _ENHANCE_MODE=mode)
        return any(aid.startswith('enhanced(') for aid in aids)

    def test_whole_never_enhances_a_crop(self):
        self.assertFalse(self.enhanced('whole', crop=True))
        self.assertTrue(self.enhanced('whole', crop=False))

    def test_auto_enhances_both_and_never_neither(self):
        self.assertEqual([self.enhanced('auto', True), self.enhanced('auto', False)], [True, True])
        self.assertEqual([self.enhanced('never', True), self.enhanced('never', False)], [False, False])

    def test_a_crop_that_fell_back_to_the_whole_photo_is_enhanced(self):
        _, aids = send(box=None, _ENHANCE_MODE='whole')
        self.assertEqual(aids[0], 'crop-fallback(no box)')
        self.assertTrue(any(aid.startswith('enhanced(') for aid in aids), aids)


class PromptV3Test(unittest.TestCase):  # Task 13
    DIGITAL = 'TYPE: digital\nREADING: 00123456\nCELLS: 0 0 1 2 3 4 5 6\nDECIMALS: 1\nQUALITY: 4'

    def test_v3_asks_every_meter_for_the_decimals(self):
        system, user = extract_readings.PROMPTS['v3']
        v2_system, v2_user = extract_readings.PROMPTS['v2']
        self.assertIn('smaller than the others', system)
        self.assertIn('DECIMALS', user)
        self.assertNotEqual((system, user), (v2_system, v2_user))
        self.assertIn('CELLS', system)  # the five lines of v2 stay

    def test_the_count_is_taken_from_a_digital_display_only_with_v3(self):
        with mock.patch.object(extract_readings, 'PROMPT', 'v3'):
            self.assertEqual(extract_decimal_digits(self.DIGITAL), 1)
            self.assertIsNone(extract_decimal_digits(self.DIGITAL.replace('DECIMALS: 1', 'DECIMALS: 9')))
        with mock.patch.object(extract_readings, 'PROMPT', 'v2'):
            self.assertIsNone(extract_decimal_digits(self.DIGITAL))
            self.assertEqual(extract_decimal_digits(self.DIGITAL.replace('digital', 'mechanical')), 1)


class RotatedBoxTest(unittest.TestCase):  # Task 14
    def test_a_tall_box_is_refused_unless_the_setting_is_on(self):
        with mock.patch.object(extract_readings, 'ROTATED', False):
            self.assertFalse(box_turned(TALL, 240, 400))
            self.assertEqual(crop_rect(TALL, 240, 400)[0], None)
        with mock.patch.object(extract_readings, 'ROTATED', True), \
                mock.patch.object(extract_readings, 'CROP_MARGIN', 0.25):
            self.assertTrue(box_turned(TALL, 240, 400))
            self.assertEqual(crop_rect(TALL, 240, 400), ((36, 20, 204, 380), None))

    def test_a_box_lying_across_is_cut_as_before(self):
        with mock.patch.object(extract_readings, 'CROP_MARGIN', 0.25):
            usual = crop_rect(BOX, 240, 400)
            with mock.patch.object(extract_readings, 'ROTATED', True):
                self.assertFalse(box_turned(BOX, 240, 400))
                self.assertEqual(crop_rect(BOX, 240, 400), usual)

    def test_a_square_or_thin_tall_box_is_still_refused(self):
        with mock.patch.object(extract_readings, 'ROTATED', True):
            self.assertEqual(crop_rect((400, 400, 600, 520), 240, 400)[0], None)  # 48x48 px: square
            self.assertEqual(crop_rect((490, 100, 510, 900), 240, 400)[0], None)  # 5x320 px: far too thin

    def test_the_cut_is_turned_upright_and_labelled(self):
        seen, aids = send(box=TALL, ROTATED=True)
        height, width = seen['shapes'][0]
        self.assertGreater(width, height)  # the tall cut now lies across
        self.assertTrue(aids[0].startswith('crop(') and aids[0].endswith(',rot90)'), aids)
        seen, aids = send(box=BOX, ROTATED=True)
        self.assertNotIn('rot', aids[0])

    def test_no_digits_and_it_is_read_turned_the_other_way(self):
        good = 'TYPE: digital\nREADING: 01234\nCELLS: 0 1 2 3 4\nQUALITY: 4'
        seen, aids = send(box=TALL, ROTATED=True, answers=['TYPE: UNKNOWN\nREADING: UNKNOWN', good])
        self.assertEqual(len(seen['shapes']), 2)
        self.assertTrue(aids[0].endswith(',rot270)'), aids)
        self.assertEqual(seen['out'][3][0], '1234')

    def test_a_read_with_digits_is_not_read_again(self):
        good = 'TYPE: digital\nREADING: 01234\nCELLS: 0 1 2 3 4\nQUALITY: 4'
        seen, aids = send(box=TALL, ROTATED=True, answers=[good])
        self.assertEqual((len(seen['shapes']), aids[0].endswith(',rot90)')), (1, True))

    def test_two_empty_reads_keep_the_first(self):
        empty = 'TYPE: UNKNOWN\nREADING: UNKNOWN'
        seen, aids = send(box=TALL, ROTATED=True, answers=[empty, empty])
        self.assertEqual(len(seen['shapes']), 2)
        self.assertTrue(aids[0].endswith(',rot90)'), aids)


class ExamplesTest(unittest.TestCase):  # Tasks 15 and 16
    def test_both_are_off_unless_asked(self):
        if 'FOTOMATES_FEWSHOT' not in os.environ:
            self.assertEqual((extract_readings.FEWSHOT, extract_readings._FEWSHOTS), (False, []))
        if 'FOTOMATES_REPEAT_HINT' not in os.environ:
            self.assertFalse(extract_readings.REPEAT_HINT)
            self.assertNotIn(extract_readings._REPEAT_LINE.strip(), extract_readings._SYSTEM_PROMPT)

    def test_the_examples_in_the_project_load(self):
        folder = Path(__file__).resolve().parents[1] / 'fewshot'
        if not list(folder.glob('*.png')):
            self.skipTest('the example photos are not in this copy of the project')
        parts = load_fewshots(folder)
        self.assertEqual(len(parts) % 3, 0)
        self.assertIn(len(parts) // 3, (3, 4))
        self.assertEqual([part['type'] for part in parts[:3]], ['text', 'image_url', 'text'])
        self.assertEqual(parts[2]['text'], 'CELLS: 0 0 7 3 7 9 5 5 5')
        self.assertTrue(parts[1]['image_url']['url'].startswith('data:image/png;base64,'))

    def test_a_missing_list_or_photo_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                load_fewshots(Path(folder))
            (Path(folder) / 'examples.csv').write_text('file,cells\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_fewshots(Path(folder))
            (Path(folder) / 'examples.csv').write_text('file,cells\ngone.png,123\n', encoding='utf-8')
            with self.assertRaises(OSError):
                load_fewshots(Path(folder))

    def test_the_examples_come_before_the_photo(self):
        shots = [{'type': 'text', 'text': 'Example 1 (another meter, already read):'},
                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,e'}},
                 {'type': 'text', 'text': 'CELLS: 1 2 3'}]
        seen, _ = send(_FEWSHOTS=shots)
        parts = seen['messages'][1]['content']
        self.assertEqual(parts[1:4], shots)
        self.assertEqual(parts[4]['text'], 'The meter photo to read:')
        self.assertEqual((len(parts), parts[5]['type']), (6, 'image_url'))
        seen, _ = send()
        self.assertEqual([part.get('text') for part in seen['messages'][1]['content'][1:]], ['Meter photo.', None])


class EnhancerTest(unittest.TestCase):  # Task 16
    def chosen(self, enhancer, report):
        pixels = np.zeros((4, 4, 3), np.uint8)
        with mock.patch.object(detect_quality, 'ENHANCER', enhancer), \
                mock.patch.object(detect_quality, 'enhance_for_ocr', return_value='gmm'), \
                mock.patch.object(detect_quality, 'adjust_light', return_value='light'):
            return detect_quality.enhance(pixels, report)

    def test_glare_gmm_uses_gmm_only_on_glare(self):
        self.assertEqual(self.chosen('glare-gmm', dict(has_glare=True)), 'gmm')
        self.assertEqual(self.chosen('glare-gmm', dict(has_glare=False, is_dark=True)), 'light')
        self.assertEqual(self.chosen('glare-gmm', None), 'light')

    def test_the_other_two_are_as_before(self):
        self.assertEqual(self.chosen('light', dict(has_glare=True)), 'light')
        self.assertEqual(self.chosen('gmm', dict(has_glare=False)), 'gmm')


class SettingsNoteTest(unittest.TestCase):
    def test_nothing_is_said_at_the_defaults(self):
        with mock.patch.multiple(extract_readings, _ENHANCE_MODE='auto', ROTATED=False, REPEAT_HINT=False,
                                 _FEWSHOTS=[]), mock.patch.object(detect_quality, 'ENHANCER', 'light'):
            self.assertEqual(settings_note(), '')

    def test_each_changed_setting_is_named(self):
        with mock.patch.multiple(extract_readings, _ENHANCE_MODE='whole', ROTATED=True, REPEAT_HINT=True,
                                 _FEWSHOTS=[{}] * 12), mock.patch.object(detect_quality, 'ENHANCER', 'glare-gmm'):
            self.assertEqual(settings_note(), '; enhance whole; enhancer glare-gmm; rotated boxes accepted; '
                                              'repeated-digit line; 4 example photos')


if __name__ == '__main__':
    unittest.main()
