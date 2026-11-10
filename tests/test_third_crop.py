"""With crop first the read after the whole photo is a second crop: cut from the box the
whole-photo read wrote on its own BOX line, with the wider THIRD_MARGIN.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
import extract_readings  # noqa: E402
from audit_readings import read_best, read_details, reread_rule  # noqa: E402
from extract_readings import _read_photo, crop_rect, system_prompt  # noqa: E402

PHOTO = Path('AL0010556_20261023092021_JURGENDU_1462__ExpAktiv_41-33_19-82.jpg')


def read(cells, score, sent, box=''):
    result = NS(text=f'READING: {cells}' + box, reading_seconds=1.0, digit_confidence=[0.9] * len(cells),
                digit_alternatives=[], image_report={}, device_type='digital', code_confidence=[],
                evidence_source=sent, clear=None, decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=1.0, whole=cells, decimals=0, match='no', parts=None)


class SettingsTest(unittest.TestCase):
    def test_the_default_follows_crop_first(self):
        if 'FOTOMATES_REREAD_MODE' not in os.environ:
            self.assertEqual(extract_readings.REREAD_MODE, 'crop' if extract_readings.CROP_FIRST else 'enlarge')
        self.assertEqual(extract_readings.THIRD_MARGIN, float(os.environ.get('FOTOMATES_THIRD_MARGIN', '0.5')))

    def test_only_the_whole_photo_is_asked_for_the_box(self):
        with mock.patch.object(extract_readings, 'REREAD_MODE', 'crop'):
            self.assertIn('BOX: x1 y1 x2 y2', system_prompt())
            self.assertNotIn('BOX: x1 y1 x2 y2', system_prompt(crop=True))
            lines = 'eight' if 'seven lines' in extract_readings._SYSTEM_PROMPT else 'six'
            self.assertIn(f'exactly these {lines} lines', system_prompt())
        with mock.patch.object(extract_readings, 'REREAD_MODE', 'enlarge'):
            self.assertEqual(system_prompt(), extract_readings._SYSTEM_PROMPT)
            self.assertEqual(system_prompt(crop=True), extract_readings._SYSTEM_PROMPT)

    def test_the_log_line_names_the_crop(self):
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), mock.patch.object(audit_readings, 'REREAD_MODE', 'crop'):
            self.assertIn("from a crop of the whole-photo read's own box, margin 0.5 (3x without a usable box)",
                          reread_rule())
            with mock.patch.object(audit_readings, 'KEEP', 'agree'):
                self.assertIn('a third read from a crop of the whole-photo', reread_rule())
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), \
                mock.patch.object(audit_readings, 'REREAD_MODE', 'enlarge'):
            self.assertIn('reread below 0.7 at 3x', reread_rule())
        with mock.patch.object(audit_readings, 'CROP_FIRST', False), \
                mock.patch.object(audit_readings, 'REREAD_MODE', 'crop'):
            self.assertEqual(reread_rule(), 'reread below 0.7 from a crop (3x without a usable box)')


class MarginTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(extract_readings, 'CROP_MARGIN', 0.25)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_wider_margin_cuts_more(self):
        # 240x400 photo, box 250..750 x 300..540: 120x96 px; margin 0.25 of 120 = 30 px, 0.5 = 60 px
        self.assertEqual(crop_rect((250, 300, 750, 540), 240, 400), ((30, 90, 210, 246), None))
        self.assertEqual(crop_rect((250, 300, 750, 540), 240, 400, 0.5), ((0, 60, 240, 276), None))
        self.assertEqual(crop_rect((250, 300, 750, 540), 240, 400, 0), ((60, 120, 180, 216), None))

    def test_a_refused_box_is_refused_with_any_margin(self):
        self.assertEqual(crop_rect((100, 100, 200, 900), 240, 400, 0.5)[0], None)

    def sent(self, box, margin):
        photo = np.full((400, 240, 3), 128, np.uint8)
        seen, aids = {}, []

        def no_model(*args, **kwargs):
            seen['system'] = kwargs['messages'][0]['content']
            raise RuntimeError('stop before the model')

        with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                mock.patch.object(extract_readings, 'REREAD_MODE', 'crop'), \
                mock.patch.object(extract_readings, 'png_data_url',
                                  side_effect=lambda pixels: seen.setdefault('shape', pixels.shape) and 'x'), \
                mock.patch.object(extract_readings, 'complete', side_effect=no_model):
            try:
                _read_photo(Path('x.jpg'), lambda label, work: work(), aids, [], [],
                            extract_readings.DEFAULT_BACKEND, 3.0, True, box, margin)
            except RuntimeError:
                pass
        return seen, aids

    def test_the_read_cuts_with_the_margin_given(self):
        usual, aids = self.sent((250, 300, 750, 540), None)   # 180x156 px cut, 650 wide
        wide, _ = self.sent((250, 300, 750, 540), 0.5)        # 240x216 px cut, 650 wide
        self.assertEqual((usual['shape'][1], wide['shape'][1]), (650, 650))
        self.assertGreater(wide['shape'][0], usual['shape'][0])  # more of the photo above and below
        self.assertTrue(aids[0].startswith('crop('), aids)
        self.assertNotIn('BOX: x1 y1 x2 y2', wide['system'])     # a crop is not asked where the display is

    def test_without_a_box_it_is_the_whole_photo_at_3x(self):
        seen, aids = self.sent(None, 0.5)
        self.assertEqual((seen['shape'][:2], aids[:2]), ((1200, 720), ['crop-fallback(no box)', 'enlarged(3x)']))


class ThirdReadTest(unittest.TestCase):
    def best(self, crop, whole, third, keep='score', crop_first=True):
        calls = []

        def scored(path, saved, model=None, scale=1.0, **kw):
            name = 'whole' if not kw.get('crop') else 'third' if 'whole' in [c[0] for c in calls] else 'crop'
            calls.append((name, scale, kw.get('box'), kw.get('margin')))
            return dict(dict(crop=crop, whole=whole, third=third)[name])

        found = dict(box=(100, 300, 900, 500), why=None, answer='{"bbox_2d": [100, 300, 900, 500]}', seconds=0.5)
        with mock.patch.object(audit_readings, 'CROP_FIRST', crop_first), \
                mock.patch.object(audit_readings, 'KEEP', keep), \
                mock.patch.object(audit_readings, 'REREAD_MODE', 'crop'), \
                mock.patch.object(audit_readings, 'REREAD_SCALE', 3.0), \
                mock.patch.object(audit_readings, 'THIRD_MARGIN', 0.5), \
                mock.patch.object(audit_readings, 'blank_photo', return_value=None), \
                mock.patch.object(audit_readings, 'find_box', return_value=found), \
                mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            best = read_best(PHOTO, '1462')
        row = read_details(best['result'], best['cells'], best.get('reread'), best.get('other'), best.get('others'),
                           best.get('box_answer'), best.get('reads'), best.get('box_reason'))
        return best, calls, row

    def test_the_third_read_is_cut_from_the_whole_reads_box(self):
        best, calls, row = self.best(read('01111', 0.5, 'crop(3x)'),
                                     read('0111', 0.6, 'original', '\nBOX: 120 310 880 480\nQUALITY: 3'),
                                     read('011112', 0.65, 'crop(2.9x)'))
        self.assertEqual(calls, [('crop', 1.0, (100, 300, 900, 500), None), ('whole', 1.0, None, None),
                                 ('third', 3.0, (120.0, 310.0, 880.0, 480.0), 0.5)])
        self.assertEqual((best['cells'], best['role'], row['third_sent'], row['crop_digits']),
                         ('011112', 'third', 'crop(2.9x)', '01111'))

    def test_without_a_box_line_the_read_falls_back(self):  # box None: _read_photo sends the 3x frame
        _, calls, _ = self.best(read('01111', 0.5, 'crop(3x)'), read('0111', 0.6, 'original'),
                                read('0111', 0.65, 'crop-fallback(no box)+enlarged(3x)'))
        self.assertEqual(calls[-1], ('third', 3.0, None, 0.5))

    def test_the_agree_rule_votes_with_it(self):
        best, calls, row = self.best(read('01111', 0.95, 'crop(3x)'),
                                     read('011112', 0.6, 'original', '\nBOX: 120 310 880 480'),
                                     read('011112', 0.5, 'crop(2.9x)'), keep='agree')
        self.assertEqual((calls[-1], best['cells'], best['role']),
                         (('third', 3.0, (120.0, 310.0, 880.0, 480.0), 0.5), '011112', 'whole'))

    def test_a_sure_whole_photo_is_not_read_again(self):
        _, calls, _ = self.best(read('01111', 0.5, 'crop(3x)'), read('0111', 0.9, 'original', '\nBOX: 1 2 3 4'), None)
        self.assertEqual([c[0] for c in calls], ['crop', 'whole'])

    def test_whole_photo_first_keeps_the_usual_margin(self):
        _, calls, _ = self.best(None, read('0111', 0.6, 'original', '\nBOX: 120 310 880 480'),
                                read('01111', 0.65, 'crop(2.9x)'), crop_first=False)
        self.assertEqual(calls, [('whole', 1.0, None, None), ('third', 3.0, (120.0, 310.0, 880.0, 480.0), None)])


if __name__ == '__main__':
    unittest.main()
