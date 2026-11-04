"""The crop re-read: the box the first read gives, checked, cut from the photo and enlarged.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_readings  # noqa: E402
from extract_readings import _read_photo, crop_rect, extract_box  # noqa: E402


def setUpModule():  # these tests were written for a margin of 0.15
    patcher = mock.patch.object(extract_readings, 'CROP_MARGIN', 0.15)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class BoxTest(unittest.TestCase):
    def test_extract_box(self):
        self.assertEqual(extract_box('READING: 0205164\nBOX: 107 206 723 303\nQUALITY: CLEAR'), (107, 206, 723, 303))
        self.assertEqual(extract_box('BOX: 107, 206, 723, 303'), (107, 206, 723, 303))
        self.assertIsNone(extract_box('BOX: NONE'))
        self.assertIsNone(extract_box('READING: 0205164'))

    def test_good_box_with_margin(self):
        # 240x400 photo, box 107..723 x 206..303: 148x39 px, margin 0.15 of 148 = 22 px
        rect, why = crop_rect((107, 206, 723, 303), 240, 400)
        self.assertIsNone(why)
        self.assertEqual(rect, (3, 60, 196, 143))

    def test_margin_clipped_to_the_photo(self):
        rect, _ = crop_rect((0, 100, 1000, 440), 240, 400)  # a display as wide as the photo
        self.assertEqual((rect[0], rect[2]), (0, 240))

    def test_bad_boxes(self):
        for box, why in (((333, 98, 656, 400), 'box aspect'),       # #9 of the probe: most of the meter
                         ((900, 100, 800, 200), 'outside'),
                         ((100, 100, 120, 130), 'box width'),
                         ((100, 100, 900, 120), 'box height'),
                         ((0, 0, 1000, 900), 'box height'),
                         (None, 'no box')):
            rect, reason = crop_rect(box, 240, 400)
            self.assertIsNone(rect)
            self.assertIn(why, reason)


class CropReadTest(unittest.TestCase):
    """_read_photo up to the model request: what image is sent, and how it is labelled."""

    def sent(self, crop, box, scale=3.0):
        photo = np.full((400, 240, 3), 128, np.uint8)
        seen = {}

        def no_model(*args, **kwargs):
            raise RuntimeError('stop before the model')

        aids = []
        with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                mock.patch.object(extract_readings, 'png_data_url',
                                  side_effect=lambda pixels: seen.setdefault('shape', pixels.shape) and 'x'), \
                mock.patch.object(extract_readings, 'complete', side_effect=no_model):
            try:
                _read_photo(Path('x.jpg'), lambda label, work: work(), aids, [], [],
                            extract_readings.DEFAULT_BACKEND, scale, crop, box)
            except RuntimeError:
                pass
        return seen['shape'], aids

    def test_crop_is_650_wide(self):
        shape, aids = self.sent(True, (250, 300, 750, 540))  # 120x96 px, with margin 156x132: 650 wide is 550 high
        self.assertEqual(shape[1], 650)
        shape, _ = self.sent(True, (107, 206, 723, 303))  # 193x83 px: 650 wide is 280 high, so 480 high wins
        self.assertEqual(shape[:2], (480, 1116))
        self.assertTrue(aids[0].startswith('crop(') and aids[0].endswith('x)'), aids)

    def test_flat_crop_keeps_the_min_side(self):
        shape, _ = self.sent(True, (0, 450, 1000, 520))  # 240x28 px: 650 wide would be 76 high
        self.assertGreaterEqual(min(shape[:2]), extract_readings._MIN_SIDE - 1)

    def test_fallback_is_the_whole_frame_at_3x(self):
        shape, aids = self.sent(True, None)
        self.assertEqual(shape[:2], (1200, 720))
        self.assertEqual(aids[:2], ['crop-fallback(no box)', 'enlarged(3x)'])

    def test_enlarge_mode_unchanged(self):
        shape, aids = self.sent(False, None)
        self.assertEqual((shape[:2], aids[0]), ((1200, 720), 'enlarged(3x)'))


if __name__ == '__main__':
    unittest.main()
