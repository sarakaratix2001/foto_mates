"""Crop first: the crop is kept at confidence 0.7 or more; else the whole photo (and 3x when
still unsure) is read too, and the most confident of all the reads is kept.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
from audit_readings import read_best, read_details  # noqa: E402

PHOTO = Path('AL0010556_20261023092021_JURGENDU_1111__ExpAktiv_41-33_19-82.jpg')


def read(cells, score, scale):
    result = NS(text=f'READING: {cells}', reading_seconds=1.0, digit_confidence=[0.9] * len(cells),
                digit_alternatives=[], image_report={}, device_type='digital', code_confidence=[],
                evidence_source=str(scale), clear=None, decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=scale, whole=cells, decimals=0, match='no', parts=None)


class CropFirstTest(unittest.TestCase):
    def run_reads(self, crop=None, whole=None, enlarged=None, box=(100, 300, 900, 500), reread_scale=3.0):
        calls = []

        def scored(path, saved, model=None, scale=1.0, crop_=False, box=None, **kw):
            is_crop = kw.get('crop', crop_)
            calls.append('crop' if is_crop else f'{scale:g}x')
            return dict(crop if is_crop else whole if scale == 1.0 else enlarged)

        found = dict(box=box, why=None if box else 'no box', answer='{"bbox_2d": [100, 300, 900, 500]}', seconds=0.5)
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), \
                mock.patch.object(audit_readings, 'REREAD_SCALE', reread_scale), \
                mock.patch.object(audit_readings, 'find_box', return_value=found), \
                mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            return read_best(PHOTO, '1111'), calls

    def test_sure_crop_is_kept_alone(self):
        best, calls = self.run_reads(crop=read('01111', 0.8, 'crop(3x)'))
        self.assertEqual((best['cells'], calls, best['reread']), ('01111', ['crop'], False))
        self.assertEqual(best['seconds'], 1.5)  # the box request and the crop read

    def test_unsure_crop_then_sure_whole_photo(self):
        best, calls = self.run_reads(crop=read('0111', 0.5, 'crop(3x)'), whole=read('01111', 0.9, 1.0))
        self.assertEqual((best['cells'], calls), ('01111', ['crop', '1x']))
        self.assertEqual(best['other']['cells'], '0111')

    def test_unsure_crop_stays_when_more_confident_than_the_whole_photo(self):
        best, calls = self.run_reads(crop=read('01111', 0.6, 'crop(3x)'), whole=read('0111', 0.4, 1.0),
                                     enlarged=read('011', 0.2, 3.0))
        self.assertEqual((best['cells'], calls), ('01111', ['crop', '1x', '3x']))
        self.assertEqual([o['cells'] for o in best['others']], ['0111', '011'])

    def test_all_three_unsure_highest_kept(self):
        best, calls = self.run_reads(crop=read('0111', 0.3, 'crop(3x)'), whole=read('01111', 0.4, 1.0),
                                     enlarged=read('011111', 0.6, 3.0))
        self.assertEqual((best['cells'], calls), ('011111', ['crop', '1x', '3x']))
        details = read_details(best['result'], best['cells'], best['reread'], best['other'], best['others'],
                               best['box_answer'])
        self.assertEqual(details['other_reading'], '01111')
        answer = details['model_answer']
        self.assertIn('--- other read (crop(3x)) ---', answer)
        self.assertIn('--- other read (1) ---', answer)
        self.assertTrue(answer.endswith('--- box (own request) ---\n{"bbox_2d": [100, 300, 900, 500]}'))

    def test_whole_photo_sure_enough_with_crop_no_3x(self):
        best, calls = self.run_reads(crop=read('0111', 0.2, 'crop(3x)'), whole=read('01111', 0.75, 1.0))
        self.assertEqual(calls, ['crop', '1x'])

    def test_no_box_reads_the_whole_photo(self):
        best, calls = self.run_reads(box=None, whole=read('01111', 0.9, 1.0))
        self.assertEqual((best['cells'], calls, best['reread']), ('01111', ['1x'], False))

    def test_reread_off_on_the_server(self):
        best, calls = self.run_reads(crop=read('0111', 0.3, 'crop(3x)'), whole=read('01111', 0.4, 1.0),
                                     reread_scale=1)
        self.assertEqual((best['cells'], calls), ('01111', ['crop', '1x']))

    def test_box_request_fails_reads_the_whole_photo(self):
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), \
                mock.patch.object(audit_readings, 'find_box', side_effect=RuntimeError('model gone')), \
                mock.patch.object(audit_readings, 'read_scored', return_value=read('01111', 0.9, 1.0)):
            self.assertEqual(read_best(PHOTO, '1111')['cells'], '01111')


if __name__ == '__main__':
    unittest.main()
