"""FOTOMATES_KEEP=agree: the crop and the whole photo are both read; the digits two reads share
are kept, and reads that do not settle it keep the highest score, capped.

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
from audit_readings import read_best, read_details, reread_rule  # noqa: E402

PHOTO = Path('AL0010556_20261023092021_JURGENDU_1462__ExpAktiv_41-33_19-82.jpg')


def read(cells, score, sent, text=None):
    result = NS(text=text or (f'READING: {cells}' if cells else 'READING: UNREADABLE'), reading_seconds=1.0,
                digit_confidence=[0.9] * len(cells), digit_alternatives=[], image_report={}, device_type='digital',
                code_confidence=[], evidence_source=sent, clear=None, decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=1.0, whole=cells, decimals=0, match='no', parts=None)


class KeepAgreeTest(unittest.TestCase):
    def best(self, crop, whole, third=None, keep='agree', reread_scale=3.0):
        calls = []

        def scored(path, saved, model=None, scale=1.0, **kw):
            name = 'crop' if kw.get('crop') else 'whole' if scale == 1.0 else 'third'
            calls.append(name)
            given = dict(crop=crop, whole=whole, third=third)[name]
            if isinstance(given, Exception):
                raise given
            return dict(given)

        found = dict(box=(100, 300, 900, 500), why=None, answer='{"bbox_2d": [1, 2, 3, 4]}', seconds=0.5)
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), \
                mock.patch.object(audit_readings, 'KEEP', keep), \
                mock.patch.object(audit_readings, 'REREAD_MODE', 'enlarge'), \
                mock.patch.object(audit_readings, 'REREAD_SCALE', reread_scale), \
                mock.patch.object(audit_readings, 'blank_photo', return_value=None), \
                mock.patch.object(audit_readings, 'find_box', return_value=found), \
                mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            best = read_best(PHOTO, '1462')
        row = read_details(best['result'], best['cells'], best.get('reread'), best.get('other'), best.get('others'),
                           best.get('box_answer'), best.get('reads'), best.get('box_reason'))
        return best, calls, row

    def test_same_digits_keep_the_crop(self):
        best, calls, row = self.best(read('001462359', 0.6, 'crop(3x)'), read('001462359', 0.9, 'original'))
        self.assertEqual((calls, best['role'], best['score'], row['reads_agree']), (['crop', 'whole'], 'crop', 0.6, 1))
        self.assertEqual(best['seconds'], 2.5)  # the box request and two reads

    def test_a_sure_crop_is_still_checked(self):  # the photo of the brief: the crop was sure and wrong
        best, calls, row = self.best(read('000462359', 0.95, 'crop(3x)'), read('001462359', 0.6, 'original'),
                                     read('001462359', 0.5, 'enlarged(3x)'))
        self.assertEqual((calls, best['cells'], best['role']), (['crop', 'whole', 'third'], '001462359', 'whole'))
        self.assertEqual((best['score'], row['reads_agree'], row['other_reading']), (0.6, 0, '000462359'))
        self.assertEqual((row['crop_score'], row['whole_score'], row['third_score']), (0.95, 0.6, 0.5))

    def test_third_read_sides_with_the_crop(self):
        best, _, _ = self.best(read('01111', 0.5, 'crop(3x)'), read('0111', 0.9, 'original'),
                               read('01111', 0.4, 'enlarged(3x)'))
        self.assertEqual((best['cells'], best['role'], best['score']), ('01111', 'crop', 0.5))

    def test_no_two_alike_keeps_the_highest_score_capped(self):
        best, _, row = self.best(read('01111', 0.95, 'crop(3x)'), read('0111', 0.6, 'original'),
                                 read('0711', 0.5, 'enlarged(3x)'))
        self.assertEqual((best['cells'], best['score'], row['reads_agree'], row['crop_score']),
                         ('01111', 0.69, 0, 0.95))

    def test_a_low_score_is_not_raised_by_the_cap(self):
        best, _, _ = self.best(read('01111', 0.4, 'crop(3x)'), read('0111', 0.3, 'original'),
                               read('0711', 0.2, 'enlarged(3x)'))
        self.assertEqual(best['score'], 0.4)

    def test_no_third_read_when_it_is_off_or_fails(self):
        best, calls, _ = self.best(read('01111', 0.95, 'crop(3x)'), read('0111', 0.6, 'original'), reread_scale=1.0)
        self.assertEqual((calls, best['cells'], best['score']), (['crop', 'whole'], '01111', 0.69))
        best, calls, row = self.best(read('01111', 0.95, 'crop(3x)'), read('0111', 0.6, 'original'),
                                     RuntimeError('server said no'))
        self.assertEqual((calls[-1], best['cells'], best['score']), ('third', '01111', 0.69))
        self.assertEqual(row['whole_sent'], 'original+reread-failed')

    def test_a_read_without_digits_takes_no_part(self):
        # a sure crop and a whole photo that shows nothing readable: the crop stands, nothing to compare
        best, calls, row = self.best(read('01111', 0.9, 'crop(3x)'), read('', None, 'original'))
        self.assertEqual((calls, best['cells'], best['score'], row['reads_agree']),
                         (['crop', 'whole'], '01111', 0.9, None))
        # an unsure one is read once more, as under the score rule; two reads with digits that agree win
        best, calls, row = self.best(read('', None, 'crop(3x)'), read('01111', 0.5, 'original'),
                                     read('01111', 0.4, 'enlarged(3x)'))
        self.assertEqual((calls, best['role'], best['score'], row['reads_agree']),
                         (['crop', 'whole', 'third'], 'whole', 0.5, 1))
        # ... and two that differ are capped
        best, _, row = self.best(read('', None, 'crop(3x)'), read('01111', 0.5, 'original'),
                                 read('01711', 0.65, 'enlarged(3x)'))
        self.assertEqual((best['cells'], best['score'], row['reads_agree']), ('01711', 0.65, 0))

    def test_no_meter_is_not_read_a_third_time(self):
        _, calls, _ = self.best(read('', None, 'crop(3x)'), read('', None, 'original', 'READING: NO METER'))
        self.assertEqual(calls, ['crop', 'whole'])

    def test_score_rule_is_unchanged(self):
        best, calls, _ = self.best(read('000462359', 0.95, 'crop(3x)'), read('001462359', 0.6, 'original'),
                                   keep='score')
        self.assertEqual((calls, best['cells'], best['score']), (['crop'], '000462359', 0.95))

    def test_the_log_line_names_the_rule(self):
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), mock.patch.object(audit_readings, 'KEEP', 'agree'):
            self.assertIn('the digits two reads share kept', reread_rule())
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), mock.patch.object(audit_readings, 'KEEP', 'score'):
            self.assertIn('the most confident read kept', reread_rule())


if __name__ == '__main__':
    unittest.main()
