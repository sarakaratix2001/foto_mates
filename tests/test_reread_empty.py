"""FOTOMATES_REREAD_EMPTY=0: a photo whose first read gave no digits is not read again.

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
from audit_readings import read_best, reread_rule  # noqa: E402

PHOTO = Path('AL0010556_20261023092021_JURGENDU_1462__ExpAktiv_41-33_19-82.jpg')
BOX = dict(box=(100, 100, 600, 300), why=None, answer='100,100,600,300', seconds=1.0)


def read(cells, score, sent, text=None):
    result = NS(text=text or f'READING: {cells or "UNREADABLE"}', reading_seconds=1.0,
                digit_confidence=[0.9] * len(cells), digit_alternatives=[], image_report={},
                device_type='digital', code_confidence=[], evidence_source=sent, clear=None, decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=1.0, whole=cells, decimals=0, match='no', parts=None)


def run(reads, box=BOX, **settings):
    """read_best with the reads given in order; returns (kept, number of reads made)."""
    made = []

    def scored(path, saved, model=None, scale=1.0, crop=False, box=None, margin=None):
        made.append((scale, crop))
        return dict(reads[len(made) - 1])

    values = dict(CROP_FIRST=True, KEEP='score', REREAD_EMPTY=False, REREAD_SCALE=3.0, REREAD_MODE='enlarge')
    values.update(settings)
    with mock.patch.multiple(audit_readings, read_scored=scored, find_box=lambda path, model=None: dict(box),
                             _black_frame=lambda path: None, **values):
        return read_best(PHOTO, '1462'), len(made)


class RereadEmptyTest(unittest.TestCase):
    def test_an_empty_crop_read_stands(self):
        kept, made = run([read('', None, 'crop(2.7x)'), read('001462', 0.9, 'enlarged(2x)')])
        self.assertEqual((made, kept['cells'], kept['reread']), (1, '', False))
        self.assertEqual([r['role'] for r in kept['reads']], ['crop'])
        self.assertEqual(kept['box_answer'], BOX['answer'])

    def test_as_before_when_the_setting_is_on(self):
        kept, made = run([read('', None, 'crop(2.7x)'), read('001462', 0.9, 'enlarged(2x)')], REREAD_EMPTY=True)
        self.assertEqual((made, kept['cells']), (2, '001462'))

    def test_an_empty_crop_read_stands_under_agree(self):
        kept, made = run([read('', None, 'crop(2.7x)'), read('001462', 0.9, 'enlarged(2x)')], KEEP='agree')
        self.assertEqual((made, kept['cells']), (1, ''))

    def test_an_unsure_crop_with_digits_is_still_read_again(self):
        kept, made = run([read('001462', 0.4, 'crop(2.7x)'), read('001462', 0.9, 'enlarged(2x)')])
        self.assertEqual((made, kept['score']), (2, 0.9))

    def test_without_a_box_an_empty_whole_read_stands(self):
        none = dict(box=None, why='no box', answer='NONE', seconds=1.0)
        kept, made = run([read('', None, 'enlarged(2x)'), read('001462', 0.9, 'enlarged(3x)')], box=none)
        self.assertEqual((made, kept['cells'], kept['reread']), (1, '', False))

    def test_without_a_box_an_unsure_whole_read_is_read_again(self):
        none = dict(box=None, why='no box', answer='NONE', seconds=1.0)
        kept, made = run([read('001462', 0.4, 'enlarged(2x)'), read('001462', 0.9, 'enlarged(3x)')], box=none)
        self.assertEqual((made, kept['score']), (2, 0.9))

    def test_the_rule_says_so(self):
        with mock.patch.multiple(audit_readings, REREAD_EMPTY=False, CROP_FIRST=True, KEEP='score'):
            self.assertIn('without digits is not read again', reread_rule())
        with mock.patch.multiple(audit_readings, REREAD_EMPTY=True, CROP_FIRST=True, KEEP='score'):
            self.assertNotIn('without digits', reread_rule())


if __name__ == '__main__':
    unittest.main()
