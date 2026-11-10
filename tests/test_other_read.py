"""After two reads of a photo, the one not kept is saved too.

    python -m unittest discover tests
"""
import os
import random
import sqlite3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
from audit_readings import (COLUMNS, _saved_match, connect, read_best, read_details,  # noqa: E402
                            split_decimals, summary)


_ENLARGE = mock.patch.object(audit_readings, 'REREAD_MODE', 'enlarge')  # here the read after the whole photo is the 3x one


def setUpModule():
    _ENLARGE.start()


def tearDownModule():
    _ENLARGE.stop()


def read(cells, score, scale, text):
    result = NS(text=text, reading_seconds=1.0, digit_confidence=[0.9] * len(cells), digit_alternatives=[],
                image_report={}, device_type='digital', code_confidence=[], evidence_source='original')
    return dict(result=result, cells=cells, score=score, scale=scale, whole=cells, decimals=0, match='no',
                parts=None)


class OtherReadTest(unittest.TestCase):
    def setUp(self):  # the whole photo first: the crop-first order is tested in test_crop_first
        for name, value in (('CROP_FIRST', False), ('blank_photo', lambda path: None)):  # no photo file here
            patcher = mock.patch.object(audit_readings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def reads(self, first, second):
        return mock.patch.object(audit_readings, 'read_scored',
                                 side_effect=lambda path, saved, model=None, scale=1.0: first if scale == 1.0 else second)

    def test_second_read_kept_first_saved(self):
        first, second = read('0111', 0.4, 1.0, 'READING: 0111'), read('01111', 0.8, 3.0, 'READING: 01111')
        with self.reads(first, second), mock.patch.object(audit_readings, 'REREAD_SCALE', 3.0):
            best = read_best(Path('x.jpg'), '1111')
        self.assertEqual(best['cells'], '01111')
        details = read_details(best['result'], best['cells'], best['reread'], best['other'])
        self.assertEqual((details['other_reading'], details['other_score'], details['other_scale'],
                          details['reads_agree']), ('0111', 0.4, '1', 0))
        self.assertEqual(details['model_answer'], 'READING: 01111\n--- other read (1) ---\nREADING: 0111')

    def test_first_read_kept_reads_agree(self):
        first, second = read('0205', 0.6, 1.0, 'a'), read('0205', 0.5, 3.0, 'b')
        with self.reads(first, second), mock.patch.object(audit_readings, 'REREAD_SCALE', 3.0):
            best = read_best(Path('x.jpg'), '205')
        details = read_details(best['result'], best['cells'], best['reread'], best['other'])
        self.assertEqual((details['other_scale'], details['reads_agree']), ('3', 1))

    def test_one_read_leaves_other_empty(self):
        first = read('0205', 0.95, 1.0, 'a')
        with self.reads(first, None):
            best = read_best(Path('x.jpg'), '205')
        details = read_details(best['result'], best['cells'], best['reread'], best.get('other'))
        self.assertEqual([details[k] for k in ('other_reading', 'other_score', 'other_scale', 'reads_agree')],
                         [None] * 4)
        self.assertEqual(details['model_answer'], 'a')

    def test_sql_match_is_split_decimals(self):
        db = sqlite3.connect(':memory:')
        rnd = random.Random(1)
        for _ in range(3000):
            saved = rnd.choice(['0', '5', '1605', '205164', '00', str(rnd.randint(0, 99999))])
            digits = ''.join(rnd.choice('0123456789') for _ in range(rnd.randint(1, 9)))
            if rnd.random() < 0.3:  # a read that does give the saved number
                digits = '0' * rnd.randint(0, 3) + saved.lstrip('0') + ''.join(rnd.choice('0123') for _ in range(rnd.randint(0, 2)))
                digits = digits or '0'
            got = db.execute(f"SELECT {_saved_match('d')} FROM (SELECT ? AS d, ? AS saved_reading)",
                             [digits, saved]).fetchone()[0]
            self.assertEqual(bool(got), split_decimals(digits, saved)[2] == 'yes', (digits, saved))

    def test_summary_counts(self):
        db = connect(':memory:')
        rows = [  # saved 1111: first read 0111 (no), second 01111 (yes), kept second
            dict(picture_name='a', month='2026-10', saved_reading='1111', extracted_reading='01111', other_reading='0111',
                 other_scale='1', reads_agree=0, status='done', match='yes'),
            # saved 205: first 0205 kept (yes), second 0206 (no)
            dict(picture_name='b', month='2026-10', saved_reading='205', extracted_reading='0205', other_reading='0206',
                 other_scale='3', reads_agree=0, status='done', match='yes'),
            dict(picture_name='c', month='2026-10', saved_reading='7', extracted_reading='07', other_reading='07',
                 other_scale='3', reads_agree=1, status='done', match='yes'),
            dict(picture_name='d', month='2026-10', saved_reading='7', extracted_reading='07', status='done', match='yes'),
        ]
        for row in rows:
            row['read_at'] = '2026-11-04T10:00:00+00:00'
            db.execute(f'INSERT INTO readings ({",".join(row)}) VALUES ({",".join("?" * len(row))})', list(row.values()))
        cursor = summary(db)
        total = dict(zip([c[0] for c in cursor.description], cursor.fetchone()))
        self.assertEqual([total[k] for k in ('reread_photos', 'reread_agree', 'reread_differ',
                                             'differ_first_matched', 'differ_second_matched')], [3, 1, 2, 1, 1])
        self.assertTrue({'other_reading', 'other_score', 'other_scale', 'reads_agree'} <= set(COLUMNS))


if __name__ == '__main__':
    unittest.main()
