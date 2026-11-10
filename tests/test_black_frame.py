"""A black frame is answered BLANK before any model request, and every read made is saved
with its own digits, probabilities, score and label.

    python -m unittest discover tests
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
import extract_readings  # noqa: E402
from audit_readings import COLUMNS, DETAIL_COLUMNS, connect, read_best, read_details, read_one, save  # noqa: E402
from extract_readings import _read_photo, blank_photo  # noqa: E402

NAME = 'AL0010556_20261023092021_JURGENDU_1111__ExpAktiv_41-33_19-82.jpg'


_ENLARGE = mock.patch.object(audit_readings, 'REREAD_MODE', 'enlarge')  # here the read after the whole photo is the 3x one


def setUpModule():
    _ENLARGE.start()


def tearDownModule():
    _ENLARGE.stop()


def read(cells, score, sent, chances=None):
    result = NS(text=f'READING: {cells}', reading_seconds=1.0,
                digit_confidence=chances or [0.9] * len(cells), digit_alternatives=[], image_report={},
                device_type='digital', code_confidence=[], evidence_source=sent, clear=None, decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=1.0, whole=cells, decimals=0, match='no', parts=None)


class BlackFrameTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def photo(self, pixels):
        path = Path(self.folder.name) / NAME
        cv2.imwrite(str(path), pixels)
        return path

    def test_black_frame_is_blank_without_the_model(self):
        noise = np.random.default_rng(1).integers(0, 4, (400, 240, 3)).astype(np.uint8)  # mean 1.5, std 1.1
        path = self.photo(noise)
        with mock.patch.object(audit_readings, 'find_box') as box, \
                mock.patch.object(audit_readings, 'read_scored') as scored, \
                mock.patch.object(extract_readings, 'complete') as model:
            row = read_one(path, '2026-10')
        self.assertEqual((box.call_count, scored.call_count, model.call_count), (0, 0, 0))
        self.assertEqual((row['status'], row['extracted_reading'], row['score'], row['photo_sent']),
                         ('blank', None, None, 'blank-check'))
        self.assertEqual((row['match'], row['reread'], row['reads_agree'], row['error']), ('no', 0, None, None))
        self.assertEqual((row['image_width'], row['image_height']), (240, 400))
        self.assertLess(row['brightness'], 5)
        self.assertLess(row['contrast'], 5)
        db = connect(':memory:')
        save(db, row)  # every value fits its column
        self.assertEqual(db.execute('SELECT status, photo_sent FROM readings').fetchone(), ('blank', 'blank-check'))

    def test_other_photos_are_read(self):
        dark = np.full((400, 240, 3), 3, np.uint8)
        dark[150:250, 40:200] = 60  # dark, yet something is on it: contrast above the limit
        for pixels in (dark, np.full((400, 240, 3), 128, np.uint8)):
            self.assertIsNone(blank_photo(self.photo(pixels)))

    def test_limit_from_the_environment(self):
        path = self.photo(np.zeros((400, 240, 3), np.uint8))
        self.assertEqual(blank_photo(path).evidence_source, 'blank-check')
        with mock.patch.object(extract_readings, 'BLANK_MAX', 0.0):  # FOTOMATES_BLANK_MAX=0: off
            self.assertIsNone(blank_photo(path))

    def test_a_photo_that_cannot_be_judged_is_read(self):
        whole = read('01111', 0.9, 'original')
        with mock.patch.object(audit_readings, 'CROP_FIRST', False), \
                mock.patch.object(audit_readings, 'blank_photo', side_effect=ValueError('not an image')), \
                mock.patch.object(audit_readings, 'read_scored', return_value=whole):
            self.assertEqual(read_best(Path(NAME), '1111')['cells'], '01111')

    def test_black_crop_is_not_brightened(self):
        photo = np.full((400, 240, 3), 128, np.uint8)
        photo[40:360] = 2  # the display cut out, margin and all, is black

        def no_model(*args, **kwargs):
            raise RuntimeError('stop before the model')

        aids = []
        with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                mock.patch.object(extract_readings, '_ENHANCE_MODE', 'auto'), \
                mock.patch.object(extract_readings, 'enhance') as enhance, \
                mock.patch.object(extract_readings, 'complete', side_effect=no_model):
            try:
                _read_photo(Path('x.jpg'), lambda label, work: work(), aids, [], [],
                            extract_readings.DEFAULT_BACKEND, 1.0, True, (100, 300, 900, 600))
            except RuntimeError:
                pass
        self.assertEqual(enhance.call_count, 0)
        self.assertIn('enhance-skipped(black)', aids)


class EveryReadTest(unittest.TestCase):
    def best(self, crop, whole=None, third=None, box=(100, 300, 900, 500), why=None, crop_error=None):
        def scored(path, saved, model=None, scale=1.0, **kw):
            if kw.get('crop'):
                if crop_error:
                    raise crop_error
                return dict(crop)
            return dict(whole if scale == 1.0 else third)

        found = dict(box=box, why=why, answer='{"bbox_2d": [1, 2, 3, 4]}', seconds=0.5)
        with mock.patch.object(audit_readings, 'CROP_FIRST', True), \
                mock.patch.object(audit_readings, 'REREAD_SCALE', 3.0), \
                mock.patch.object(audit_readings, 'blank_photo', return_value=None), \
                mock.patch.object(audit_readings, 'find_box', return_value=found), \
                mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            return read_best(Path(NAME), '1111')

    @staticmethod
    def details(best):
        return read_details(best['result'], best['cells'], best.get('reread'), best.get('other'), best.get('others'),
                            best.get('box_answer'), best.get('reads'), best.get('box_reason'))

    def test_three_reads_each_in_its_columns(self):
        best = self.best(read('01111', 0.5, 'crop(3x)', [0.9, 0.8, 0.7, 0.6, 0.5]), read('0111', 0.4, 'original'),
                         read('01111', 0.6, 'enlarged(3x)'))
        row = self.details(best)
        self.assertEqual((row['crop_digits'], row['crop_score'], row['crop_sent'], row['crop_probabilities']),
                         ('01111', 0.5, 'crop(3x)', '0.900 0.800 0.700 0.600 0.500'))
        self.assertEqual((row['whole_digits'], row['whole_score'], row['whole_sent']), ('0111', 0.4, 'original'))
        self.assertEqual((row['third_digits'], row['third_score'], row['third_sent']), ('01111', 0.6, 'enlarged(3x)'))
        self.assertEqual(row['reads_agree'], 0)  # two of the three agree, the whole photo does not
        self.assertEqual((row['box_answer'], row['box_reason']), ('{"bbox_2d": [1, 2, 3, 4]}', None))
        self.assertEqual(best['cells'], '01111')  # the kept read is still the highest score

    def test_all_reads_agree(self):
        row = self.details(self.best(read('0205', 0.5, 'crop(3x)'), read('0205', 0.4, 'original'),
                                     read('0205', 0.6, 'enlarged(3x)')))
        self.assertEqual(row['reads_agree'], 1)

    def test_one_read_leaves_the_rest_empty(self):
        row = self.details(self.best(read('01111', 0.9, 'crop(3x)')))
        self.assertEqual((row['crop_digits'], row['crop_score'], row['reads_agree']), ('01111', 0.9, None))
        for name in ('whole_digits', 'whole_score', 'whole_sent', 'third_digits', 'third_probabilities'):
            self.assertIsNone(row.get(name), name)

    def test_refused_box_gives_its_reason(self):
        row = self.details(self.best(None, read('01111', 0.9, 'original'), box=None, why='box aspect 0.61'))
        self.assertEqual((row['box_reason'], row.get('crop_digits'), row['whole_digits']),
                         ('box aspect 0.61', None, '01111'))

    def test_failed_crop_read_gives_its_reason(self):
        row = self.details(self.best(None, read('01111', 0.9, 'original'), crop_error=RuntimeError('server said no')))
        self.assertEqual(row['box_reason'], 'crop read failed: RuntimeError: server said no')

    def test_failed_second_read_is_labelled(self):
        first = read('0111', 0.4, 'original')

        def scored(path, saved, model=None, scale=1.0):
            if scale != 1.0:
                raise RuntimeError('server said no')
            return first

        with mock.patch.object(audit_readings, 'CROP_FIRST', False), \
                mock.patch.object(audit_readings, 'REREAD_SCALE', 3.0), \
                mock.patch.object(audit_readings, 'blank_photo', return_value=None), \
                mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            best = read_best(Path(NAME), '1111')
        row = self.details(best)
        self.assertEqual((row['photo_sent'], row['whole_sent'], row.get('third_digits'), row['reads_agree']),
                         ('original+reread-failed', 'original+reread-failed', None, None))

    def test_old_calls_still_work(self):  # web_app and older callers pass no reads
        row = read_details(read('01111', 0.9, 'original')['result'], '01111', False)
        self.assertEqual((row['reads_agree'], row['box_answer'], row.get('crop_digits')), (None, None, None))

    def test_old_database_gets_the_columns(self):
        new = {name for name, _ in DETAIL_COLUMNS if name.split('_')[0] in ('crop', 'whole', 'third', 'box')}
        self.assertEqual(len(new), 14)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'old.sqlite3')
            connect(path).close()
            db = sqlite3.connect(path)
            db.execute('DROP TRIGGER readings_record_attempt')
            for table in ('readings', 'reading_attempts'):
                for name in new:
                    db.execute(f'ALTER TABLE {table} DROP COLUMN {name}')
            db.commit()
            db.close()
            db = connect(path)
            for table in ('readings', 'reading_attempts'):
                self.assertLessEqual(new, {column[1] for column in db.execute(f'PRAGMA table_info({table})')})
            save(db, dict.fromkeys(COLUMNS) | dict(picture_name='a.jpg', match='no', status='done', crop_digits='01111'))
            self.assertEqual(db.execute('SELECT crop_digits FROM reading_attempts').fetchone(), ('01111',))
            db.close()


if __name__ == '__main__':
    unittest.main()
