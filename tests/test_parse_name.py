"""Photo names are read by position: Contract_Time_Reader_Reading_AnomalyCode_Register_Lat_Lon.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path

# audit_readings needs a model server configured to import; these tests never call it.
os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_readings import parse_name  # noqa: E402


class ParseNameTest(unittest.TestCase):
    def test_reading_without_anomaly(self):
        row = parse_name('AL0010556_20261023092021_JURGENDU_12345__ExpAktiv_41-33_19-82.jpg')
        self.assertEqual(row, dict(contract_id='AL0010556', date='2026-10-23', month='2026-10',
                                   register='ExpAktiv', saved_reading='12345', operator='JURGENDU',
                                   anomaly_code=None, latitude=41.33, longitude=19.82))

    def test_reading_with_anomaly(self):
        row = parse_name('AL0001116_20261023203720_OLSIIB_50_57-5701_ExpAmbPerbashket_41-47428715_19-55966892.jpg')
        self.assertEqual(row['contract_id'], 'AL0001116')
        self.assertEqual(row['date'], '2026-10-23')
        self.assertEqual(row['operator'], 'OLSIIB')
        self.assertEqual(row['saved_reading'], '50')  # kept although an anomaly is filled
        self.assertEqual(row['anomaly_code'], '57-5701')
        self.assertEqual(row['register'], 'ExpAmbPerbashket')
        self.assertEqual((row['latitude'], row['longitude']), (41.47428715, 19.55966892))

    def test_anomaly_without_reading(self):
        row = parse_name('XX0000002_20261023105104_READERB__51-5101_Aktiv_40-11111111_19-22222222.jpg')
        self.assertIsNone(row['saved_reading'])
        self.assertEqual(row['anomaly_code'], '51-5101')
        self.assertEqual(row['register'], 'Aktiv')
        self.assertEqual(row['operator'], 'READERB')

    def test_twenty_digit_time_stamp(self):
        row = parse_name('AL0010556_20261023092021123456_JURGENDU_12345__ExpAktiv_41-33_19-82.jpg')
        self.assertEqual((row['date'], row['month']), ('2026-10-23', '2026-10'))
        self.assertEqual(row['saved_reading'], '12345')

    def test_no_coordinates(self):
        row = parse_name('XX0000001_20261023091345_READERA_0__ExpAktiv_0-0_0-0.jpg')
        self.assertIsNone(row['latitude'])
        self.assertIsNone(row['longitude'])
        self.assertEqual(row['saved_reading'], '0')
        self.assertEqual(row["operator"], "READERA")

    def test_reader_with_underscore(self):
        row = parse_name('AL0010556_20261023092021_JURGEN_DU_12345__ExpAktiv_41-33_19-82.jpg')
        self.assertEqual(row['operator'], 'JURGEN_DU')
        self.assertEqual(row['saved_reading'], '12345')

    def test_unreadable_date_uses_month_given(self):
        row = parse_name('AL0010556_2026XX23092021_JURGENDU_12345__ExpAktiv_41-33_19-82.jpg', '2026-10')
        self.assertIsNone(row['date'])
        self.assertEqual(row['month'], '2026-10')
        self.assertEqual(row['saved_reading'], '12345')

    def test_short_name(self):
        row = parse_name('AL0010556_20261023092021.jpg', '2026-10')
        self.assertEqual((row['contract_id'], row['date'], row['saved_reading']), ('AL0010556', '2026-10-23', None))


if __name__ == '__main__':
    unittest.main()
