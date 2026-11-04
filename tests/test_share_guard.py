"""A photo share that stops answering pauses the run instead of failing its photos.

    python -m unittest discover tests
"""
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
from audit_readings import ShareGuard, read_with_retries  # noqa: E402
from extract_readings import OcrFailure, PhotoFileError  # noqa: E402

PHOTO = Path('Z:/share/2026/10/23/AL0010556_20261023092021_JURGENDU_12345__ExpAktiv_41-33_19-82.jpg')


def share_error():
    """A file error as the read raises it: wrapped in OcrFailure."""
    try:
        raise PhotoFileError(None, 'OSError: [WinError 64] The specified network name is no longer available',
                             str(PHOTO))
    except PhotoFileError as error:
        try:
            raise OcrFailure('file', '') from error
        except OcrFailure as failure:
            return failure


class Share:
    """A share that is down for the first `down` checks."""

    def __init__(self, down):
        self.down, self.checks = down, 0

    def answers(self, folder):
        self.checks += 1
        return self.checks > self.down


class ShareGuardTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(audit_readings, 'RETRY_WAITS', (0,))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_outage_waits_and_reads_again_without_using_retries(self):
        share = Share(down=0)
        guard = ShareGuard(poll=0.01, limit=5, answers=share.answers)
        reads = []

        def read_best(path, saved, model=None):
            reads.append(time.monotonic())
            if len(reads) <= 5:  # five outages in a row: more than PHOTO_RETRIES
                share.down, share.checks = 3, 0  # this outage: the first check and two probes fail
                raise share_error()
            return 'result'

        with mock.patch.object(audit_readings, 'read_best', read_best):
            self.assertEqual(read_with_retries(PHOTO, '12345', share=guard), 'result')
        self.assertEqual(len(reads), 6)

    def test_outage_longer_than_the_wait_is_an_error(self):
        guard = ShareGuard(poll=0.01, limit=0.05, answers=lambda folder: False)
        with mock.patch.object(audit_readings, 'read_best', mock.Mock(side_effect=share_error())):
            with self.assertRaises(OcrFailure):
                read_with_retries(PHOTO, '12345', share=guard)

    def test_missing_file_on_a_working_share_uses_the_retries(self):
        guard = ShareGuard(poll=0.01, limit=5, answers=lambda folder: True)
        read_best = mock.Mock(side_effect=share_error())
        with mock.patch.object(audit_readings, 'read_best', read_best):
            with self.assertRaises(OcrFailure):
                read_with_retries(PHOTO, '12345', share=guard)
        self.assertEqual(read_best.call_count, audit_readings.PHOTO_RETRIES + 1)

    def test_model_error_keeps_the_retries(self):
        guard = ShareGuard(poll=0.01, limit=5, answers=lambda folder: self.fail('the share is not checked'))
        read_best = mock.Mock(side_effect=OcrFailure('Connection error', ''))
        with mock.patch.object(audit_readings, 'read_best', read_best):
            with self.assertRaises(OcrFailure):
                read_with_retries(PHOTO, '12345', share=guard)
        self.assertEqual(read_best.call_count, audit_readings.PHOTO_RETRIES + 1)

    def test_every_worker_pauses_during_an_outage(self):
        share = Share(down=10)
        guard = ShareGuard(poll=0.02, limit=5, answers=share.answers)
        failing = threading.Thread(target=guard.wait_for, args=(PHOTO,))
        failing.start()
        while guard.open.is_set():
            time.sleep(0.001)
        resumed = []
        other = threading.Thread(target=lambda: (guard.pause(), resumed.append(share.checks)))
        other.start()
        other.join()
        failing.join()
        self.assertGreater(resumed[0], 10)  # the other worker went on only once the share answered
        self.assertTrue(guard.open.is_set())


if __name__ == '__main__':
    unittest.main()
