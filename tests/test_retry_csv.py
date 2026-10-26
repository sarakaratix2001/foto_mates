import argparse
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import audit_readings as audit
import extract_readings as extraction
import prepare_retry_csv as inventory


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as out:
        writer=csv.DictWriter(out, fields);writer.writeheader();writer.writerows(rows)


class RetryTests(unittest.TestCase):
    def test_inventory_retries_only_absent_and_technical_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);day=root/'22';day.mkdir()
            for name in ['done.jpg','failed.jpg','new.jpg','excluded.jpg','notes.txt']:(day/name).touch()
            prior=root/'prior.csv'
            write_csv(prior,['picture_name','status'],[dict(picture_name='done.jpg',status='missing'),dict(picture_name='failed.jpg',status='error')])
            out=root/'retry.csv'
            with patch.object(inventory,'__file__',str(root/'prepare_retry_csv.py')):
                (root/'processing_exclusions.json').write_text(json.dumps({'files':[{'name':'excluded.jpg'}]}))
                with redirect_stdout(io.StringIO()):inventory.build_inventory(root,[prior],['22'],out,1)
            with out.open(encoding='utf-8-sig',newline='') as src:rows=list(csv.DictReader(src))
            self.assertEqual({r['picture_name']:r['reason'] for r in rows},{'failed.jpg':'technical_error','new.jpg':'not_processed'})
            self.assertTrue(all(Path(r['source_path']).is_absolute() for r in rows))

    def test_explicit_csv_deduplicates_and_never_scans_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'retry.csv'
            rows=[dict(picture_name=name,source_path=str(root/name)) for name in ['a.jpg','a.jpg','excluded.jpg','b.png']]
            write_csv(manifest,['picture_name','source_path'],rows)
            with patch.object(audit,'processing_exclusions',return_value={'excluded.jpg'}),patch.object(audit.os,'scandir',side_effect=AssertionError('Unexpected scan')):
                self.assertEqual([p.name for p in audit.csv_photos(manifest)],['a.jpg','b.png'])
            write_csv(manifest,['picture_name','source_path'],[dict(picture_name='wrong.jpg',source_path=str(root/'a.jpg'))])
            with self.assertRaises(ValueError):list(audit.csv_photos(manifest))

    def test_ram_run_persists_failures_before_final_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'retry.csv';name='AL0000001_20260922000000_TEST_1__ExpAktiv_0_0.jpg'
            write_csv(manifest,['picture_name','source_path'],[dict(picture_name=name,source_path=str(root/name))])
            args=argparse.Namespace(db=':memory:',model='default',workers=1,word_dir=str(root/'report'),files_csv=str(manifest),folders=[],only_failed=False,skip_errors=False,no_word=True,month='2026-09',verbose=False,progress=100,report_every=0,limit=None)
            row=dict(picture_name=name,source_path=str(root/name),month='2026-09',status='error',match='no',error='APIConnectionError <- ConnectError: reset',seconds=1)
            def check_export(*args,**kwargs):
                evidence=root/'report/progress/retry_failures.csv'
                self.assertTrue(evidence.exists(),'Failure must be saved before final export')
                with evidence.open(encoding='utf-8-sig',newline='') as src:records=list(csv.DictReader(src))
                self.assertEqual(records[0]['picture_name'],name)
                self.assertEqual(records[0]['source_path'],str(root/name))
            with patch.object(audit,'server_slots',return_value=None),patch.object(audit,'read_one',return_value=row),patch.object(audit,'export_reports',side_effect=check_export),redirect_stdout(io.StringIO()):audit.command_read(args)
            self.assertFalse(list(root.rglob('*.sqlite3')))

    def test_underlying_connection_cause_is_logged_and_secrets_redacted(self):
        try:
            try:raise OSError(104,'Connection reset by peer')
            except OSError as cause:raise RuntimeError('Connection error. token-secret') from cause
        except RuntimeError as error:
            with patch.object(extraction,'OCR_API_KEY','token-secret'):
                message=extraction._failure_reason(error)
        self.assertIn('Connection reset by peer',message)
        self.assertIn('ConnectionResetError',message)
        self.assertNotIn('token-secret',message)


if __name__=='__main__':unittest.main()
