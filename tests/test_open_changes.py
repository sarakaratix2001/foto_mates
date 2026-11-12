"""The open changes of 11 Nov 2026: test tooling (Task 0), SOURCE (1), OFF (2), far (3), near-black
frames (4), the cell-count line (5), cell history (6), short answers (6a), stream timings (6b),
the cached prefix (6c), the box photo size (7), few-shots on the third read (10), cost (11) and
capacity (12). Every reading change is behind a setting that is off by default.

    python -m unittest discover tests
"""
import contextlib
import csv
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
import compare_runs  # noqa: E402
import extract_readings  # noqa: E402
from audit_readings import (CellHistory, _cells_hint_read, capacity_line, capacity_rate, connect,  # noqa: E402
                            finish_time, read_one)
from extract_readings import (extract_off, extract_source, parse_box, with_reading_line,  # noqa: E402
                              _cells_digits, extract_device_type)

NAME = 'AL0060991_20261027125226_READER_60991_57-5701_ExpAktiv_41-33_19-82.jpg'


def result(text, cells='', sent='crop(2.7x)', width=240, height=400):
    return NS(text=text, digit_confidence=[0.99] * len(cells), digit_alternatives=[], image_report=dict(
        width=width, height=height), evidence_source=sent, device_type='digital', code_confidence=[], clear=None,
        decimal_digits=None, reading_seconds=1.0, digit_margin=None, digit_runner_up=None)


def kept(text, cells, score=0.99, box_answer='100,300,700,450', box_reason=None, **extra):
    one = dict(result=result(text, cells), cells=cells, whole=cells.lstrip('0') or None, decimals=0,
               match='no', parts=None, score=score, scale='crop(2.7x)', role='crop')
    return dict(one, reread=False, seconds=1.0, box_answer=box_answer, box_reason=box_reason, reads=[one], **extra)


def row_for(read, name=NAME):
    with mock.patch.object(audit_readings, 'read_with_retries', return_value=read):
        return read_one(Path(name), None)


class SourceTest(unittest.TestCase):  # Task 1
    def test_the_source_line_is_parsed(self):
        self.assertEqual(extract_source('READING: 60991\nQUALITY: 9\nSOURCE: HANDWRITTEN'), 'HANDWRITTEN')
        self.assertEqual(extract_source('SOURCE: lcd'), 'LCD')
        self.assertIsNone(extract_source('READING: 1'))
        self.assertIsNone(extract_source('SOURCE: SOMETHING'))

    def test_handwritten_digits_give_no_reading(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: 6 0 9 9 1\nREADING: 60991\nQUALITY: 10\nSOURCE: HANDWRITTEN',
                           '60991'))
        self.assertEqual((row['status'], row['reason_code'], row['match'], row['extracted_whole']),
                         ('no_meter', 'source:HANDWRITTEN', 'no', None))
        self.assertEqual((row['extracted_reading'], row['crop_digits'], row['source']),
                         ('60991', '60991', 'HANDWRITTEN'))  # the digits stay in the export

    def test_a_display_reads_as_before(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: 6 0 9 9 1\nREADING: 60991\nQUALITY: 10\nSOURCE: LCD', '60991'))
        self.assertEqual((row['status'], row['reason_code'], row['source']), ('done', None, 'LCD'))

    def test_no_digits_with_source_none_stays_missing(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: NONE\nREADING: UNREADABLE\nQUALITY: 0\nSOURCE: NONE', ''))
        self.assertEqual(row['status'], 'missing')

    def test_the_v3_prompts_ask_for_it(self):
        self.assertIn('exactly these six lines', extract_readings._V3_SYSTEM_PROMPT)
        self.assertTrue(extract_readings._V3_SYSTEM_PROMPT.rstrip().split('\n')[-1].startswith('SOURCE:'))
        self.assertIn('SOURCE (always, last)', extract_readings._V3_USER_PROMPT)
        self.assertNotIn('SOURCE', extract_readings._V2_SYSTEM_PROMPT + extract_readings._V2_USER_PROMPT)


class OffTest(unittest.TestCase):  # Task 2
    def test_off_is_its_own_status(self):
        self.assertTrue(extract_off('READING: OFF'))
        self.assertFalse(extract_off('READING: UNREADABLE'))
        row = row_for(kept('TYPE: DIGITAL\nCELLS: NONE\nREADING: OFF\nQUALITY: 0\nSOURCE: NONE', ''))
        self.assertEqual(row['status'], 'off')

    def test_the_database_and_reports_take_it(self):
        db = connect(':memory:')
        audit_readings.save(db, dict(picture_name=NAME, month='2026-10', match='no', status='off'))
        totals = [dict(zip([c[0] for c in cursor.description], r)) for cursor in [audit_readings.summary(db)]
                  for r in cursor]
        self.assertEqual(totals[0]['display_off'], 1)
        self.assertIn('1 display off', audit_readings.summary_lines(totals)[0])
        missing = db.execute(*audit_readings.list_query('missing')).fetchall()
        self.assertEqual(len(missing), 1)

    def test_v3_offers_off(self):
        self.assertIn('or BLANK, or OFF', extract_readings._V3_SYSTEM_PROMPT)
        self.assertIn('READING: OFF', extract_readings._V3_USER_PROMPT)


class FarTest(unittest.TestCase):  # Task 3
    def test_a_small_box_is_far(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: 1 2 3\nREADING: 123', '123', box_answer='400,400,560,450'))
        self.assertEqual((row['box_px_w'], row['box_px_h'], row['reason_code']), (38, 20, 'far:38'))
        self.assertEqual(row['extracted_reading'], '123')

    def test_a_normal_box_is_not(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: 1 2 3\nREADING: 123', '123', box_answer='100,300,700,450'))
        self.assertEqual((row['box_px_w'], row['reason_code']), (144, None))

    def test_no_box_used_no_size(self):
        row = row_for(kept('TYPE: DIGITAL\nCELLS: 1 2 3\nREADING: 123', '123', box_answer='NONE',
                           box_reason='no box'))
        self.assertEqual((row['box_px_w'], row['reason_code']), (None, None))


class NearBlackTest(unittest.TestCase):  # Task 4
    def blank(self, pixels, **settings):
        with mock.patch.object(extract_readings, 'decode_image', return_value=pixels), \
                mock.patch.multiple(extract_readings, **settings):
            return extract_readings.blank_photo(Path('x.jpg'))

    def test_off_by_default_a_dim_frame_is_read(self):
        dim = np.full((400, 240, 3), 12, np.uint8)
        self.assertIsNone(self.blank(dim, BLANK_MAX=5.0, BLANK_P99=0.0))

    def test_with_the_percentile_rule_a_dim_frame_is_blank(self):
        dim = np.full((400, 240, 3), 12, np.uint8)
        self.assertIn('near-black', self.blank(dim, BLANK_MAX=5.0, BLANK_P99=40.0, BLANK_DIM=20.0).text)

    def test_a_lit_display_in_a_dark_photo_is_kept(self):
        dark = np.full((400, 240, 3), 4, np.uint8)
        dark[200:230, 60:180] = 220  # a lit display: 3.75% of the photo
        self.assertIsNone(self.blank(dark, BLANK_MAX=0.0, BLANK_P99=40.0, BLANK_DIM=20.0))

    def test_the_old_rule_still_holds(self):
        black = np.zeros((400, 240, 3), np.uint8)
        self.assertIn('black frame', self.blank(black, BLANK_MAX=5.0, BLANK_P99=0.0).text)


class RepeatLineTest(unittest.TestCase):  # Task 5
    def test_the_line_counts_the_cells(self):
        self.assertTrue(extract_readings._REPEAT_LINE.startswith('- Count the cells before reading. A 1 fills'))


class CellHistoryTest(unittest.TestCase):  # Task 6
    def history(self):
        db = connect(':memory:')
        for name, date, digits, score in (('A_1', '2026-09-27', '0717182', 0.95), ('A_2', '2026-08-27', '07171', 0.99),
                                          ('A_3', '2026-10-01', '07', 0.5)):
            audit_readings.save(db, dict(picture_name=name, contract_id='AL1', register='ExpAktiv', date=date,
                                         month=date[:7], extracted_reading=digits, score=score, match='yes',
                                         status='done'))
        history = CellHistory()
        history.load(db)
        return db, history

    def test_the_latest_confident_earlier_read_counts(self):
        _, history = self.history()
        self.assertEqual(history.expected('AL1', 'ExpAktiv', '2026-10-27', 'X'), 7)
        self.assertEqual(history.expected('AL1', 'ExpAktiv', '2026-09-01', 'X'), 5)  # only earlier photos
        self.assertIsNone(history.expected('AL1', 'ExpPik', '2026-10-27', 'X'))
        self.assertIsNone(history.expected('AL1', 'ExpAktiv', '2026-08-01', 'X'))

    def test_a_saved_photo_is_recorded_or_forgotten(self):
        db, history = self.history()
        history.record(db, dict(picture_name='A_4', contract_id='AL1', register='ExpAktiv', date='2026-10-20',
                                extracted_reading='07171820', score=0.97, status='done'))
        self.assertEqual(history.expected('AL1', 'ExpAktiv', '2026-10-27', 'X'), 8)
        history.record(db, dict(picture_name='A_4', contract_id='AL1', register='ExpAktiv', date='2026-10-20',
                                extracted_reading='07171820', score=0.4, status='done'))
        self.assertEqual(history.expected('AL1', 'ExpAktiv', '2026-10-27', 'X'), 7)

    def test_a_short_read_is_read_again_told_the_count(self):
        short = kept('CELLS: 0 7 7 1 8 2\nREADING: 077182', '077182', score=0.8)
        hinted = dict(result=result('CELLS: 0 7 1 7 1 8 2\nREADING: 0717182', '0717182'), cells='0717182',
                      whole='717182', decimals=0, match='yes', parts=None, score=0.9, scale='crop(2.7x)')
        calls = []

        def scored(*args, **kwargs):
            calls.append(kwargs)
            return dict(hinted)
        with mock.patch.object(audit_readings, 'read_scored', side_effect=scored):
            best = _cells_hint_read(Path(NAME), '717182', None, short, 7)
        self.assertEqual(calls[0]['hint'], 'This display has 7 cells.')
        self.assertEqual(calls[0]['box'], [100.0, 300.0, 700.0, 450.0])
        self.assertEqual((best['cells'], best['cells_hint'], best['other']['cells']), ('0717182', 1, '077182'))
        self.assertEqual([r['role'] for r in best['reads']], ['crop', 'hint'])

    def test_a_hinted_read_of_another_length_is_not_kept(self):
        short = kept('CELLS: 0 7 7 1 8 2\nREADING: 077182', '077182', score=0.8)
        other = dict(result=result('READING: 77182', '77182'), cells='77182', whole='77182', decimals=0,
                     match='no', parts=None, score=0.9, scale='crop')
        with mock.patch.object(audit_readings, 'read_scored', return_value=other):
            best = _cells_hint_read(Path(NAME), '717182', None, short, 7)
        self.assertEqual((best['cells'], best['cells_hint']), ('077182', 0))

    def test_a_full_read_is_left_alone(self):
        full = kept('READING: 0717182', '0717182')
        with mock.patch.object(audit_readings, 'read_scored', side_effect=AssertionError):
            self.assertIs(_cells_hint_read(Path(NAME), '717182', None, full, 7), full)

    def test_the_hint_goes_after_the_image(self):
        photo = np.full((400, 240, 3), 128, np.uint8)
        seen = {}

        def model(**kwargs):
            seen.update(kwargs)
            return NS(model_extra=None, usage=None, choices=[NS(message=NS(content='READING: 1'), logprobs=None)])
        with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                mock.patch.object(extract_readings, 'complete', side_effect=lambda server=None, **k: model(**k)):
            extract_readings.ocr_image(Path('x.jpg'), hint='This display has 7 cells.')
        parts = seen['messages'][1]['content']
        self.assertEqual((parts[-2]['type'], parts[-1]['text']), ('image_url', 'This display has 7 cells.'))


class ShortAnswerTest(unittest.TestCase):  # Task 6a
    def test_the_box_as_four_numbers(self):
        self.assertEqual(parse_box('110,120,335,869'), [110.0, 120.0, 335.0, 869.0])
        self.assertEqual(parse_box('```json\n[{"bbox_2d": [110, 120, 335, 869], "label": "x"}]\n```'),
                         [110.0, 120.0, 335.0, 869.0])
        self.assertIsNone(parse_box('NONE'))

    def test_v4_has_no_reading_line(self):
        system = extract_readings._V4_SYSTEM_PROMPT
        self.assertNotIn('READING:', system)
        self.assertIn('TYPE: D (digital) or M (mechanical)', system)
        self.assertIn('exactly these five lines', system)

    def test_a_v4_answer_is_read_from_cells(self):
        with mock.patch.object(extract_readings, 'PROMPT', 'v4'):
            text = with_reading_line('TYPE: D\nCELLS: 0 0 1 4 6 2\nDECIMALS: 0\nQUALITY: 9\nSOURCE: LCD')
            self.assertTrue(text.endswith('READING: 001462'))
            self.assertEqual(extract_device_type(text), 'digital')
            self.assertEqual(extract_readings._reading_digits(text), '001462')
            self.assertEqual(_cells_digits('CELLS: 001462'), '001462')
            self.assertTrue(extract_readings.extract_no_meter(with_reading_line('TYPE: D\nCELLS: NO METER')))
            self.assertTrue(extract_off(with_reading_line('TYPE: D\nCELLS: OFF')))
        with mock.patch.object(extract_readings, 'PROMPT', 'v2'):
            self.assertIsNone(_cells_digits('CELLS: 001462'))  # v2 as before
            self.assertEqual(with_reading_line('CELLS: 1 2'), 'CELLS: 1 2')


class CostTest(unittest.TestCase):  # Tasks 6b, 11
    def test_each_call_is_counted(self):
        response = NS(usage=NS(prompt_tokens=1670, completion_tokens=45, prompt_tokens_details=None),
                      model_extra={'timings': {'prompt_n': 120}}, choices=[])
        extract_readings.start_photo_stats()
        with mock.patch.object(extract_readings, 'complete', return_value=response):
            extract_readings.model_call(model='m', messages=[])
            extract_readings.model_call(model='m', messages=[])
        stats = extract_readings.photo_stats()
        self.assertEqual((stats['model_calls'], stats['prompt_tokens'], stats['completion_tokens'],
                          stats['cached_tokens'], stats['first_token_ms']), (2, 3340, 90, 3100, None))

    def test_a_stream_is_put_back_together(self):
        def token(text, logprob):
            return NS(token=text, logprob=logprob, top_logprobs=[])
        chunks = [NS(usage=None, model_extra={}, choices=[NS(delta=NS(content='READING: '), finish_reason=None,
                                                             logprobs=NS(content=[token('READING: ', 0.0)]))]),
                  NS(usage=None, model_extra={}, choices=[NS(delta=NS(content='12'), finish_reason='stop',
                                                             logprobs=NS(content=[token('1', -0.1), token('2', -0.2)]))]),
                  NS(usage=NS(prompt_tokens=10, completion_tokens=3), model_extra={'timings': {'prompt_n': 10}},
                     choices=[])]
        seen = {}

        def complete(server=None, **request):
            seen.update(request)
            return iter(chunks)
        extract_readings.start_photo_stats()
        with mock.patch.object(extract_readings, 'complete', side_effect=complete), \
                mock.patch.object(extract_readings, 'STREAM', True):
            response = extract_readings.model_call(model='m', messages=[], logprobs=True)
        self.assertEqual((seen['stream'], seen['stream_options']), (True, {'include_usage': True}))
        choice = response.choices[0]
        self.assertEqual(choice.message.content, 'READING: 12')
        self.assertEqual([t.token for t in choice.logprobs.content], ['READING: ', '1', '2'])
        self.assertEqual(response.usage.completion_tokens, 3)
        stats = extract_readings.photo_stats()
        self.assertEqual(stats['model_calls'], 1)
        self.assertIsNotNone(stats['first_token_ms'])
        self.assertIsNotNone(stats['generation_ms'])

    def test_the_columns_are_saved(self):
        db = connect(':memory:')
        audit_readings.save(db, dict(picture_name=NAME, match='no', status='done', model_calls=3, prompt_tokens=4410,
                                     completion_tokens=140, first_token_ms=900.0))
        self.assertEqual(db.execute('SELECT model_calls, prompt_tokens FROM readings').fetchone(), (3, 4410))
        self.assertEqual(db.execute("SELECT count(*) FROM sqlite_master WHERE name = 'cell_counts'").fetchone(), (1,))

    def test_the_closing_lines(self):
        lines = audit_readings.cost_lines([10.0, 20.0, 30.0], [2, 3, 3], [3000, 4000, 4000], [90, 140, 140],
                                          [300.0, 400.0], [700.0, 600.0])
        self.assertIn('20.0 s per photo (median)', lines[0])
        self.assertIn('2.67 model calls per photo', lines[0])
        self.assertIn('35% of model time before the first token', lines[1])


class PrefixTest(unittest.TestCase):  # Task 6c
    def test_the_instructions_go_before_the_image(self):
        photo = np.full((400, 240, 3), 128, np.uint8)
        seen = {}

        def model(server=None, **kwargs):
            seen.update(kwargs)
            return NS(model_extra=None, usage=None, choices=[NS(message=NS(content='READING: 1'), logprobs=None)])
        with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                mock.patch.object(extract_readings, 'complete', side_effect=model), \
                mock.patch.multiple(extract_readings, PREFIX=True, CACHE_PROMPT=True, SLOT_IDS=[3]):
            extract_readings.ocr_image(Path('x.jpg'))
            read = seen['messages']
            extract_readings.find_box(Path('x.jpg'))
            box = seen['messages']
            crop_system = extract_readings.system_prompt(crop=True)
        self.assertIn(extract_readings._USER_PROMPT, read[0]['content'])
        self.assertEqual([part['type'] for part in read[1]['content']], ['image_url'])
        self.assertEqual(seen['extra_body']['cache_prompt'], True)
        self.assertEqual(seen['extra_body']['id_slot'], 3)
        self.assertEqual(box[0]['content'], crop_system)
        self.assertEqual([part['type'] for part in box[1]['content']], ['image_url', 'text'])

    def test_the_slot_list(self):
        self.assertEqual(extract_readings._slot_ids('0-3,7'), [0, 1, 2, 3, 7])


class BoxSideTest(unittest.TestCase):  # Task 7
    def test_zero_sends_the_photo_as_it_is(self):
        photo = np.full((400, 240, 3), 128, np.uint8)
        shapes = []

        def png(pixels):
            shapes.append(pixels.shape[:2])
            return 'data:image/png;base64,x'
        answer = NS(model_extra=None, usage=None, choices=[NS(message=NS(content='NONE'))])
        for side in (480.0, 0.0):
            with mock.patch.object(extract_readings, 'decode_image', return_value=photo), \
                    mock.patch.object(extract_readings, 'png_data_url', side_effect=png), \
                    mock.patch.object(extract_readings, 'complete', return_value=answer), \
                    mock.patch.object(extract_readings, 'BOX_SIDE', side):
                extract_readings.find_box(Path('x.jpg'))
        self.assertEqual(shapes, [(800, 480), (400, 240)])


class CompareRunsTest(unittest.TestCase):  # Task 0
    def test_the_rule(self):
        self.assertTrue(compare_runs.digits_match('0009261', '9261'))
        self.assertTrue(compare_runs.digits_match('00092619', '9261'))      # one decimal
        self.assertTrue(compare_runs.digits_match('0009261123', '9261'))    # three
        self.assertFalse(compare_runs.digits_match('00926112345', '9261'))  # five
        self.assertTrue(compare_runs.digits_match('000926112345', '92611'))  # four on 9+ cells
        self.assertFalse(compare_runs.digits_match('00926', '9261'))
        self.assertTrue(compare_runs.within_one('0009262', '9261'))
        self.assertFalse(compare_runs.right('', '9261'))

    def runs(self, folder, old, new):
        fields = ['picture_name', 'saved_reading', 'extracted_reading', 'seconds', 'box_answer', 'crop_sent',
                  'whole_sent', 'third_sent']
        for name, rows in (('old.csv', old), ('new.csv', new)):
            with open(Path(folder) / name, 'w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, fields)
                writer.writeheader()
                writer.writerows(rows)
        with open(Path(folder) / 'test_set.csv', 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, ['picture_name', 'source_path', 'group', 'colour', 'truth'])
            writer.writeheader()
            writer.writerows([dict(picture_name='a.jpg', group='mismatch', colour='red'),
                              dict(picture_name='b.jpg', group='mismatch', colour='green'),
                              dict(picture_name='c.jpg', group='low_quality', colour='red', truth='0717182'),
                              dict(picture_name='d.jpg', group='confident')])

    def run_compare(self, folder):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_runs.main([str(Path(folder) / n) for n in ('old.csv', 'new.csv', 'test_set.csv')])
        return out.getvalue()

    def test_a_run_against_itself_changes_nothing(self):
        rows = [dict(picture_name='a.jpg', saved_reading='9261', extracted_reading='="000926"', seconds='20',
                     box_answer='1,2,3,4', crop_sent='crop(2x)'),
                dict(picture_name='b.jpg', saved_reading='50', extracted_reading='51', seconds='18'),
                dict(picture_name='c.jpg', saved_reading='71718', extracted_reading='077182', seconds='30'),
                dict(picture_name='d.jpg', saved_reading='4', extracted_reading='00004', seconds='10')]
        with tempfile.TemporaryDirectory() as folder:
            self.runs(folder, rows, rows)
            table, _ = compare_runs.compare(*(compare_runs.load_run(Path(folder) / n) for n in ('old.csv', 'new.csv')),
                                            compare_runs.load_test_set(Path(folder) / 'test_set.csv'))
            text = self.run_compare(folder)
        self.assertEqual(sum(c['fixed'] for c in table.values()), 0)
        self.assertEqual(sum(c['broken'] for c in table.values()), 0)
        self.assertIn('OLD: 4 photos, 19.5 s per photo (mean)', text)

    def test_fixed_and_broken_are_counted(self):
        old = [dict(picture_name='a.jpg', saved_reading='9261', extracted_reading='000926', seconds='20'),
               dict(picture_name='c.jpg', saved_reading='71718', extracted_reading='077182', seconds='30'),
               dict(picture_name='d.jpg', saved_reading='4', extracted_reading='00004', seconds='10')]
        new = [dict(picture_name='a.jpg', saved_reading='9261', extracted_reading='0009261', seconds='12'),
               dict(picture_name='c.jpg', saved_reading='71718', extracted_reading='0717182', seconds='12'),
               dict(picture_name='d.jpg', saved_reading='4', extracted_reading='00009', seconds='12')]
        with tempfile.TemporaryDirectory() as folder:
            self.runs(folder, old, new)
            table, listed = compare_runs.compare(
                *(compare_runs.load_run(Path(folder) / n) for n in ('old.csv', 'new.csv')),
                compare_runs.load_test_set(Path(folder) / 'test_set.csv'))
        self.assertEqual(table[('mismatch', 'red')]['fixed'], 1)
        self.assertEqual(table[('low_quality', 'red')]['fixed'], 1)  # judged on the typed truth
        self.assertEqual(table[('confident', '')]['broken'], 1)
        self.assertEqual(table[('mismatch', 'green')]['missing'], 1)


@unittest.skipUnless(__import__('importlib').util.find_spec('openpyxl'), 'openpyxl is not installed')
class MakeTestSetTest(unittest.TestCase):  # Task 0
    def test_the_set_is_written(self):
        import make_test_set
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill
        names = [f'AL00{n:05d}_20261027120000_R_{n}_57-5701_ExpAktiv_41-33_19-82.jpg' for n in range(1, 8)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with open(folder / 'details.csv', 'w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, ['picture_name', 'match', 'score', 'device_type'])
                writer.writeheader()
                for n, name in enumerate(names):
                    writer.writerow(dict(picture_name=name, match='yes' if n < 4 else 'no', score='0.95',
                                         device_type='mechanical' if n == 6 else 'digital'))
            book = Workbook()
            sheet = book.active
            sheet.title = 'Mospërputhje'
            sheet.append(['Emri i fotos', 'Lexuar'])
            for name, rgb in ((names[0], 'FF00B050'), (names[1], 'FFFF0000'), (names[2], 'FFFFFF00'), (names[3], None)):
                sheet.append([Path(name).stem, 1])
                if rgb:
                    sheet.cell(sheet.max_row, 1).fill = PatternFill('solid', fgColor=rgb)
            low = book.create_sheet('Foto me cilësi të dobët')
            low.append(['Foto'])
            low.append([names[4]])
            low.cell(2, 1).fill = PatternFill('solid', fgColor='FFC00000')
            book.save(folder / 'raport.xlsx')
            labels = Workbook()
            labels.active.title = 'Etiketat'
            labels.active.append(['Emri', 'Leximi i saktë'])
            labels.active.append([names[4], '0717182'])
            labels.save(folder / 'labels.xlsx')
            out = folder / 'test_set.csv'
            with contextlib.redirect_stderr(io.StringIO()):
                make_test_set.main(['--details', str(folder / 'details.csv'), '--photo-dir', 'Z:/27',
                                    '--mismatch-xlsx', str(folder / 'raport.xlsx'),
                                    '--lowq-xlsx', str(folder / 'raport.xlsx'), '--labels-xlsx',
                                    str(folder / 'labels.xlsx'), '--out', str(out)])
            with open(out, encoding='utf-8-sig', newline='') as file:
                rows = list(csv.DictReader(file))
        got = [(r['picture_name'], r['group'], r['colour'], r['truth']) for r in rows if r['group'] in
               ('mismatch', 'low_quality')]
        self.assertEqual(got, [(names[0], 'mismatch', 'green', ''), (names[1], 'mismatch', 'red', ''),
                               (names[2], 'mismatch', 'yellow', ''), (names[4], 'low_quality', 'red', '0717182')])
        self.assertEqual(len([r for r in rows if r['group'] == 'confident']), 4)
        self.assertEqual([r['picture_name'] for r in rows if r['group'] == 'mechanical'], [names[6]])
        self.assertTrue(rows[0]['source_path'].endswith(names[0]))

    def test_colours(self):
        import make_test_set
        self.assertEqual([make_test_set.colour_of(c) for c in ('FF00B050', 'FFFF0000', 'FFFFFF00', 'FFFFFFFF',
                                                               'C6EFCE', 'FFC7CE', 'FFEB9C')],
                         ['green', 'red', 'yellow', None, 'green', 'red', 'yellow'])


class CapacityTest(unittest.TestCase):  # Task 12
    def test_one_photo_a_second(self):
        rate = capacity_rate([float(t) for t in range(100)])
        self.assertAlmostEqual(rate * 3600, 3600)
        self.assertAlmostEqual(rate * 86400, 86400)
        self.assertIsNone(capacity_rate([5.0]))

    def test_the_finish_time(self):
        now = datetime(2026, 11, 12, 12, 0)
        self.assertEqual(finish_time(3600, 1.0, now), datetime(2026, 11, 12, 13, 0))
        line = capacity_line(1.0, 0.5, 98300, now=now)
        self.assertEqual(line, 'CAPACITY: 3,600 photos/hour now (86,400/day) | average 1,800/hour | 98,300 left '
                               '| finishes about Fri 13 Nov 15:18')

    def test_a_run_prints_the_three_kinds_of_line(self):
        with tempfile.TemporaryDirectory() as folder:
            photos = Path(folder) / '2026' / '10' / '27'
            photos.mkdir(parents=True)
            for n in range(300):
                (photos / f'AL{n:07d}_20261027120000_R_5_57-5701_ExpAktiv_41-33_19-82.jpg').write_bytes(b'x')

            def fake(path, month, model=None):
                return dict(audit_readings.parse_name(path.name, month), picture_name=path.name, match='yes',
                            status='done', score=0.95, extracted_reading='0005', seconds=1.0, error=None,
                            model_calls=2, prompt_tokens=3000, completion_tokens=90, retake=0)
            args = Namespace(db=str(Path(folder) / 'run.sqlite3'), model='default', workers=4,
                             word_dir=str(Path(folder) / 'reports'), all_reports=False, month=None, no_word=True,
                             files_csv=None, only_failed=False, folders=[str(photos)], skip_errors=False, limit=None,
                             verbose=False, progress=100, report_every=0)
            out = io.StringIO()
            with mock.patch.object(audit_readings, 'read_one', side_effect=fake), \
                    mock.patch.object(audit_readings, 'server_slots', return_value=None), \
                    contextlib.redirect_stdout(out):
                audit_readings.command_read(args)
            sqlite3.connect(args.db).close()
        text = out.getvalue()
        self.assertIn('TO READ: 300 photos for 2026-10-27 (0 already saved, 300 left)', text)
        self.assertEqual(text.count('\nCAPACITY: '), 3)
        self.assertIn('200 left', text)
        self.assertRegex(text, r'done: 300 photos in \d+ min \([\d,]+ photos/hour, [\d,]+/day\), 2.0 model calls')
        self.assertIn('COST: 300 photos read', text)


if __name__ == '__main__':
    unittest.main()
