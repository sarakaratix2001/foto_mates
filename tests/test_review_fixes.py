"""Fixes after the review of the open changes (12 Nov 2026): a stream whose text arrives as
delta.reasoning, a box answer cut off at max_tokens, the truth of a green Mospërputhje row,
status changes in compare_runs, and OFF / SOURCE on the web page.

    python -m unittest discover tests
"""
import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_runs  # noqa: E402
import extract_readings  # noqa: E402


class StreamTest(unittest.TestCase):
    def test_text_sent_as_reasoning_is_the_answer(self):
        def chunk(text, logprob):
            delta = NS(content=None, model_extra={'reasoning': text})
            return NS(usage=None, model_extra={}, choices=[NS(delta=delta, finish_reason=None, logprobs=NS(
                content=[NS(token=text, logprob=logprob, top_logprobs=[])]))])
        chunks = [NS(usage=None, model_extra={}, choices=[NS(delta=NS(content='', model_extra={}),
                                                             finish_reason=None, logprobs=None)]),
                  chunk('READING: ', 0.0), chunk('1', -0.1), chunk('2', -0.2),
                  NS(usage=NS(prompt_tokens=10, completion_tokens=3), model_extra={}, choices=[])]
        with mock.patch.object(extract_readings, 'complete', return_value=iter(chunks)), \
                mock.patch.object(extract_readings, 'STREAM', True):
            response = extract_readings.model_call(model='m', messages=[])
        self.assertEqual(response.choices[0].message.content, 'READING: 12')
        self.assertEqual([t.token for t in response.choices[0].logprobs.content], ['READING: ', '1', '2'])

    def test_content_is_used_when_the_server_sends_it(self):
        chunks = [NS(usage=None, model_extra={}, choices=[NS(
            delta=NS(content='12', model_extra={'reasoning': 'thinking'}), finish_reason='stop', logprobs=None)])]
        with mock.patch.object(extract_readings, 'complete', return_value=iter(chunks)), \
                mock.patch.object(extract_readings, 'STREAM', True):
            response = extract_readings.model_call(model='m', messages=[])
        self.assertEqual(response.choices[0].message.content, '12')


class BoxAnswerTest(unittest.TestCase):
    def find(self, answer, finish, **settings):
        choice = NS(message=NS(content=answer), finish_reason=finish)
        values = dict(BOX_ANSWER='numbers')
        values.update(settings)
        with mock.patch.multiple(extract_readings, decode_image=lambda path: np.zeros((400, 240, 3), np.uint8),
                                 png_data_url=lambda image: 'data:image/png;base64,x',
                                 model_call=lambda **request: NS(choices=[choice]), **values):
            return extract_readings.find_box(Path('photo.jpg'))

    def test_a_whole_answer_gives_the_box(self):
        found = self.find('[321, 393, 714, 538]', 'stop')
        self.assertEqual((found['box'], found['why']), ((321.0, 393.0, 714.0, 538.0), None))

    def test_a_cut_answer_is_no_box(self):
        found = self.find('[321, 393, 714, 53', 'length')
        self.assertEqual((found['box'], found['why']), (None, 'box answer cut off'))
        self.assertEqual(found['answer'], '[321, 393, 714, 53')

    def test_the_json_answer_is_as_before(self):
        found = self.find('{"bbox_2d": [321, 393, 714, 538]}', 'length', BOX_ANSWER='json')
        self.assertEqual(found['box'], (321.0, 393.0, 714.0, 538.0))

    def test_room_for_brackets_and_spaces(self):
        self.assertGreaterEqual(extract_readings.BOX_MAX_TOKENS, 32)


class GreenTruthTest(unittest.TestCase):
    def test_a_green_row_takes_the_ai_reading(self):
        import make_test_set
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill
        names = [f'AL00{n:05d}_20261027120000_R_{n}_57-5701_ExpAktiv_41-33_19-82.jpg' for n in range(1, 4)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with open(folder / 'details.csv', 'w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, ['picture_name', 'match', 'score', 'device_type', 'extracted_reading',
                                               'extracted_whole'])
                writer.writeheader()
                writer.writerow(dict(picture_name=names[0], match='no', score='0.95', device_type='digital',
                                     extracted_reading='="00092619"', extracted_whole='9261'))
                writer.writerow(dict(picture_name=names[1], match='no', score='0.95', device_type='digital',
                                     extracted_reading='="0717182"', extracted_whole=''))
                writer.writerow(dict(picture_name=names[2], match='no', score='0.95', device_type='digital',
                                     extracted_reading='="0055"', extracted_whole='55'))
            book = Workbook()
            sheet = book.active
            sheet.title = 'Mospërputhje'
            sheet.append(['Emri i fotos'])
            for name, rgb in zip(names, ('FF00B050', 'FF00B050', 'FFFF0000')):
                sheet.append([Path(name).stem])
                sheet.cell(sheet.max_row, 1).fill = PatternFill('solid', fgColor=rgb)
            book.save(folder / 'raport.xlsx')
            with contextlib.redirect_stderr(io.StringIO()):
                make_test_set.main(['--details', str(folder / 'details.csv'), '--photo-dir', 'Z:/27',
                                    '--mismatch-xlsx', str(folder / 'raport.xlsx'), '--confident', '0',
                                    '--mechanical', '0', '--out', str(folder / 'test_set.csv')])
            with open(folder / 'test_set.csv', encoding='utf-8-sig', newline='') as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([(r['colour'], r['truth']) for r in rows],
                         [('green', '9261'), ('green', '0717182'), ('red', '')])

    def test_a_green_row_read_differently_is_broken(self):
        item = dict(picture_name='a.jpg', group='mismatch', colour='green', truth='9261')
        old = {'a.jpg': dict(saved_reading='9281', extracted_reading='="00092619"')}
        same = {'a.jpg': dict(saved_reading='9281', extracted_reading='="0009261"')}
        other = {'a.jpg': dict(saved_reading='9281', extracted_reading='="00092819"')}  # now the reader's value
        table, _ = compare_runs.compare(old, same, [item])
        self.assertEqual(table[('mismatch', 'green')]['unchanged'], 1)
        table, _ = compare_runs.compare(old, other, [item])
        self.assertEqual(table[('mismatch', 'green')]['broken'], 1)


class StatusChangesTest(unittest.TestCase):
    def test_status_and_reason_codes_are_listed(self):
        test_set = [dict(picture_name='a.jpg', group='low_quality', colour='green', truth='', review_row='89'),
                    dict(picture_name='b.jpg', group='low_quality', colour='green', truth='', review_row='65'),
                    dict(picture_name='c.jpg', group='mismatch', colour='red', truth='', review_row='7')]
        old = {name: dict(picture_name=name, saved_reading='12', extracted_reading='', status='missing', seconds='1')
               for name in ('a.jpg', 'b.jpg', 'c.jpg')}
        new = {'a.jpg': dict(old['a.jpg'], status='off'),
               'b.jpg': dict(old['b.jpg'], reason_code='far:45'),
               'c.jpg': dict(old['c.jpg'], status='no_meter', reason_code='source:HANDWRITTEN')}
        out = io.StringIO()
        result = compare_runs.report(old, new, test_set, out=out)
        self.assertEqual(result['status_changed'][('low_quality', 'green', 'missing', 'off')], ['89'])
        self.assertEqual(result['reasons'][('low_quality', 'green', 'far')], ['65'])
        self.assertEqual(result['reasons'][('mismatch', 'red', 'source:HANDWRITTEN')], ['7'])
        self.assertIn('missing -> off: 1; 89', out.getvalue())

    def test_a_run_against_itself_lists_nothing(self):
        test_set = [dict(picture_name='a.jpg', group='confident', colour='', truth='')]
        run = {'a.jpg': dict(picture_name='a.jpg', saved_reading='12', extracted_reading='012', status='done',
                             seconds='1')}
        out = io.StringIO()
        result = compare_runs.report(run, run, test_set, out=out)
        self.assertEqual((result['total']['fixed'], result['total']['broken']), (0, 0))
        self.assertNotIn('status changed', out.getvalue())


class WebVerdictTest(unittest.TestCase):
    def test_off_and_source(self):
        import web_app
        off = web_app.verdict('', text='TYPE: DIGITAL\nCELLS: NONE\nREADING: OFF')
        self.assertEqual((off['status'], off['retake']), ('off', True))
        written = web_app.verdict('60991', 0.99, text='CELLS: 6 0 9 9 1\nREADING: 60991\nQUALITY: 9\nSOURCE: HANDWRITTEN')
        self.assertEqual(written['status'], 'no_meter')
        self.assertIn('handwritten', written['retake_reason'])
        lcd = web_app.verdict('60991', 0.99, text='CELLS: 6 0 9 9 1\nREADING: 60991\nQUALITY: 9\nSOURCE: LCD')
        self.assertEqual(lcd['status'], 'done')
        none = web_app.verdict('', text='TYPE: DIGITAL\nREADING: NO METER')
        self.assertEqual(none['status'], 'no_meter')


if __name__ == '__main__':
    unittest.main()
