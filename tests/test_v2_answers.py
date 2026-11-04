"""How a model answer is read: the compact prompt's seven lines and v2's five.

    python -m unittest discover tests
"""
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_readings import (_clear_probability, _digit_confidence, _reading_digits,  # noqa: E402
                              extract_decimal_digits, extract_device_type)


def choice(text, sure=None):
    """A model reply whose tokens are single characters, each with probability 0.99,
    or sure[i] for the i-th digit of the CELLS line, with no alternatives."""
    tokens, digit, in_cells = [], 0, False
    for line in text.splitlines(keepends=True):
        in_cells = line.startswith('CELLS:')
        pieces = [line[:len(line.split(':')[0]) + 1]] + list(line[len(line.split(':')[0]) + 1:])
        for piece in pieces:
            p = 0.99
            if in_cells and piece.isdigit() and sure:
                p, digit = sure[digit], digit + 1
            tokens.append(NS(token=piece, logprob=math.log(p), top_logprobs=[]))
    return NS(message=NS(content=text), logprobs=NS(content=tokens))


def quality(first, alternatives, following=None):
    """A reply ending QUALITY: <first>[<following>], first with these {token: p} alternatives."""
    tokens = [NS(token='QUALITY:', logprob=0.0, top_logprobs=[]),
              NS(token=' ' + first, logprob=math.log(alternatives[first]),
                 top_logprobs=[NS(token=' ' + k, logprob=math.log(p)) for k, p in alternatives.items()])]
    if following:
        token, alts = following
        tokens.append(NS(token=token, logprob=math.log(alts[token]),
                         top_logprobs=[NS(token=k, logprob=math.log(p)) for k, p in alts.items()]))
    return NS(message=NS(content=''), logprobs=NS(content=tokens))


V2_DIGITAL = 'TYPE: DIGITAL\nCELLS: 0 2 0 5 1 6 4\nREADING: 0205164\nDECIMALS: UNKNOWN\nQUALITY: 8'
V2_MECHANICAL = 'TYPE: MECHANICAL\nCELLS: 0 1 6 0 5 4 7\nREADING: 0160547\nDECIMALS: 1\nQUALITY: 10'
COMPACT_DIGITAL = ('TYPE: DIGITAL (LCD or LED segments)\nCELLS: NONE\nWINDOWS: NONE\nREADING: 0205164\n'
                   'ROLLING: NONE\nDECIMALS: UNKNOWN\nQUALITY: POOR')


class AnswerTest(unittest.TestCase):
    def test_v2_digital(self):
        self.assertEqual(extract_device_type(V2_DIGITAL), 'digital')
        self.assertEqual(_reading_digits(V2_DIGITAL), '0205164')
        self.assertIsNone(extract_decimal_digits(V2_DIGITAL))

    def test_v2_digital_cells_win_over_reading(self):
        text = V2_DIGITAL.replace('READING: 0205164', 'READING: 020516')
        self.assertEqual(_reading_digits(text), '0205164')

    def test_v2_digital_unreadable_stays_unreadable(self):
        text = 'TYPE: DIGITAL\nCELLS: NONE\nREADING: UNREADABLE\nDECIMALS: UNKNOWN\nQUALITY: 0'
        self.assertEqual(_reading_digits(text), '')

    def test_v2_digital_glued_code_is_cut_from_cells(self):
        text = V2_DIGITAL.replace('0 2 0 5 1 6 4', '3 8 0 0 2 0 5 1 6 4').replace('0205164', '3800205164')
        self.assertEqual(_reading_digits(text), '0205164')

    def test_v2_mechanical(self):
        self.assertEqual(_reading_digits(V2_MECHANICAL), '0160547')
        self.assertEqual(extract_decimal_digits(V2_MECHANICAL), 1)

    def test_mechanical_reading_not_its_cells_is_dropped(self):
        self.assertEqual(_reading_digits(V2_MECHANICAL.replace('READING: 0160547', 'READING: 016054')), '')

    def test_decimals_line_alone(self):
        text = V2_MECHANICAL + '\nWINDOWS: dark dark dark dark dark dark red-small'
        self.assertEqual(extract_decimal_digits(text.replace('DECIMALS: 1', 'DECIMALS: 0')), 0)

    def test_compact_digital_still_read(self):
        self.assertEqual(_reading_digits(COMPACT_DIGITAL), '0205164')

    def test_cells_give_the_digit_probabilities_for_digital(self):
        sure = [0.9, 0.8, 0.99, 0.7, 0.99, 0.99, 0.6]
        weakest, per_digit, *_ = _digit_confidence(choice(V2_DIGITAL, sure))
        self.assertEqual([round(p, 2) for p in per_digit], sure)
        self.assertAlmostEqual(weakest, 0.6)

    def test_compact_digital_probabilities_from_reading(self):
        weakest, per_digit, *_ = _digit_confidence(choice(COMPACT_DIGITAL))
        self.assertEqual(len(per_digit), 7)

    def test_quality_clear_or_poor(self):
        self.assertAlmostEqual(_clear_probability(quality('CLEAR', {'CLEAR': 0.8, 'POOR': 0.2})), 0.8)

    def test_quality_number(self):
        self.assertAlmostEqual(_clear_probability(quality('8', {'8': 0.5, '6': 0.5})), 0.7)

    def test_quality_ten_split_in_two_tokens(self):
        clear = _clear_probability(quality('1', {'1': 0.9, '9': 0.1}, ('0', {'0': 0.75, '\n': 0.25})))
        self.assertAlmostEqual(clear, (0.9 * (1 + 9 * 0.75) + 0.1 * 9) / 10)

    def test_quality_ten_as_one_token(self):
        self.assertAlmostEqual(_clear_probability(quality('10', {'10': 1.0})), 1.0)


if __name__ == '__main__':
    unittest.main()
