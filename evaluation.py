
from __future__ import annotations

import os

SCORE_THRESHOLD = float(os.environ.get('FOTOMATES_EVAL_SCORE', '0.8'))

GROUPS = [
    ('matched', 'Matched the saved reading'),
    ('mismatch_high', f'Mismatched, evaluation score {SCORE_THRESHOLD} or higher'),
    ('mismatch_low', f'Mismatched, evaluation score below {SCORE_THRESHOLD}'),
    ('no_meter', 'No meter in the picture'),
    ('no_digits', 'No digits found (meter, but nothing readable)'),
    ('no_saved', 'No saved reading in the file name to compare with'),
    ('failed', 'Failed (model error)'),
]
LABELS = dict(GROUPS)


def group(row) -> str:
    """The group of one photo: a dict with saved, extracted, score, status
    ('done', 'missing' / 'no_digits', 'error') and no_meter (bool)."""
    if row.get('status') == 'error':
        return 'failed'
    if row.get('no_meter'):
        return 'no_meter'
    if row.get('extracted') in (None, ''):
        return 'no_digits'
    if row.get('saved') in (None, ''):
        return 'no_saved'
    if row['extracted'] == row['saved']:
        return 'matched'
    score = row.get('score')
    return 'mismatch_high' if score is not None and score >= SCORE_THRESHOLD else 'mismatch_low'


def evaluate(rows) -> dict:
    """{'counts': {group: n}, 'files': {group: [row, ...]}, 'photos': n}; each
    row keeps name, saved, extracted, all_digits and score."""
    files = {key: [] for key, _ in GROUPS}
    for row in rows:
        files[group(row)].append({k: row.get(k) for k in ('name', 'saved', 'extracted', 'all_digits', 'score')})
    for listed in files.values():
        listed.sort(key=lambda row: row['name'] or '')
    return dict(photos=sum(map(len, files.values())), counts={k: len(v) for k, v in files.items()}, files=files)


def lines(evaluation: dict) -> list[tuple[str, str, str]]:
    """(group key, label, count) for every group, the last two only when not empty."""
    counts = evaluation['counts']
    return [(key, label, str(counts[key])) for key, label in GROUPS
            if counts[key] or key not in ('no_saved', 'failed')]
