"""Compare two runs of the test set, photo by photo.

    python compare_runs.py OLD.csv NEW.csv test_set.csv

OLD.csv and NEW.csv are details_<month>.csv files written by audit_readings.py (one row per
photo, every column); test_set.csv is written by make_test_set.py. A photo is right when its
digits fit the truth (the reading typed in the review when there is one, else the saved
reading) by the report's rule: up to three trailing decimals, four on reads of nine or more
cells, or within 1. Per group and colour it prints how many photos were fixed (wrong in OLD,
right in NEW), broken (right, then wrong), changed but still wrong, and unchanged; then the
seconds and model calls per photo of both runs.

A change ships only when fixed clearly exceeds broken on the red rows and the confident
matches lose nothing.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_DECIMALS = 3       # trailing decimals a reading may carry beyond the saved value
MAX_DECIMALS_LONG = 4  # on reads of LONG_CELLS cells or more
LONG_CELLS = 9


def _digits(value) -> str:
    return re.sub(r'\D', '', str(value or ''))


def _wholes(digits: str):
    """The whole parts the digits can stand for: 0 to 3 (4 on 9+ cells) trailing decimals cut."""
    most = MAX_DECIMALS_LONG if len(digits) >= LONG_CELLS else MAX_DECIMALS
    for cut in range(0, min(most, len(digits) - 1) + 1):
        yield int(digits[:len(digits) - cut] or 0)


def digits_match(digits, saved) -> bool:
    """The digits read give the saved reading once up to three trailing decimals (four on a read
    of nine or more cells) are cut; leading zeros never count."""
    digits, saved = _digits(digits), _digits(saved)
    if not digits or not saved:
        return False
    return int(saved) in set(_wholes(digits))


def within_one(digits, saved) -> bool:
    """As digits_match, but the whole part may differ from the saved reading by 1."""
    digits, saved = _digits(digits), _digits(saved)
    if not digits or not saved:
        return False
    return any(abs(whole - int(saved)) <= 1 for whole in _wholes(digits))


def right(digits, truth) -> bool:
    return digits_match(digits, truth) or within_one(digits, truth)


def load_run(path) -> dict:
    with open(path, encoding='utf-8-sig', newline='') as file:
        return {row['picture_name']: row for row in csv.DictReader(file)}


def load_test_set(path) -> list:
    with open(path, encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def _excel_text(value):
    """="0205164" (how the reports keep leading zeros for Excel) -> 0205164."""
    match = re.fullmatch(r'="(.*)"', value or '')
    return match[1] if match else (value or '')


def model_calls(row) -> int:
    """The model requests made for a photo: the model_calls column when the run wrote it, else the
    box request (when a box answer is saved) plus each read made (crop, whole, third, hint)."""
    if (row.get('model_calls') or '').strip():
        return int(float(row['model_calls']))
    calls = int(bool((row.get('box_answer') or '').strip()))
    calls += sum(bool((row.get(f'{read}_sent') or '').strip()) for read in ('crop', 'whole', 'third', 'hint'))
    if not calls and (row.get('photo_sent') or '').strip() not in ('', 'blank-check'):
        calls = 1  # a run from before the per-read columns: one read at least
    return calls


def compare(old: dict, new: dict, test_set: list):
    """{(group, colour): Counter(fixed, broken, changed_wrong, unchanged, right_old, right_new, photos,
    missing)} and the photos of each kind, for the listing."""
    table = defaultdict(Counter)
    listed = defaultdict(list)
    for item in test_set:
        name = item['picture_name']
        key = (item.get('group') or '', item.get('colour') or '')
        counts = table[key]
        counts['photos'] += 1
        before, after = old.get(name), new.get(name)
        if before is None or after is None:
            counts['missing'] += 1
            continue
        truth = _digits(item.get('truth')) or _digits(after.get('saved_reading') or before.get('saved_reading'))
        was, now = _excel_text(before.get('extracted_reading')), _excel_text(after.get('extracted_reading'))
        good_before, good_after = right(was, truth), right(now, truth)
        counts['right_old'] += good_before
        counts['right_new'] += good_after
        if not good_before and good_after:
            kind = 'fixed'
        elif good_before and not good_after:
            kind = 'broken'
        elif not good_before and was != now:
            kind = 'changed_wrong'
        else:
            kind = 'unchanged'
        counts[kind] += 1
        if kind in ('fixed', 'broken'):
            listed[kind].append((key, name, truth, was, now))
    return table, listed


def run_cost(run: dict, names) -> dict:
    seconds = [float(run[n]['seconds']) for n in names if n in run and (run[n].get('seconds') or '').strip()]
    calls = [model_calls(run[n]) for n in names if n in run]
    return dict(photos=len(calls),
                seconds_mean=statistics.fmean(seconds) if seconds else None,
                seconds_median=statistics.median(seconds) if seconds else None,
                calls_mean=statistics.fmean(calls) if calls else None)


def report(old: dict, new: dict, test_set: list, out=None, show=20) -> dict:
    out = out or sys.stdout
    table, listed = compare(old, new, test_set)
    columns = ('photos', 'fixed', 'broken', 'changed_wrong', 'unchanged', 'right_old', 'right_new', 'missing')
    print(f"{'group':<14}{'colour':<9}" + ''.join(f'{c:>14}' for c in columns), file=out)
    total = Counter()
    for (group, colour), counts in sorted(table.items()):
        total.update(counts)
        print(f'{group:<14}{colour:<9}' + ''.join(f'{counts[c]:>14}' for c in columns), file=out)
    print(f"{'TOTAL':<23}" + ''.join(f'{total[c]:>14}' for c in columns), file=out)
    names = {item['picture_name'] for item in test_set}
    for label, run in (('OLD', old), ('NEW', new)):
        cost = run_cost(run, names)
        if cost['photos']:
            print(f"{label}: {cost['photos']} photos, "
                  + (f"{cost['seconds_mean']:.1f} s per photo (mean), {cost['seconds_median']:.1f} s (median), "
                     if cost['seconds_mean'] is not None else 'no seconds, ')
                  + f"{cost['calls_mean']:.2f} model calls per photo", file=out)
    for kind in ('fixed', 'broken'):
        if listed[kind]:
            print(f'\n{kind} ({len(listed[kind])}' + (f', first {show}' if len(listed[kind]) > show else '') + '):',
                  file=out)
            for (group, colour), name, truth, was, now in listed[kind][:show]:
                print(f'  {group}/{colour or "-"} {name}: truth {truth}, old {was or "-"}, new {now or "-"}', file=out)
    return dict(table=table, total=total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('old', help='details CSV of the run before the change')
    parser.add_argument('new', help='details CSV of the run after the change')
    parser.add_argument('test_set', help='test_set.csv from make_test_set.py')
    parser.add_argument('--show', type=int, default=20, help='fixed and broken photos listed by name (default 20)')
    args = parser.parse_args(argv)
    report(load_run(args.old), load_run(args.new), load_test_set(args.test_set), show=args.show)


if __name__ == '__main__':
    main()
