"""Write test_set.csv: the photos every reading change is judged on.

    python make_test_set.py --details details_2026-10-27.csv --photo-dir "Z:\\mobiread2\\Foto\\2026\\10\\27" \\
        --mismatch-xlsx Raport_27_tetor_2026_te_dhenat_brendshem.xlsx \\
        --lowq-xlsx Raport_27_tetor_2026_te_dhenat_brendshem.xlsx \\
        --labels-xlsx photoreview_finetune.xlsx

Rows (one per photo and group; a photo can be in more than one group):
  mismatch     every coloured row of the Mospërputhje sheet: green (AI right, reader wrong),
               red (AI wrong on a visible display), yellow (unclear photo). A green row's
               truth is what the AI read in the details CSV (the saved reading is the wrong
               one there), so a later run that reads it differently counts as broken
  low_quality  every coloured row of the low-quality review ("Foto me cilësi të dobët"): green
               (unreadable by a person too), red (readable, AI failed), yellow (undecided); with
               the true reading from photoreview_finetune.xlsx (sheet Etiketat) when typed
  confident    100 random confident matches of the details CSV (match yes, score above 0.9)
  mechanical   100 random mechanical meters of the details CSV

Columns: picture_name, source_path, group, colour, truth, review_row. A run of the test set is
    python audit_readings.py --db test.sqlite3 read --photos test_set.csv

A sheet is found by name (--mismatch-sheet, --lowq-sheet, --labels-sheet), a photo in a row by
any cell that holds a photo name (AL0060991_20261027125226...), its colour by the row's fill.
A review kept as CSV instead (picture_name and colour columns, e.g. _review_list.csv) is read
with --lowq-csv.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import random
import re
import sys
from pathlib import Path

NAME = re.compile(r'[A-Z]{2}\d{5,}_\d{14}[^\s/\\]*', re.IGNORECASE)
COLOURS = ('green', 'red', 'yellow')
FIELDS = ('picture_name', 'source_path', 'group', 'colour', 'truth', 'review_row')


def _stem(name: str) -> str:
    return Path(str(name).strip()).stem if str(name).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) \
        else str(name).strip()


def colour_of(rgb: str | None) -> str | None:
    """green, red or yellow for a fill colour given as RRGGBB or AARRGGBB; None for white, grey,
    black or any other colour."""
    if not rgb or not re.fullmatch(r'[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8}', rgb):
        return None
    red, green, blue = (int(rgb[-6:][i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, light, saturation = colorsys.rgb_to_hls(red, green, blue)
    if saturation < 0.25 or light > 0.97 or light < 0.08:
        return None
    hue *= 360
    if hue < 25 or hue >= 330:
        return 'red'
    if 40 <= hue < 70:
        return 'yellow'
    if 75 <= hue < 170:
        return 'green'
    return None


def _fill_rgb(cell) -> str | None:
    fill = getattr(cell, 'fill', None)
    if fill is None or fill.fill_type not in ('solid', 'gray125', 'darkGray', 'lightGray', 'mediumGray'):
        return None
    color = fill.fgColor if fill.fgColor is not None and fill.fgColor.type == 'rgb' else fill.start_color
    if color is not None and color.type == 'rgb' and isinstance(color.rgb, str):
        return color.rgb
    if color is not None and color.type == 'indexed' and color.indexed is not None:
        from openpyxl.styles.colors import COLOR_INDEX
        return COLOR_INDEX[color.indexed] if color.indexed < len(COLOR_INDEX) else None
    return None


def _sheet(workbook, wanted: str | None, hints: tuple):
    if wanted:
        return workbook[wanted]
    for sheet in workbook.worksheets:
        lowered = sheet.title.lower()
        if any(hint in lowered for hint in hints):
            return sheet
    raise SystemExit(f'no sheet named like {hints} in {[s.title for s in workbook.worksheets]}; use the --*-sheet option')


def coloured_rows(path, sheet_name, hints, unresolved: list):
    """(picture name, colour, row number) for every row with a photo name and a green, red or
    yellow fill. A row with a fill whose colour is a theme colour is listed in unresolved."""
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=False, data_only=True)
    sheet = _sheet(workbook, sheet_name, hints)
    for row in sheet.iter_rows():
        name = next((NAME.search(str(c.value))[0] for c in row if c.value and NAME.search(str(c.value))), None)
        if not name:
            continue
        colours = [colour_of(_fill_rgb(c)) for c in row]
        colour = next((c for c in colours if c), None)
        if colour:
            yield name, colour, row[0].row
        elif any(getattr(c.fill, 'fill_type', None) == 'solid' for c in row):
            unresolved.append((sheet.title, row[0].row, name))


def csv_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as file:
        for number, row in enumerate(csv.DictReader(file), 2):
            name, colour = row.get('picture_name') or '', (row.get('colour') or row.get('color') or '').lower()
            if name and colour in COLOURS:
                yield name, colour, row.get('review_row') or number


def truths(path, sheet_name, column):
    """picture name -> the true reading typed in the labels workbook."""
    from openpyxl import load_workbook
    workbook = load_workbook(path, data_only=True)  # small; read_only would hold the file open on Windows
    sheet = workbook[sheet_name] if sheet_name in workbook.sheetnames else workbook.worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    header = [str(v or '').strip() for v in next(rows)]
    if column:
        index = header.index(column)
    else:
        candidates = [i for i, h in enumerate(header) if re.search(r'true|vert|sakt|correct|lexim|etiket|label', h, re.I)
                      and not re.search(r'emri|name|foto|file', h, re.I)]
        if not candidates:
            workbook.close()
            raise SystemExit(f'which column of {path} holds the true reading? {header}; use --truth-column')
        index = candidates[0]
    found = {}
    for values in rows:
        name = next((NAME.search(str(v))[0] for v in values if v and NAME.search(str(v))), None)
        value = re.sub(r'\D', '', str(values[index] if index < len(values) and values[index] is not None else ''))
        if name and value:
            found[_stem(name)] = value
    workbook.close()
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--details', required=True, help='details_<month>.csv of the 27 October run')
    parser.add_argument('--photo-dir', required=True, help='the folder of the 27 October photos (source_path)')
    parser.add_argument('--mismatch-xlsx', help='the workbook with the coloured Mospërputhje sheet')
    parser.add_argument('--mismatch-sheet', help='its sheet (default: the first named like Mosp...)')
    parser.add_argument('--lowq-xlsx', help='the workbook with the coloured low-quality review')
    parser.add_argument('--lowq-sheet', help='its sheet (default: the first named like "cilësi" or "dobët")')
    parser.add_argument('--lowq-csv', help='the low-quality review as CSV (picture_name, colour) instead')
    parser.add_argument('--labels-xlsx', help='photoreview_finetune.xlsx: the true reading of reviewed photos')
    parser.add_argument('--labels-sheet', default='Etiketat')
    parser.add_argument('--truth-column', help='the header of the true-reading column (default: found by name)')
    parser.add_argument('--confident', type=int, default=100)
    parser.add_argument('--mechanical', type=int, default=100)
    parser.add_argument('--seed', type=int, default=27, help='the random draw, so the set can be made again')
    parser.add_argument('--fewshot', default=str(Path(__file__).resolve().parent / 'fewshot' / 'examples.csv'),
                        help='the few-shot list; the photos its source column names are left out of the set, '
                             'as the model is shown them with their answer')
    parser.add_argument('--out', default='test_set.csv')
    args = parser.parse_args(argv)

    with open(args.details, encoding='utf-8-sig', newline='') as file:
        details = list(csv.DictReader(file))
    by_stem = {_stem(row['picture_name']): row['picture_name'] for row in details}
    # what the AI read in this run: its whole part when written, else all its digits
    ai_read = {_stem(row['picture_name']): re.sub(r'\D', '', row.get('extracted_whole') or '')
               or re.sub(r'\D', '', row.get('extracted_reading') or '') for row in details}
    folder = Path(args.photo_dir)
    rows, missing, unresolved = [], [], []
    examples = set()
    if args.fewshot and Path(args.fewshot).is_file():
        with open(args.fewshot, encoding='utf-8-sig', newline='') as file:
            examples = {_stem(row['source']) for row in csv.DictReader(file) if (row.get('source') or '').strip()}
    left_out = []

    def add(name, group, colour='', truth='', review_row=''):
        picture = by_stem.get(_stem(name))
        if picture and _stem(picture) in examples:
            left_out.append(picture)
            return
        if not picture:
            missing.append((group, name))
            return
        rows.append(dict(picture_name=picture, source_path=str(folder / picture), group=group, colour=colour,
                         truth=truth, review_row=review_row))

    if args.mismatch_xlsx:
        for name, colour, number in coloured_rows(args.mismatch_xlsx, args.mismatch_sheet, ('mosp',), unresolved):
            add(name, 'mismatch', colour, ai_read.get(_stem(name), '') if colour == 'green' else '',
                review_row=number)
    labels = truths(args.labels_xlsx, args.labels_sheet, args.truth_column) if args.labels_xlsx else {}
    review = (csv_rows(args.lowq_csv) if args.lowq_csv else
              coloured_rows(args.lowq_xlsx, args.lowq_sheet, ('cil', 'dob'), unresolved) if args.lowq_xlsx else [])
    for name, colour, number in review:
        add(name, 'low_quality', colour, labels.get(_stem(name), ''), number)

    draw = random.Random(args.seed)
    confident = [r for r in details if r.get('match') == 'yes' and float(r.get('score') or 0) > 0.9]
    mechanical = [r for r in details if r.get('device_type') == 'mechanical']
    for group, pool, size in (('confident', confident, args.confident), ('mechanical', mechanical, args.mechanical)):
        for row in draw.sample(pool, min(size, len(pool))):
            add(row['picture_name'], group)

    with open(args.out, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        key = (row['group'], row['colour'])
        counts[key] = counts.get(key, 0) + 1
    print(f'{args.out}: {len(rows)} rows, {len({r["picture_name"] for r in rows})} photos', file=sys.stderr)
    for (group, colour), number in sorted(counts.items()):
        print(f'  {group:<12} {colour or "-":<7} {number}', file=sys.stderr)
    print(f'  with a true reading: {sum(bool(r["truth"]) for r in rows)}', file=sys.stderr)
    if left_out:
        print(f'{len(set(left_out))} few-shot example photo(s) left out of the set', file=sys.stderr)
    if missing:
        print(f'{len(missing)} reviewed photo(s) not in the details CSV, left out: '
              + ', '.join(name for _, name in missing[:10]) + (' ...' if len(missing) > 10 else ''), file=sys.stderr)
    if unresolved:
        print(f'{len(unresolved)} filled row(s) whose colour could not be told (theme colours?), left out: '
              + ', '.join(f'{s}!{r}' for s, r, _ in unresolved[:10]), file=sys.stderr)


if __name__ == '__main__':
    main()
