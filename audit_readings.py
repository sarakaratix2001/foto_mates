from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import queue
import re
import sqlite3
import sys
import threading
import time
from evaluation import GROUPS as EVALUATION_GROUPS, evaluate, lines as evaluation_lines
from extract_readings import (BACKENDS as MODEL_BACKENDS, backend as model_backend, OCR_MODEL, PHOTO_TYPES, PROMPT, CROP_FIRST, REREAD_MODE, RETAKE_SCORE, extract_box, find_box, image_workers, photo_bytes, PhotoFileError, photo_retake_reason, _reading_digits, ocr_image,
                              server_slots, extract_blank, extract_no_meter, low_score_reasons, score_reading)

AUDIT_DB = Path(__file__).resolve().parent / os.environ.get('FOTOMATES_AUDIT_DB', 'audit_readings.sqlite3')
BATCH_LOOKUP = 900  # below SQLite's limit of host parameters per statement


def processing_exclusions():
    path = Path(__file__).resolve().parent / 'processing_exclusions.json'
    if not path.exists():
        return set()
    return {row['name'] for row in json.loads(path.read_text(encoding='utf-8'))['files']}

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    picture_name      TEXT PRIMARY KEY,
    contract_id       TEXT,
    date              TEXT,     -- YYYY-MM-DD the photo was taken
    month             TEXT,     -- YYYY-MM
    register          TEXT,     -- ExpAktiv, ExpReaktiv, ExpPik, ...
    saved_reading     TEXT,     -- the reading in the system (from the file name)
    extracted_reading TEXT,     -- every digit read from the photo, leading zeros and decimals included
    extracted_whole   TEXT,     -- the consumption read: no leading zeros, decimals stripped
    extracted_decimals INTEGER, -- how many digits were stripped as decimals
    match             TEXT NOT NULL,  -- yes / no
    status            TEXT NOT NULL,  -- done / missing / no_meter / blank / no_saved_reading / error
    score             REAL,
    retake            INTEGER,
    seconds           REAL,
    error             TEXT,
    read_at           TEXT NOT NULL,
    model             TEXT,     -- the model that read the photo
    visual_decimals   INTEGER,  -- visually identified mechanical fractional wheels, or NULL
    clear             REAL,     -- how clear the model judged the photo, 0..1 (QUALITY; information only)
    quality           TEXT      -- the model's QUALITY answer: CLEAR or POOR, or 0-10 (information only)
) WITHOUT ROWID;  -- the rows live in the picture_name index itself: one copy, smaller file
CREATE INDEX IF NOT EXISTS readings_month ON readings(month, status, match);
CREATE INDEX IF NOT EXISTS readings_contract ON readings(contract_id, month, register);
"""

# Everything else known about a read, kept so a result can be checked or a rule applied
# again later without reading the photo again. Added to old databases on connect.
DETAIL_COLUMNS = (
    ('operator', 'TEXT'),               # who took the photo, from the file name
    ('anomaly_code', 'TEXT'),           # the anomaly the reader chose (57-5701), from the file name
    ('latitude', 'REAL'),               # where, from the file name (41-35034332 -> 41.35034332)
    ('longitude', 'REAL'),
    ('device_type', 'TEXT'),            # digital / mechanical, as the model answered TYPE
    ('digit_probabilities', 'TEXT'),    # one per digit of extracted_reading, e.g. "0.982 1.000 0.931"
    ('second_choices', 'TEXT'),         # per digit, the likeliest other digit: "7:0.120 -:- 0:0.010"
    ('weakest_digit', 'REAL'),          # the lowest of digit_probabilities
    ('code_removed', 'TEXT'),           # register code cut from the front of READING (380, 1580, ...)
    ('code_probabilities', 'TEXT'),     # one per digit of code_removed
    ('reject_reason', 'TEXT'),          # why retake is 1
    ('too_many_digits', 'INTEGER'),     # 1: 12 or more digits read, score set to 0
    ('rolling', 'TEXT'),                # mechanical wheels caught turning, e.g. "6:6-7"
    ('reread', 'INTEGER'),              # 1: read a second time, enlarged, after an unsure first read
    ('other_reading', 'TEXT'),          # the read not kept, its digits as read (zeros kept); NULL for one read
    ('other_score', 'REAL'),            # its score
    ('other_scale', 'TEXT'),            # how it was read: 1 (as the photo is) or 3 (enlarged 3x)
    ('reads_agree', 'INTEGER'),         # 1: both reads gave the same digits, 0: they differ, NULL: one read
    ('photo_sent', 'TEXT'),             # what the model saw: original / enlarged(2x) / enhanced(is_dark,...)
    ('image_width', 'INTEGER'),         # the photo as stored, in pixels
    ('image_height', 'INTEGER'),
    ('blur', 'REAL'),                   # image checks before the read (higher blur value: sharper)
    ('contrast', 'REAL'),
    ('brightness', 'REAL'),
    ('glare_fraction', 'REAL'),
    ('image_flags', 'TEXT'),            # is_blurry, is_dark, has_glare, ... that were true
    ('model_answer', 'TEXT'),           # the model's whole answer, with the timing lines after it
)


def connect(path, **options):
    db = sqlite3.connect(path, **options)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript(SCHEMA)
    present = {column[1] for column in db.execute('PRAGMA table_info(readings)')}
    added = (('model', 'TEXT'), ('extracted_decimals', 'INTEGER'), ('visual_decimals', 'INTEGER'),
             ('clear', 'REAL'), ('quality', 'TEXT'), *DETAIL_COLUMNS)
    for name, kind in added:
        if name not in present:  # a database from before the column
            db.execute(f'ALTER TABLE readings ADD COLUMN {name} {kind}')
    attempts = {column[1] for column in db.execute('PRAGMA table_info(reading_attempts)')}
    for name, kind in added:
        if attempts and name not in attempts:  # the copy of earlier results gets the column too
            db.execute(f'ALTER TABLE reading_attempts ADD COLUMN {name} {kind}')
    # Preserve the previous result of every image when a retry replaces the
    # current row. The trigger also covers writers already running older code.
    db.execute('CREATE TABLE IF NOT EXISTS reading_attempts AS SELECT * FROM readings')
    columns = ','.join(COLUMNS)
    values = ','.join('NEW.' + column for column in COLUMNS)
    # Made again on every connect, so a database from before a new column also records it.
    db.execute('DROP TRIGGER IF EXISTS readings_record_attempt')
    db.execute(f'''CREATE TRIGGER readings_record_attempt
        AFTER INSERT ON readings BEGIN
        INSERT INTO reading_attempts ({columns}) VALUES ({values}); END''')
    db.execute('CREATE INDEX IF NOT EXISTS reading_attempts_picture ON reading_attempts(picture_name, read_at)')
    db.commit()
    return db


MAX_DECIMALS = 3  # the most decimals a display shows (000000.123)


def split_decimals(reading, saved, visual_decimals=None):
    saved = re.sub(r'\D', '', saved or '')
    reading = re.sub(r'\D', '', reading or '')  # a reading stored with its decimal point
    if not reading:
        return None, None, 'no'
    # The saved reading's length places the decimal point, also on mechanical counters: the
    # model's own count of decimal wheels (colour, "-small") is only used when the file name
    # has no saved reading, or a saved 0, to go by.
    if saved and int(saved):
        visual_decimals = None
    if visual_decimals is not None:
        if type(visual_decimals) is not int or not 0 <= visual_decimals < len(reading):
            raise ValueError('visual_decimals must be a nonnegative count smaller than the digit count')
        whole = str(int(reading) // 10 ** visual_decimals)
        return whole, visual_decimals, 'yes' if saved and int(whole) == int(saved) else 'no'
    if not saved:
        return reading.lstrip('0') or '0', None, 'no'
    saved_value = int(saved)
    # A saved 0 has no whole digits: its 0 is one of the display's leading zeros, so what
    # follows them is decimals (000000.19 reads 0.19, not 1.9), at most MAX_DECIMALS of them.
    saved_digits = math.floor(math.log10(saved_value)) + 1 if saved_value else 0
    leading_zeros = len(reading) - len(reading.lstrip('0'))
    decimals = max(0, len(reading) - leading_zeros - saved_digits)
    if not saved_value:
        decimals = min(decimals, MAX_DECIMALS)
    whole = int(reading) // 10 ** decimals
    return str(whole), decimals, 'yes' if whole == saved_value else 'no'


NAME_FIELDS = 8  # Contract_Time_Reader_Reading_AnomalyCode_Register_Lat_Lon


def _coordinate(field):
    """41-35034332 -> 41.35034332; None for anything else."""
    match = re.fullmatch(r'(\d+)-(\d+)', field or '')
    return float(f'{match[1]}.{match[2]}') if match else None


def parse_name(name, month=None):
    """What a photo's name says, read by position:
    Contract_Time_Reader_Reading_AnomalyCode_Register_Lat_Lon, e.g.
    AL0001116_20261023203720_OLSIIB_50_57-5701_ExpAmbPerbashket_41-47428715_19-55966892.jpg.
    Contract and time come from the left; lat, lon, register, anomaly and reading from the
    right; the reader is everything between (a reader's name may hold a _). The date is
    the first 14 digits of the time field; an unreadable one gives date None and month as
    given. The saved reading is the reading field when it is all digits, else None."""
    fields = Path(name).stem.split('_')
    taken = fields[1][:14] if len(fields) > 1 else ''
    try:
        stamp = datetime.strptime(taken, '%Y%m%d%H%M%S') if taken.isdigit() else None
    except ValueError:
        stamp = None
    date, month = (stamp.strftime('%Y-%m-%d'), stamp.strftime('%Y-%m')) if stamp else (None, month)
    if len(fields) < NAME_FIELDS:  # not the usual shape: only the contract and date are sure
        return dict(contract_id=fields[0], date=date, month=month, register=None, saved_reading=None,
                    operator=None, anomaly_code=None, latitude=None, longitude=None)
    reading, anomaly, register, lat, lon = fields[-5:]
    latitude, longitude = _coordinate(lat), _coordinate(lon)
    if not latitude and not longitude:  # 0-0_0-0: no position was recorded
        latitude = longitude = None
    return dict(contract_id=fields[0], date=date, month=month, register=register or None,
                saved_reading=reading if reading.isdigit() else None,
                operator='_'.join(fields[2:-5]) or None, anomaly_code=anomaly or None,
                latitude=latitude, longitude=longitude)


def photos(folders):
    stack = [Path(folder) for folder in folders]
    excluded = processing_exclusions()
    while stack:
        folder = stack.pop()
        found = 0
        print(f'Source scan: listing {folder} (a day folder on the share takes a few minutes) ...', flush=True)
        # Each photo is yielded as soon as it is listed, so reading starts at once.
        # Read through prefetch(), the listing never waits for the reads: the SMB
        # enumeration stays open only as long as the listing itself takes.
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.name not in excluded and Path(entry.name).suffix.lower() in PHOTO_TYPES:
                    found += 1
                    if found % 10000 == 0:
                        print(f'Source scan: {folder}: {found} photos listed so far ...', flush=True)
                    yield Path(entry.path)
        print(f'Source scan: {folder}: {found} eligible photos', flush=True)


def prefetch(iterable, size=0):
    """The items of iterable, produced in a thread of their own: folders on the
    share are listed while their photos are being read, so the model server is
    not left idle for the minutes a 100 000-file listing takes. The queue has no
    limit (size 0): the listing never waits for the reads, so a network folder's
    enumeration is never held open. An error of the listing is raised here,
    where the caller handles it."""
    items, end = queue.Queue(size), object()

    def fill():
        try:
            for item in iterable:
                items.put(item)
        except BaseException as exc:
            items.put(exc)
        finally:
            items.put(end)

    threading.Thread(target=fill, name='folder-listing', daemon=True).start()
    while (item := items.get()) is not end:
        if isinstance(item, BaseException):
            raise item
        yield item


def unread(db, paths, retry_errors=True, only_failed=False):
    """The paths whose picture is not yet in the database (or failed before);
    with only_failed, just the ones that failed before."""
    batch = []
    excluded = processing_exclusions()

    def flush():
        names = [path.name for path in batch]
        marks = ','.join('?' * len(names))
        if only_failed:
            failed = {row[0] for row in db.execute(
                f"SELECT picture_name FROM readings WHERE picture_name IN ({marks}) AND status = 'error'", names)}
            fresh = [path for path in batch if path.name in failed]
            batch.clear()
            return fresh
        skip = "AND status != 'error'" if retry_errors else ''
        known = {row[0] for row in db.execute(
            f'SELECT picture_name FROM readings WHERE picture_name IN ({marks}) {skip}', names)}
        fresh = [path for path in batch if path.name not in known]
        batch.clear()
        return fresh

    for path in paths:
        if path.name in excluded:
            continue
        batch.append(path)
        if len(batch) >= BATCH_LOOKUP:
            yield from flush()
    if batch:
        yield from flush()


def failed_photos(db, folders, names=None):
    """The photos saved with a technical error (status 'error', "Gabim teknik" in
    the reports), found by name inside the given folders instead of by listing
    them: a day folder on the share holds over 100 000 files, and only these few
    are needed. A photo whose date (from its name) matches a folder's YYYY/MM/DD
    goes to that folder unchecked (a file check on the share costs ~2 s; a photo
    missing there fails its read and stays a technical error); any other is
    looked for in every folder, and one not found is reported and left as it is.
    names: only these photos (the ones that failed in the current run)."""
    excluded = processing_exclusions()
    roots = [Path(folder) for folder in folders]
    missing = 0
    for (name,) in db.execute("SELECT picture_name FROM readings WHERE status = 'error' ORDER BY picture_name"):
        if name in excluded or (names is not None and name not in names):
            continue
        # The date opens the name's second field, also in names of an unusual shape
        # (a longer time stamp, empty fields).
        match = re.match(r'[^_]+_(\d{4})(\d{2})(\d{2})', name)
        day = Path(*match.groups()) if match else None
        dated = next((root / name for root in roots if day and root.parts[-3:] == day.parts), None)
        path = dated or next((root / name for root in roots if (root / name).is_file()), None)
        if path:
            yield path
        else:
            missing += 1
            print(f'not found in the given folders, left as a technical error: {name}', flush=True)
    if missing:
        print(f'{missing} failed photo(s) not found in the given folders', flush=True)


def csv_photos(filename):
    """Explicit retry paths, without scanning folders or reading a saved database."""
    seen = set()
    excluded = processing_exclusions()
    with Path(filename).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'picture_name', 'source_path'}.issubset(reader.fieldnames or []):
            raise ValueError('Retry CSV requires picture_name and source_path columns')
        for line, row in enumerate(reader, 2):
            path = Path(row['source_path'])
            name = row['picture_name']
            if not row['source_path'] or not path.is_absolute() or path.name != name:
                raise ValueError(f'Invalid source_path or picture_name on retry CSV line {line}')
            if path.suffix.lower() not in PHOTO_TYPES:
                raise ValueError(f'Unsupported photo extension on retry CSV line {line}')
            if name in seen or name in excluded:
                continue
            seen.add(name)
            yield path


def record_retry_failure(filename, row):
    """Persist failed paths independently of the optional in-memory database."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fields = ['picture_name', 'source_path', 'reason', 'error', 'read_at', 'status']
    header = not filename.exists() or filename.stat().st_size == 0
    with filename.open('a', encoding='utf-8-sig' if header else 'utf-8', newline='') as output:
        writer = csv.DictWriter(output, fields)
        if header:
            writer.writeheader()
        writer.writerow({key: ('technical_error' if key == 'reason' else row.get(key, '')) for key in fields})
        output.flush()
        os.fsync(output.fileno())


# Production values: a photo scored under 0.7 is read once more at 3x; FOTOMATES_REREAD_SCALE=1 turns it off.
REREAD_BELOW = float(os.environ.get('FOTOMATES_REREAD_BELOW', '0.7'))
# Every small photo is already read enlarged to FOTOMATES_MIN_SIDE (2x for 240x400), so the reread goes further.
REREAD_SCALE = float(os.environ.get('FOTOMATES_REREAD_SCALE', '3'))


def reread_rule():
    """The re-read rule in words, for the log: 'reread below 0.7 at 3x' or 'reread off'."""
    crop = f'crop first, kept at {REREAD_BELOW:g} or more, else the whole photo; ' if CROP_FIRST else ''
    if REREAD_SCALE == 1:
        return crop + 'reread off'
    how = (f'from a crop ({REREAD_SCALE:g}x without a usable box)' if REREAD_MODE == 'crop' and not CROP_FIRST
           else f'at {REREAD_SCALE:g}x')
    return crop + f'reread below {REREAD_BELOW:g} {how}' + ('; the most confident read kept' if CROP_FIRST else '')


def read_scored(path, saved, model=None, scale=1.0, crop=False, box=None):
    """One read of a photo with its consumption and its one score. 'scale' says how it was
    read: 1.0 as it is, 3.0 enlarged, or the crop's label, e.g. 'crop(2.71x)'."""
    result = ocr_image(path, model=model, scale=scale, crop=crop, box=box)
    label = re.search(r'crop\([^)]*\)', result.evidence_source or '') if crop else None
    scale = label[0] if label else scale
    cells = _reading_digits(result.text) or ''
    # Scored from the photo alone, then compared with the saved reading.
    parts = score_reading(cells, result.digit_confidence, result.decimal_digits, result.digit_margin,
                          result.digit_runner_up, result.digit_alternatives,
                          result.device_type == 'mechanical') if cells else None
    whole, decimals, match = split_decimals(cells or None, saved, result.decimal_digits)
    return dict(result=result, cells=cells, whole=whole, decimals=decimals, match=match, parts=parts,
                score=parts['total'] if parts else None, scale=scale)


def read_best(path, saved, model=None):
    """The read of a photo. With CROP_FIRST: the photo cut to its display (the box asked in
    a request of its own) is read first and kept when its score is REREAD_BELOW (0.7) or
    more; otherwise the whole photo is read too, then once more at REREAD_SCALE when still
    unsure, and the read with the highest score of all is kept. Without CROP_FIRST: the whole
    photo, and once more enlarged only when it is unsure; the higher score is kept. The saved
    value never decides which read is kept. Adds 'reread' (more than one read made),
    'seconds' (model time of every request made), and, after two reads or more, 'other' (the
    best read not kept) and 'others' (every read not kept)."""
    if not CROP_FIRST:
        return _read_whole(path, saved, model)
    crop, box, seconds = None, None, 0.0
    try:
        box = find_box(path, model)
        seconds += box['seconds']
        if box['box']:
            crop = read_scored(path, saved, model, crop=True, box=box['box'])
            seconds += crop['result'].reading_seconds or 0.0
    except Exception as exc:
        if _file_error(exc) is not None:
            raise  # the share is gone: read_with_retries waits for it
        crop = None  # the box or the crop could not be read: the whole photo is read
    if crop and crop['score'] is not None and crop['score'] >= REREAD_BELOW:
        return dict(crop, reread=False, seconds=seconds, box_answer=box['answer'])
    whole = _read_whole(path, saved, model, unsure=lambda score: max(
        score if score is not None else -1, crop['score'] if crop and crop['score'] is not None else -1) < REREAD_BELOW)
    reads = [crop] if crop else []
    reads += [whole] + ([whole['other']] if whole.get('other') else [])
    reads = [dict(r, other=None) for r in reads]
    best = max(reads, key=lambda r: r['score'] if r['score'] is not None else -1)  # a tie keeps the earlier read
    others = sorted((r for r in reads if r is not best), key=lambda r: r['score'] if r['score'] is not None else -1,
                    reverse=True)
    return dict(best, reread=len(reads) > 1, seconds=seconds + whole['seconds'], other=others[0] if others else None,
                others=others, box_answer=box['answer'] if box else None)


def _read_whole(path, saved, model=None, unsure=None):
    """The whole photo, and once more enlarged when unsure (by default: its score below
    REREAD_BELOW); the read with the higher score is kept, the other in 'other'."""
    unsure = unsure or (lambda score: score is None or score < REREAD_BELOW)
    first = read_scored(path, saved, model)
    first.update(reread=False, seconds=first['result'].reading_seconds or 0.0)
    if REREAD_SCALE == 1:
        return first  # no resizing: the same image read again gives the same answer (temperature 0)
    if not unsure(first['score']):
        return first
    if extract_no_meter(first['result'].text):
        return first  # no meter in the photo: enlarging it shows none either
    try:
        if REREAD_MODE == 'crop':  # the box the first read gave; without a usable one, the 3x frame
            second = read_scored(path, saved, model, REREAD_SCALE, crop=True,
                                 box=extract_box(first['result'].text))
        else:
            second = read_scored(path, saved, model, REREAD_SCALE)
    except Exception:
        return first  # the first read stands when the second fails
    seconds = (first['result'].reading_seconds or 0.0) + (second['result'].reading_seconds or 0.0)
    best, other = (second, first) if (second['score'] or -1) > (first['score'] or -1) else (first, second)
    return dict(best, reread=True, seconds=seconds, other=other)


def read_details(result, reading, reread, other=None, others=None, box_answer=None):
    """What a read knows beyond the reading: per-digit probabilities and second choices,
    the register code cut off, the model's answer and what it was shown, and, after two
    reads, the read not kept (other, from read_best): its digits, score and scale, whether
    both gave the same digits, and its answer after the kept one's."""
    def numbers(values):
        return ' '.join(f'{p:.3f}' for p in values) if values else None
    probabilities = list(result.digit_confidence or [])
    aligned = reading is not None and len(probabilities) == len(reading)
    second = []
    for index, options in enumerate(result.digit_alternatives or [] if aligned else []):
        others = sorted(((p, d) for d, p in (options or {}).items() if d != reading[index]), reverse=True)
        second.append(f'{others[0][1]}:{others[0][0]:.3f}' if others else '-:-')
    image = result.image_report or {}
    flags = [name for name, value in image.items() if name.startswith(('is_', 'has_')) and value]
    answer = re.findall(r'READING:[ \t]*([^\n]*)', result.text, re.IGNORECASE)
    written = re.sub(r'\D', '', answer[-1]) if answer else ''
    return dict(
        device_type=result.device_type,
        digit_probabilities=numbers(probabilities) if aligned else None,
        second_choices=' '.join(second) or None,
        weakest_digit=round(min(probabilities), 3) if aligned and probabilities else None,
        code_removed=written[:len(result.code_confidence or [])] or None,
        code_probabilities=numbers(result.code_confidence),
        rolling=None,  # no longer asked (prompt v2 has no ROLLING line); the column stays
        reread=int(bool(reread)), photo_sent=result.evidence_source,
        image_width=image.get('width'), image_height=image.get('height'),
        blur=image.get('blur_variance'), contrast=image.get('contrast'), brightness=image.get('brightness'),
        glare_fraction=image.get('glare_fraction'), image_flags=','.join(flags) or None,
        other_reading=(other['cells'] or None) if other else None,
        other_score=other['score'] if other else None,
        other_scale=_scale_label(other['scale']) if other else None,
        reads_agree=int((reading or '') == other['cells']) if other else None,
        model_answer=result.text + ''.join(f"\n--- other read ({_scale_label(o['scale'])}) ---\n{o['result'].text}"
                                           for o in (others if others is not None else [other] if other else []))
        + (f"\n--- box (own request) ---\n{box_answer}" if box_answer else ''))


def _scale_label(scale):
    """1.0 -> '1', 3.0 -> '3', a crop's label as it is."""
    return f'{scale:g}' if isinstance(scale, (int, float)) else str(scale)


# A photo whose read fails (the share or the model server gone for a moment) is read again
# this many times, after these waits in seconds, before it is saved as an error.
PHOTO_RETRIES = int(os.environ.get('FOTOMATES_PHOTO_RETRIES', '3'))
RETRY_WAITS = (2, 5, 10)


# A photo share that stops answering (an OSError reading the file, before any model
# request) is an outage, not a failure of the photo: every worker pauses while one probe
# looks at the photo's folder every SHARE_POLL seconds, for at most SHARE_WAIT seconds.
SHARE_POLL = float(os.environ.get('FOTOMATES_SHARE_POLL', '15'))
SHARE_WAIT = float(os.environ.get('FOTOMATES_SHARE_WAIT', '1800'))
SHARE_LOG_EVERY = 60


def _file_error(exc):
    """The PhotoFileError behind exc (a read failure wraps it), or None."""
    while exc is not None:
        if isinstance(exc, PhotoFileError):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def _folder_answers(folder):
    try:
        with os.scandir(folder) as entries:
            next(entries, None)
        return True
    except OSError:
        return False


class ShareGuard:
    """One wait for the whole run while the photo share is gone: workers pause before
    each read, one probe thread checks the failing photo's folder, and all resume
    when it answers (or give up when SHARE_WAIT has passed)."""

    def __init__(self, poll=SHARE_POLL, limit=SHARE_WAIT, answers=_folder_answers):
        self.poll, self.limit, self.answers = poll, limit, answers
        self.open = threading.Event()
        self.open.set()
        self.lock = threading.Lock()
        self.outage = None

    def pause(self):
        """Wait here while an outage is going on."""
        self.open.wait()

    def wait_for(self, path):
        """After the file of path could not be read: None when its folder answers at once
        (the share is fine, the photo's own file failed), True when the share came back
        after a wait (read the photo again), False when SHARE_WAIT passed without it."""
        with self.lock:
            outage = self.outage
            if outage is None:
                if self.answers(path.parent):
                    return None
                outage = self.outage = dict(done=threading.Event(), back=False)
                self.open.clear()
                threading.Thread(target=self._probe, args=(path, outage), daemon=True,
                                 name='share-probe').start()
        outage['done'].wait()
        return outage['back']

    def _probe(self, path, outage):
        started = last_log = time.monotonic()
        print(f'SHARE photo share not answering ({path.parent}); every worker paused, checking '
              f'every {self.poll:g} s for up to {self.limit:g} s', flush=True)
        while True:
            time.sleep(self.poll)
            if self.answers(path.parent):
                outage['back'] = True
                break
            now = time.monotonic()
            if now - started >= self.limit:
                break
            if now - last_log >= SHARE_LOG_EVERY:
                last_log = now
                print(f'SHARE still waiting for the photo share: {now - started:.0f} s of {self.limit:g} s', flush=True)
        photo_bytes.cache_clear()  # never use a copy taken while the share was failing
        waited = time.monotonic() - started
        print(f'SHARE photo share back after {waited:.0f} s; reading resumes' if outage['back'] else
              f'SHARE photo share still gone after {waited:.0f} s; the photos waiting are saved as errors',
              flush=True)
        with self.lock:
            self.outage = None
            self.open.set()
            outage['done'].set()


SHARE = ShareGuard()


def read_with_retries(path, saved, model=None, share=None):
    """read_best with retries. A model-server error is read again up to PHOTO_RETRIES
    times after RETRY_WAITS; a file that cannot be read because the share is gone waits
    for the share instead (not counted), and the photo is then read as if for the first time."""
    share = share or SHARE
    attempt = 0
    while True:
        share.pause()
        try:
            return read_best(path, saved, model)
        except Exception as exc:
            failure = exc
            if _file_error(exc) is not None:
                back = share.wait_for(path)
                if back:
                    attempt = 0
                    continue  # the share answers again: read the photo afresh
                if back is False:
                    raise  # the share stayed gone for SHARE_WAIT: saved as an error
                # back is None: the folder answers, so this photo's own file failed: a normal retry
        if attempt == PHOTO_RETRIES:
            raise failure
        wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
        print(f'retry {attempt + 1}/{PHOTO_RETRIES} in {wait} s: {path.name}: '
              f'{type(failure).__name__}: {str(failure)[:160]}', flush=True)
        attempt += 1
        time.sleep(wait)
        photo_bytes.cache_clear()  # never read again a copy taken while the share was failing


def read_one(path, month, model=None):
    row = parse_name(path.name, month)  # contract, date, register, saved reading, reader, anomaly, place
    started = time.monotonic()
    try:
        read = read_with_retries(path, row['saved_reading'], model)
        result = read['result']
    except Exception as exc:
        return dict(row, model=model_backend(model).model, picture_name=path.name, extracted_reading=None, extracted_whole=None,
                    extracted_decimals=None, match='no', status='error', score=None, retake=None,
                    seconds=round(time.monotonic() - started, 2), error=str(exc)[:2000], source_path=str(path.absolute()))
    reading = _reading_digits(result.text) or None  # as read: leading zeros kept
    if extract_blank(result.text):
        status = 'blank'
    elif extract_no_meter(result.text):
        status = 'no_meter'
    elif not reading:
        status = 'missing'
    elif not row['saved_reading']:
        status = 'no_saved_reading'
    else:
        status = 'done'
    whole, decimals, match, score = read['whole'], read['decimals'], read['match'], read['score']
    quality = re.findall(r'^QUALITY:[ \t]*\**[ \t]*(CLEAR|POOR|\d+)', result.text, re.IGNORECASE | re.MULTILINE)
    reason = photo_retake_reason(reading, result.clear) if reading else None
    if not reason and score is not None and score < RETAKE_SCORE:
        reason = low_score_reasons(read['parts'])[0] if read['parts'] else 'the overall confidence is low'
    details = read_details(result, reading, read.get('reread'), read.get('other'), read.get('others'),
                           read.get('box_answer'))
    too_long = bool(read['parts'] and read['parts'].get('too_long'))
    return dict(row, **details, model=model_backend(model).model, picture_name=path.name, extracted_reading=reading,
                extracted_whole=whole, extracted_decimals=decimals, match=match,
                status=status, score=score, visual_decimals=result.decimal_digits,
                clear=None if result.clear is None else round(result.clear, 3),
                quality=quality[-1].upper() if quality else None,
                retake=int(bool(reason)), reject_reason=reason, too_many_digits=int(too_long),
                # the same, under the names the run's log lines use
                code=details['code_removed'], reason=reason, too_long=too_long,
                seconds=round(read['seconds'] or time.monotonic() - started, 2), error=None)


COLUMNS = ('picture_name', 'contract_id', 'date', 'month', 'register', 'saved_reading',
           'extracted_reading', 'extracted_whole', 'extracted_decimals', 'match', 'status', 'score',
           'retake', 'seconds', 'error', 'read_at', 'model', 'visual_decimals', 'clear', 'quality',
           *(name for name, _ in DETAIL_COLUMNS))


def save(db, row):
    row['read_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    row['model'] = row.get('model') or OCR_MODEL
    db.execute(f'INSERT OR REPLACE INTO readings ({",".join(COLUMNS)}) '
               f'VALUES ({",".join("?" * len(COLUMNS))})', [row.get(c) for c in COLUMNS])


def command_read(args):
    db = connect(args.db)
    choice = model_backend(args.model)
    slots = server_slots(choice.servers)
    workers = args.workers or image_workers(choice, slots)
    print(f'reading with {workers} worker(s) into {args.db}; model {choice.label}; model server slots: '
          f'{slots or "not reported"}; prompt {PROMPT}; {reread_rule()}', flush=True)
    started = time.monotonic()
    count = matched = failed = rejected = codes = too_long = 0
    failed_now = {}  # picture name -> path of the photos saved as an error in this run
    saved_names_so_far = {row[0] for row in db.execute('SELECT picture_name FROM readings')}
    print(f'Resuming with {len(saved_names_so_far)} saved photos; completed photos are skipped.', flush=True)
    model_seconds = 0.0
    pending = set()
    seen = []  
    interruption = None
    last_report_count = 0

    progress_dir = Path(args.word_dir) / 'progress'
    # Failures are always in the database (status error, read again by --only-failed);
    # this CSV of them is written only with --all-reports.
    failure_csv = progress_dir / 'retry_failures.csv' if args.all_reports else None
    if failure_csv:
        print(f'Technical failures are saved immediately to: {failure_csv}', flush=True)
    # Checkpoint reports are written in a thread of their own, with their own
    # database connection (WAL lets it read while results are being written):
    # exporting the whole database (seconds) no longer stops photos being sent.
    reporter = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reports')
    report_job = None

    def write_reports(connection):
        try:
            totals = export_reports(connection, progress_dir, args.month, quiet=True, everything=args.all_reports)
            print(f'Reports saved: {progress_dir} (all saved results in this database for the selected month)', flush=True)
            for line in summary_lines(totals):
                print(f'SUMMARY {line}', flush=True)
        except Exception as exc:
            print(f'Report export could not finish: {exc}. Results remain saved in {args.db}.', flush=True)

    def reader_connection():
        """A connection the export thread can read from. A file database is
        opened again; an in-memory one (--db :memory:) exists only inside this
        connection, so it is copied instead."""
        if str(args.db) in (':memory:', '') or str(args.db).startswith('file::memory'):
            copy = sqlite3.connect(':memory:', check_same_thread=False)
            db.backup(copy)
            return copy
        return sqlite3.connect(args.db, check_same_thread=False)

    def background_reports(reader):
        try:
            write_reports(reader)
        finally:
            reader.close()

    def checkpoint_reports():
        nonlocal last_report_count, report_job
        if report_job is not None and not report_job.done():
            return  # the previous export is still writing; a later checkpoint catches up
        last_report_count = count
        try:
            reader = reader_connection()
        except Exception as exc:
            print(f'Report export could not start: {exc}. Results remain saved in {args.db}.', flush=True)
            return
        report_job = reporter.submit(background_reports, reader)

    def listed(paths):
        for path in paths:
            if not args.no_word:
                seen.append(path.name)
            yield path

    def collect(done):
        nonlocal count, matched, failed, rejected, codes, too_long, model_seconds
        for future in done:
            row = future.result()
            save(db, row)
            db.commit()  # a printed result is already durable, even after a disconnect
            if row['status'] == 'error':
                failed_now[row['picture_name']] = row.get('source_path')
                if failure_csv:
                    record_retry_failure(failure_csv, row)
            else:
                failed_now.pop(row['picture_name'], None)
            saved_names_so_far.add(row['picture_name'])
            count += 1
            matched += row['match'] == 'yes'
            failed += row['status'] == 'error'
            rejected += bool(row.get('retake'))
            codes += bool(row.get('code'))
            too_long += bool(row.get('too_long'))
            model_seconds += row['seconds'] or 0
            if args.verbose:
                result = (('MATCH' if row['match'] == 'yes' else 'MISMATCH') if row['status'] == 'done' else
                          {'no_meter': 'NO_METER_FOUND'}.get(row['status'], row['status'].upper()))
                score = '-' if row['score'] is None else f"{row['score']:.3f}"
                elapsed = time.monotonic() - started
                print(f"[{count} this run | {len(saved_names_so_far)} saved] {row['picture_name']} "
                      f"| saved by system={row['saved_reading'] or '-'} "
                      f"| extracted by AI={row['extracted_whole'] or '-'} "
                      f"| raw digits={row['extracted_reading'] or '-'} | score={score} "
                      f"| {result} | photo={row['seconds']}s | elapsed={elapsed:.0f}s "
                      f"| visual decimals={row['visual_decimals'] if row.get('visual_decimals') is not None else '-'}"
                      + (f" | quality={row['quality'] or '-'} ({100 * row['clear']:.0f}% clear, info only)"
                         if row.get('clear') is not None else '')
                      + (f" | code removed={row['code']}" if row.get('code') else '')
                      + (f" | REJECTED: {row['reason']}" if row.get('reason') else '')
                      + (f" | other={row['other_reading'] or '-'}/"
                         + ('-' if row.get('other_score') is None else f"{row['other_score']:.3f}")
                         if row.get('reads_agree') is not None else ''), flush=True)
                if row['error']:
                    print(f"  error: {row['error']}", flush=True)
            if count % args.progress == 0:
                elapsed = time.monotonic() - started
                rate = count / elapsed
                # failed: photos the model server could not answer at all (not mismatches),
                # shown only when there are any
                print(f'PROGRESS {len(saved_names_so_far)} total saved; {count} read this run in {elapsed / 60:.0f} min, '
                      f'{rate:.2f} photos/s ({rate * 86400:,.0f}/day), {model_seconds / count:.1f} s per photo | '
                      f'{100 * matched / count:.1f}% match, {rejected} rejected ({100 * rejected / count:.1f}%), '
                      f'{codes} register codes removed, {too_long} with 12+ digits'
                      + (f' | {failed} failed (model server gave no answer; --only-failed reads them again)'
                         if failed else ''), flush=True)
        report_every = getattr(args, 'report_every', 1000)
        if report_every and count - last_report_count >= report_every:
            checkpoint_reports()

    def read_all(paths, limit=None):
        nonlocal pending
        for path in paths:
            # Save and print what has finished as it finishes: with many workers
            # the queue is rarely full, so waiting for that would hold every result.
            done, pending = wait(pending, timeout=0 if len(pending) < workers * 2 else None,
                                 return_when=FIRST_COMPLETED)
            collect(done)
            pending.add(pool.submit(read_one, path, args.month, choice.key))
            if limit and count + len(pending) >= limit:
                break

    def errors_again():
        """One more pass over the photos saved as an error in this run."""
        nonlocal pending, count, failed
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
        if not failed_now:
            return
        names = dict(failed_now)
        print(f'Reading again the {len(names)} photo(s) saved as an error in this run', flush=True)
        count -= len(names)  # each is counted once, by its new result
        failed -= len(names)
        if args.folders:
            paths = failed_photos(db, args.folders, set(names))
        else:  # --files-csv: the path each was read from
            paths = (Path(source) for source in names.values() if source)
        read_all(paths)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
        print(f'After the second pass: {len(names) - len(failed_now)} read, {len(failed_now)} still an error',
              flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            # --only-failed: the photos with a technical error, found by name; no
            # listing of the share, and nothing new is read.
            if getattr(args, 'files_csv', None):
                print(f'Reading only explicit paths in: {args.files_csv}', flush=True)
                paths = unread(db, listed(csv_photos(args.files_csv)), retry_errors=not args.skip_errors)
            else:
                paths = (listed(failed_photos(db, args.folders)) if args.only_failed else
                         unread(db, listed(prefetch(photos(args.folders))), retry_errors=not args.skip_errors))
            read_all(paths, args.limit)
            if not args.skip_errors:
                errors_again()
        except (OSError, KeyboardInterrupt) as exc:
            interruption = exc
            print(f'Scan interrupted: {exc or "keyboard interrupt"}. Saving remaining completed requests before exit.', flush=True)
        finally:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
    db.commit()
    reporter.shutdown(wait=True)  # a checkpoint still writing finishes before the final one
    write_reports(db)
    elapsed = time.monotonic() - started
    print(f'{"stopped safely" if interruption else "done"}: {count} photos in {elapsed:.0f} s'
          + (f' ({count / elapsed:.2f} photos/s)' if elapsed else ''), flush=True)
    if not args.no_word:
        try:
            where, params = ('WHERE month IS ?', [_month(args.month)]) if args.month else ('', [])
            saved_names = [row[0] for row in db.execute(f'SELECT picture_name FROM readings {where}', params)]
            print(f'Word report: {write_word_report(db, args.folders, saved_names, count, args.word_dir)}', flush=True)
        except Exception as exc:
            print(f'Word export could not finish: {exc}. CSV reports and database results are retained.', flush=True)
    db.close()
    if interruption:
        raise SystemExit(1)


def folder_rows(db, names):
    """The stored result of each named photo, as dicts; a photo not read yet is left out."""
    rows = []
    for start in range(0, len(names), BATCH_LOOKUP):
        batch = names[start:start + BATCH_LOOKUP]
        cursor = db.execute(
            'SELECT picture_name AS name, status, saved_reading AS saved, extracted_whole AS extracted, '
            'extracted_reading AS all_digits, score, error FROM readings '
            f'WHERE picture_name IN ({",".join("?" * len(batch))})', batch)
        columns = [column[0] for column in cursor.description]
        rows += [dict(zip(columns, row)) for row in cursor]
    return rows


# Files listed per group in the Word report; the CSV beside it lists them all.
WORD_LIST_MAX = int(os.environ.get('FOTOMATES_WORD_LIST_MAX', '2000'))


def write_word_report(db, folders, names, read_now, out_dir):
    from docx import Document  # python-docx; only this report needs it

    rows = folder_rows(db, sorted(set(names)))
    groups = evaluate(dict(row, no_meter=row['status'] == 'no_meter', blank=row['status'] == 'blank') for row in rows)
    errors = {row['name']: row['error'] for row in rows if row['status'] == 'error'}
    now = datetime.now()

    document = Document()
    document.add_heading('FotoMates reading report', level=1)
    document.add_paragraph(f'Date: {now:%Y-%m-%d %H:%M}\nFolder(s): {", ".join(map(str, folders))}\n'
                           f'Model: {OCR_MODEL}\nThe consumption read from each photo (no leading zeros, '
                           'decimals stripped) is compared with the reading saved in the system.')

    table = document.add_table(rows=1, cols=3, style='Light Grid Accent 1')
    for cell, text in zip(table.rows[0].cells, ('', 'Photos', 'Share / note')):
        cell.text = text
    summary_lines = [('Photos in the folder(s)', len(set(names)), ''),
                     ('Photos read', len(rows), f'{read_now} now, {len(rows) - read_now} in an earlier run')]
    summary_lines += [(label, int(count), f'{100 * int(count) / len(rows):.1f}%' if rows else '-')
                      for _, label, count in evaluation_lines(groups)]
    for label, number, note in summary_lines:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = label, str(number), note

   
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lists = out / f'fotomates_report_{now:%Y%m%d_%H%M%S}_files.csv'
    with open(lists, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['group', 'file_name', 'saved_reading', 'extracted_rules_applied', 'all_digits_read',
                         'score', 'error'])
        for key, label in EVALUATION_GROUPS:
            for row in groups['files'][key]:
                writer.writerow([label, row['name'], row['saved'], row['extracted'], row['all_digits'],
                                 '' if row['score'] is None else f'{row["score"]:.3f}', errors.get(row['name'], '')])
    document.add_paragraph(f'Every photo with its group, in full: {lists.name}')

    for key, label in EVALUATION_GROUPS:
        listed = groups['files'][key]
        if key == 'matched' or not listed:
            continue
        document.add_heading(f'{label} ({len(listed)})', level=2)
        if len(listed) > WORD_LIST_MAX:
            document.add_paragraph(f'The first {WORD_LIST_MAX} of {len(listed)}; all of them are in {lists.name}.')
            listed = listed[:WORD_LIST_MAX]
        if key == 'failed':
            for row in listed:
                document.add_paragraph(f'{row["name"]}: {errors.get(row["name"]) or ""}', style='List Bullet')
            continue
        table = document.add_table(rows=1, cols=5, style='Light Grid Accent 1')
        for cell, text in zip(table.rows[0].cells, ('File name', 'Saved reading', 'Extracted (rules applied)',
                                                    'All digits read', 'Score')):
            cell.text = text
        for row in listed:
            cells = table.add_row().cells
            for cell, value in zip(cells, (row['name'], row['saved'], row['extracted'], row['all_digits'],
                                           None if row['score'] is None else f'{row["score"]:.2f}')):
                cell.text = value or ''

    path = out / f'fotomates_report_{now:%Y%m%d_%H%M%S}.docx'
    document.save(path)
    return path


def summary(db, month=None):
    """One row per month: photos, how many were read correctly, mismatched,
    missing or failed, the share correct, contracts, and seconds per photo."""
    where, params = ('WHERE month IS ?', [_month(month)]) if month else ('', [])
    return db.execute(f"""
        SELECT COALESCE(month, 'unknown') AS month,
               COUNT(*) AS photos,
               SUM(match = 'yes') AS read_correctly,
               SUM(status = 'done' AND match = 'no') AS mismatch,
               SUM(status = 'done' AND match = 'no' AND score >= {RETAKE_SCORE}) AS mismatch_high_score,
               SUM(status = 'done' AND match = 'no' AND score < {RETAKE_SCORE}) AS mismatch_low_score,
               SUM(status = 'done' AND match = 'no' AND score IS NULL) AS mismatch_no_score,
               SUM(status = 'missing') AS missing_reading,
               SUM(status = 'no_meter') AS no_meter,
               SUM(status = 'blank') AS blank,
               SUM(status = 'no_saved_reading') AS no_saved_reading,
               SUM(status = 'error') AS failed,
               ROUND(100.0 * SUM(match = 'yes') / NULLIF(SUM(status IN ('done', 'missing')), 0), 2)
                   AS percent_correct,
               COUNT(DISTINCT contract_id) AS contracts,
               ROUND(AVG(seconds), 2) AS seconds_per_photo,
               SUM(reads_agree IS NOT NULL) AS reread_photos,
               SUM(reads_agree = 1) AS reread_agree,
               SUM(reads_agree = 0) AS reread_differ,
               SUM(reads_agree = 0 AND {_saved_match(_FIRST_READ)}) AS differ_first_matched,
               SUM(reads_agree = 0 AND {_saved_match(_SECOND_READ)}) AS differ_second_matched
        FROM readings {where} GROUP BY month ORDER BY month""", params)


# Of two reads, the first (as the photo is, scale 1) and the second (enlarged): the one
# not kept is other_reading, the kept one extracted_reading.
_FIRST_READ = "CASE WHEN other_scale = '1' THEN other_reading ELSE extracted_reading END"
_SECOND_READ = "CASE WHEN other_scale = '1' THEN extracted_reading ELSE other_reading END"


def _saved_match(digits):
    """SQL: 1 when these digits, by the saved reading's length (split_decimals), give
    the saved reading: its leading digits, past the zeros, are the saved number (for a
    saved 0: at most MAX_DECIMALS digits past the zeros)."""
    value, saved = f"ltrim(COALESCE({digits}, ''), '0')", "ltrim(saved_reading, '0')"
    return (f"(CASE WHEN saved_reading IS NULL OR {digits} IS NULL THEN 0 "
            f"WHEN {saved} = '' THEN length({value}) <= {MAX_DECIMALS} "
            f"ELSE length({value}) >= length({saved}) AND substr({value}, 1, length({saved})) = {saved} END)")


LIST_KINDS = ('mismatches', 'missing', 'duplicates', 'all')


def _month(month):
    """'unknown' is how photos without a date in their name are listed."""
    return None if month == 'unknown' else month
# The reading in the system next to the consumption read from the photo, then
# every digit as read and how many of them were stripped as decimals.
_PICKED = ('picture_name, contract_id, date, register, saved_reading, extracted_whole AS read_reading, '
           'match, extracted_reading AS all_digits_read, extracted_decimals AS decimals_stripped, '
           'status, score, seconds, model, clear AS clear_info_only, quality AS quality_info_only')


def list_query(kind, month=None, contract=None):
    """The SQL and parameters of one list: mismatches (both readings present,
    different), missing (no reading, no saved reading, or failed), duplicates
    (a contract and register photographed more than once in a month) or all."""
    if kind not in LIST_KINDS:
        raise ValueError(f'kind must be one of {", ".join(LIST_KINDS)}')
    filters, params = [], []
    if month:
        filters.append('month IS ?')
        params.append(_month(month))
    if contract:
        filters.append('contract_id = ?')
        params.append(contract)
    also = ''.join(' AND ' + f for f in filters)
    if kind == 'duplicates':
        inner = ''.join(' AND ' + f for f in filters)
        return (f"""
            SELECT r.contract_id, r.month, r.register, d.photos, r.picture_name, r.date,
                   r.saved_reading, r.extracted_whole AS read_reading, r.match,
                   r.extracted_reading AS all_digits_read
            FROM readings r JOIN (
                SELECT contract_id, month, register, COUNT(*) AS photos FROM readings
                WHERE contract_id IS NOT NULL {inner}
                GROUP BY contract_id, month, register HAVING COUNT(*) > 1) d
              ON r.contract_id = d.contract_id AND r.month IS d.month AND r.register IS d.register
            ORDER BY r.contract_id, r.month, r.register, r.date""", params)
    if kind == 'mismatches':
        return (f"SELECT {_PICKED} FROM readings WHERE status = 'done' AND match = 'no'{also} "
                'ORDER BY month, contract_id', params)
    if kind == 'missing':
        return (f"SELECT {_PICKED}, error FROM readings "
                f"WHERE status IN ('missing', 'no_meter', 'blank', 'error', 'no_saved_reading'){also} "
                'ORDER BY status, month, contract_id', params)
    where = ' WHERE ' + ' AND '.join(filters) if filters else ''
    return (f'SELECT month, {_PICKED}, retake, error, read_at FROM readings{where} '
            'ORDER BY month, contract_id', params)


# One result line of `read --verbose`.
_LOG_LINE = re.compile(r'^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d )?\[\d+(?: this run \| \d+ saved)?\] (?P<name>\S+) \| saved by system=(?P<saved>\S+) \| extracted by AI=\S+ '
                       r'\| raw digits=(?P<raw>\S+) \| score=(?P<score>\S+) \| (?P<result>[A-Z_]+) '
                       r'\| photo=(?P<seconds>[\d.]+)s')


def command_import_log(args):
    db = connect(args.db)
    known = {name for (name,) in db.execute('SELECT picture_name FROM readings')}
    added = failed = present = 0
    for log in args.logs:
        for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines():
            m = _LOG_LINE.match(line.strip())
            if not m:
                continue
            name, result = m['name'], m['result']
            if result == 'ERROR':
                failed += 1
                continue
            if name in known:
                present += 1
                continue
            row = parse_name(name, args.month)
            raw = None if m['raw'] == '-' else m['raw']
            visual = re.search(r'\| visual decimals=(\d+|-)', line)
            visual_decimals = int(visual[1]) if visual and visual[1] != '-' else None
            whole, decimals, match = split_decimals(raw, row['saved_reading'], visual_decimals)
            status = {'MATCH': 'done', 'MISMATCH': 'done', 'NO_METER_FOUND': 'no_meter'}.get(result, result.lower())
            if status == 'done' and not row['saved_reading']:
                status = 'no_saved_reading'
            score = None if m['score'] == '-' else float(m['score'])
            save(db, dict(row, picture_name=name, extracted_reading=raw, extracted_whole=whole,
                          extracted_decimals=decimals, visual_decimals=visual_decimals, match=match, status=status, score=score,
                          retake=int(score is not None and score < RETAKE_SCORE),
                          seconds=float(m['seconds']), error=None))
            known.add(name)
            added += 1
    db.commit()
    print(f'{added} photo(s) imported into {args.db}; {present} already there; '
          f'{failed} failed in the log and left to read again', flush=True)


NAME_COLUMNS = ('contract_id', 'date', 'month', 'register', 'operator', 'saved_reading', 'anomaly_code',
                'latitude', 'longitude')


def command_reparse(args):
    """Read every stored photo's name again by the current rule (contract, date, month,
    register, reader, saved reading, anomaly code, place) and match its stored reading
    again, without reading any photo again. A read photo's status follows its saved
    reading: done with one, no_saved_reading without. Prints how many rows changed."""
    db = connect(args.db)
    changed = total = 0
    columns = (*NAME_COLUMNS, 'extracted_whole', 'extracted_decimals', 'match', 'status')
    rows = db.execute(f'SELECT picture_name, extracted_reading, visual_decimals, {",".join(columns)} '
                      'FROM readings').fetchall()
    for name, reading, visual_decimals, *before in rows:
        before = dict(zip(columns, before))
        row = parse_name(name, args.month or before['month'])
        whole, decimals, match = split_decimals(reading, row['saved_reading'], visual_decimals)
        status = before['status']
        if status in ('done', 'no_saved_reading'):
            status = 'done' if row['saved_reading'] else 'no_saved_reading'
        after = dict({c: row[c] for c in NAME_COLUMNS}, extracted_whole=whole, extracted_decimals=decimals,
                     match=match, status=status)
        total += 1
        if after != before:
            changed += 1
            db.execute(f'UPDATE readings SET {", ".join(c + " = ?" for c in columns)} WHERE picture_name = ?',
                       [after[c] for c in columns] + [name])
    db.commit()
    print(f'{total} photo names read again, {changed} rows changed', flush=True)


def command_rematch(args):
    """Match every stored reading again by the current rule, without reading
    any photo again."""
    db = connect(args.db)
    changed = total = 0
    for name, reading, saved, before, visual_decimals in db.execute(
            'SELECT picture_name, extracted_reading, saved_reading, match, visual_decimals FROM readings').fetchall():
        whole, decimals, match = split_decimals(reading, saved, visual_decimals)
        db.execute('UPDATE readings SET extracted_whole = ?, extracted_decimals = ?, match = ? '
                   'WHERE picture_name = ?', (whole, decimals, match, name))
        total += 1
        changed += match != before
    db.commit()
    print(f'{total} readings matched again, {changed} changed', flush=True)


# Columns holding digits as read, leading zeros included: Excel opens 0205164 as the number 205164.
DIGIT_TEXT_COLUMNS = {'Raw digits', 'all_digits_read', 'extracted_reading', 'Code removed', 'code_removed'}


def excel_row(columns, row):
    """A report row as Excel should show it: digits starting with 0 are written ="0205164",
    which Excel shows as 0205164 instead of turning it into a number."""
    return [f'="{value}"' if column in DIGIT_TEXT_COLUMNS and isinstance(value, str)
            and len(value) > 1 and value[0] == '0' and value.isdigit() else value
            for column, value in zip(columns, row)]


def write_csv(path, cursor):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            columns = [column[0] for column in cursor.description]
            writer.writerow(columns)
            rows = 0
            for row in cursor:
                writer.writerow(excel_row(columns, row))
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)  # retain the last good report if an export fails
        return rows
    finally:
        temporary.unlink(missing_ok=True)


# A photo's result in words, as the reports show it.
_RESULT = f'''CASE WHEN status = 'error' THEN 'Processing error'
                    WHEN status = 'no_meter' THEN 'No meter'
                    WHEN status = 'blank' THEN 'Blank photo'
                    WHEN status = 'missing' THEN 'No readable digits'
                    WHEN status = 'no_saved_reading' THEN 'No saved reading'
                    WHEN match = 'yes' THEN 'Match'
                    WHEN score >= {RETAKE_SCORE} THEN 'Mismatch - high score'
                    WHEN score < {RETAKE_SCORE} THEN 'Mismatch - low score'
                    ELSE 'Mismatch - no score' END'''

# The lists written only with --all-reports: each is a filter or a total of details_<month>.csv,
# or (reading_attempts) a copy of the database table of earlier attempts.
EXTRA_REPORTS = ('summary.csv', 'comparison.csv', 'mismatches.csv', 'missing.csv', 'duplicates.csv',
                 'mismatches_high_score.csv', 'mismatches_low_score.csv', 'no_digits.csv', 'no_meter.csv',
                 'blank.csv', 'no_saved_reading.csv', 'errors.csv', 'reading_attempts.csv', 'retry_failures.csv')


def export_reports(db, out, month=None, quiet=False, everything=False):
    """all_readings_<month>.csv and details_<month>.csv, one pair per month (a whole year
    would not open in Excel): every photo, details with every column and the result in
    words. everything also writes the EXTRA_REPORTS lists; without it, those left in out
    by an earlier run are removed, so an old list is never taken for a current one.
    Returns the summary, one dict per month, and prints it unless quiet."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    months = [month] if month else [m for (m,) in db.execute('SELECT DISTINCT month FROM readings ORDER BY month')]
    for each in months:
        each = each or 'unknown'
        write_csv(out / f'all_readings_{each}.csv', db.execute(*list_query('all', each)))
        # Every column of every photo, the model's whole answer and the image checks included.
        write_csv(out / f'details_{each}.csv', db.execute(
            f'SELECT *, {_RESULT} AS result FROM readings WHERE month IS ? ORDER BY picture_name',
            [_month(each)]))
    if everything:
        write_extra_reports(db, out, month)
    else:
        for name in EXTRA_REPORTS:
            (out / name).unlink(missing_ok=True)
    cursor = summary(db, month)
    columns = [column[0] for column in cursor.description]
    totals = [dict(zip(columns, row)) for row in cursor]
    if not quiet:
        print(f'{out}: ' + ', '.join(f'all_readings_{m or "unknown"}.csv, details_{m or "unknown"}.csv' for m in months)
              + (', ' + ', '.join(EXTRA_REPORTS[:-1]) if everything else ''))
        for line in summary_lines(totals):
            print('  ' + line)
    return totals


def summary_lines(totals):
    """The summary as log lines, one per month."""
    return [f"{t['month']}: {t['photos']} photos, {t['read_correctly']} read correctly ({t['percent_correct']}%), "
            f"{t['mismatch']} mismatch ({t['mismatch_high_score']} high score, {t['mismatch_low_score']} low score), "
            f"{t['missing_reading']} no reading, {t['no_meter']} no meter, {t['blank']} blank, "
            f"{t['no_saved_reading']} no saved reading, {t['failed']} failed, "
            f"{t['seconds_per_photo']} s per photo"
            + (f"; read twice {t['reread_photos']}: {t['reread_agree']} same digits, {t['reread_differ']} different "
               f"(of these, the first read matched the saved reading {t['differ_first_matched']} times, "
               f"the second {t['differ_second_matched']})" if t.get('reread_photos') else '')
            for t in totals]


def write_extra_reports(db, out, month=None):
    write_csv(out / 'summary.csv', summary(db, month))
    comparison_filter = 'WHERE month IS ?' if month else ''
    write_csv(out / 'comparison.csv', db.execute(f'''
        SELECT picture_name AS "File name", saved_reading AS "Saved by system",
               extracted_whole AS "Extracted by AI", extracted_reading AS "Raw digits",
               extracted_decimals AS "Digits stripped", score AS "Score",
               {_RESULT} AS "Result",
               model AS "AI model", error AS "Error",
               CASE WHEN retake = 1 THEN 'yes' WHEN retake = 0 THEN 'no' END AS "Rejected",
               reject_reason AS "Why rejected", device_type AS "Meter type",
               digit_probabilities AS "Digit probabilities", second_choices AS "Second choices",
               weakest_digit AS "Weakest digit", code_removed AS "Code removed",
               code_probabilities AS "Code probabilities", too_many_digits AS "12+ digits",
               reread AS "Read twice", photo_sent AS "Photo sent to model",
               clear AS "Chance of CLEAR (info only)", quality AS "Quality (info only)",
               operator AS "Operator", latitude AS "Latitude", longitude AS "Longitude"
        FROM readings {comparison_filter} ORDER BY month, picture_name
    ''', [_month(month)] if month else []))
    for kind in ('mismatches', 'missing', 'duplicates'):
        write_csv(out / f'{kind}.csv', db.execute(*list_query(kind, month)))
    mismatch_sql, mismatch_params = list_query('mismatches', month)
    for label, condition in (('high_score', 'score >= ?'), ('low_score', 'score < ?')):
        write_csv(out / f'mismatches_{label}.csv', db.execute(
            f'SELECT * FROM ({mismatch_sql}) WHERE {condition}', [*mismatch_params, RETAKE_SCORE]))
    for label, status in (('no_digits', 'missing'), ('no_meter', 'no_meter'), ('blank', 'blank'),
                          ('no_saved_reading', 'no_saved_reading'), ('errors', 'error')):
        write_csv(out / f'{label}.csv', db.execute(
            f'SELECT month, {_PICKED}, error, read_at FROM readings WHERE status = ?' +
            (' AND month IS ?' if month else '') + ' ORDER BY picture_name',
            [status, _month(month)] if month else [status]))
    write_csv(out / 'reading_attempts.csv', db.execute(
        'SELECT * FROM reading_attempts' + (' WHERE month IS ?' if month else '') +
        ' ORDER BY read_at, picture_name', [_month(month)] if month else []))


def command_report(args):
    db = connect(args.db)
    try:
        export_reports(db, args.out, args.month, everything=args.all_reports)
    finally:
        db.close()


class _LogTee:
    """Everything a run prints, each line starting with the time, written both to the
    terminal and to a log file, so the record survives a dropped connection."""

    def __init__(self, stream, log, lock):
        self.stream, self.log, self.lock, self.line_start = stream, log, lock, True

    def write(self, text):
        with self.lock:
            out = []
            for piece in text.splitlines(keepends=True):
                if self.line_start:
                    out.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S '))
                out.append(piece)
                self.line_start = piece.endswith('\n')
            out = ''.join(out)
            self.stream.write(out)
            self.log.write(out)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # encoding, isatty, ... of the terminal
        return getattr(self.stream, name)


def log_to_file(path):
    """Send stdout and stderr (tracebacks included) to the terminal and to path, timestamped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log, lock = path.open('a', encoding='utf-8', buffering=1), threading.Lock()
    sys.stdout = _LogTee(sys.stdout, log, lock)
    sys.stderr = _LogTee(sys.stderr, log, lock)
    print(f'Log file: {path.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default=str(AUDIT_DB))
    commands = parser.add_subparsers(dest='command', required=True)

    read = commands.add_parser('read', help='read photos and store the results')
    read.add_argument('folders', nargs='*')
    read.add_argument('--files-csv', help='read only paths in a CSV with picture_name and source_path; no folder scan')
    read.add_argument('--month', help='YYYY-MM for photos whose name carries no date')
    read.add_argument('--workers', type=int, help='photos read at once (default: from the model server)')
    read.add_argument('--model', choices=list(MODEL_BACKENDS), default='default',
                      help='model choice: default (FOTOMATES_OCR_URL) or m1, m2, ... from FOTOMATES_MODELS')
    read.add_argument('--limit', type=int, help='stop after this many photos (for a speed test)')
    read.add_argument('--skip-errors', action='store_true', help='do not retry photos that failed before, nor, at the end, '
                           'the ones that failed in this run')
    read.add_argument('--only-failed', action='store_true',
                      help='read again only the photos saved with a technical error ("Gabim teknik"), '
                           'nothing new; they are found by name, the folders are not listed')
    read.add_argument('--progress', type=int, default=100, help='print a progress line every N photos')
    read.add_argument('--report-every', type=int, default=1000,
                      help='refresh CSV reports after N newly processed photos (default: 1000; 0: only on exit)')
    read.add_argument('--verbose', action='store_true', help='print the file name, saved and AI readings, score and result for every photo')
    read.add_argument('--word-dir', default='reports', help='folder for the Word report of the run')
    read.add_argument('--no-word', action='store_true', help='do not write the Word report')
    read.add_argument('--log', help='log file for everything the run prints, each line timestamped '
                                    '(default: <word-dir>/logs/read_<date>_<time>.log)')
    read.add_argument('--no-log', action='store_true', help='print to the terminal only, write no log file')
    read.add_argument('--all-reports', action='store_true',
                      help='besides all_readings_<month>.csv and details_<month>.csv, also write summary, comparison, '
                           'the mismatch and missing lists, duplicates, reading_attempts and retry_failures')
    read.set_defaults(run=command_read)

    imported = commands.add_parser('import-log', help='store the results of a saved `read --verbose` output, '
                                                     'so those photos are not read again')
    imported.add_argument('logs', nargs='+', help='text file(s) with the terminal output of a read --verbose run')
    imported.add_argument('--month', help='YYYY-MM for photos whose name carries no date')
    imported.set_defaults(run=command_import_log)

    rematch = commands.add_parser('rematch', help='match the stored readings again, without reading photos')
    rematch.set_defaults(run=command_rematch)

    reparse = commands.add_parser('reparse', help='read every stored photo name again by the current rule '
                                                  '(and match again), without reading photos')
    reparse.add_argument('--month', help='YYYY-MM for photos whose name carries no readable date '
                                          '(default: the month each row has now)')
    reparse.set_defaults(run=command_reparse)

    report = commands.add_parser('report', help='write the CSV reports')
    report.add_argument('--out', default='reports/audit')
    report.add_argument('--month', help='only this YYYY-MM')
    report.add_argument('--full', action='store_true', help='no effect: every row is always exported (kept for older commands)')
    report.add_argument('--all-reports', action='store_true',
                        help='also write summary, comparison, the mismatch and missing lists, duplicates and reading_attempts')
    report.set_defaults(run=command_report)

    args = parser.parse_args()
    if args.command == 'read':
        if not args.folders and not args.files_csv:
            parser.error('read requires photo folders or --files-csv')
        if args.files_csv and (args.folders or args.only_failed):
            parser.error('--files-csv cannot be combined with folders or --only-failed')
        if not args.no_log:
            log_to_file(args.log or Path(args.word_dir) / 'logs' / f'read_{datetime.now():%Y%m%d_%H%M%S}.log')
            print('Command: ' + ' '.join(sys.argv), flush=True)
    args.run(args)


if __name__ == '__main__':
    main()
