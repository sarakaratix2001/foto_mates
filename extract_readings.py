import base64
import functools
import itertools
import json
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import cv2
import httpx
import numpy as np
from openai import OpenAI
from detect_quality import assess, enhance


def _load_env_file(path: Path) -> None:
    """Settings from .env when run outside Docker (docker compose loads it
    itself); a variable already set in the environment wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        name, sep, value = line.strip().partition('=')
        if sep and name and not name.startswith('#'):
            os.environ.setdefault(name.strip(), value.strip())


_load_env_file(Path(__file__).resolve().parent / '.env')


def _required_env(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise RuntimeError(f'{name} is not set; FotoMates has no default model server (see .env)')
    return value


# Enhancement before reading. auto: only photos assess() flags with one of
# ENHANCE_WHEN are corrected, every other photo is sent as its original file.
# always: every photo is corrected. never: every photo is sent as its original file.
_ENHANCE_MODE = os.environ.get('FOTOMATES_ENHANCE', 'auto').strip().lower()
if _ENHANCE_MODE not in ('auto', 'always', 'never'):
    raise ValueError(f"FOTOMATES_ENHANCE must be 'auto', 'always' or 'never', not {_ENHANCE_MODE!r}")
_ENHANCE_WHEN = tuple(flag.strip() for flag in os.environ.get(
    'FOTOMATES_ENHANCE_WHEN', 'is_low_contrast,is_dark,is_bright,has_glare').split(',') if flag.strip())

# The model's thinking mode is always off: a reading needs no reasoning text,
# and thinking would cost hundreds of tokens per photo. Sent with every request.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


# Longest answer the model may write: the four reply lines fit well within it.
_READ_MAX_TOKENS = int(os.environ.get('FOTOMATES_READ_MAX_TOKENS', '112'))


# One model server, or several separated by commas: each photo goes to whichever
# has a free slot, so their slots add up.
OCR_URLS = [url.strip() for url in _required_env('FOTOMATES_OCR_URL').split(',') if url.strip()]
OCR_URL = OCR_URLS[0]
OCR_MODEL = _required_env('FOTOMATES_OCR_MODEL')
OCR_API_KEY = os.environ.get('FOTOMATES_OCR_API_KEY', 'EMPTY')





# TCP keep-alive on every connection to a model server: with no time limit a
# photo waits as long as the server needs, but a connection that died (network
# drop, server restart) would wait forever. A live server answers the probes
# however busy it is; a dead one misses KEEPCNT of them, INTVL seconds apart,
# after IDLE seconds of silence: the request then fails (re-read later with
# --only-failed or "Re-read failed photos") instead of hanging.
_KEEPALIVE = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)] + [
    (socket.IPPROTO_TCP, getattr(socket, option), value)
    for option, value in (('TCP_KEEPIDLE', 60), ('TCP_KEEPINTVL', 15), ('TCP_KEEPCNT', 4))
    if hasattr(socket, option)]


class _Server:
    """One model server: its client and its /slots address (read once, to size
    a batch). Requests go straight to it: the server queues what it cannot run yet."""

    def __init__(self, url: str):
        self.url = url
        self.client = OpenAI(
            base_url=url,
            api_key=OCR_API_KEY,
            timeout=None,  # no time limit: a photo waits as long as the server needs
            max_retries=0,  # complete() owns retries; the SDK would stack its own on top
            http_client=httpx.Client(timeout=None, transport=httpx.HTTPTransport(socket_options=_KEEPALIVE),
                                     limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)),
        )
        self.slots_url = re.sub(r'/v1/?$', '', url.rstrip('/')) + '/slots'


SERVERS = [_Server(url) for url in OCR_URLS]


class Backend:
    """One model to read with: its name on the server, the server(s) serving it,
    how many photos to send it at once, and the time one photo may take there,
    queueing inside the server included. A choice given its own worker count
    (fixed_workers) sends exactly that many at once and lets the server queue
    them, whether or not the server reports slots."""

    def __init__(self, key: str, label: str, model: str, servers: list,
                 workers: int | None = None):
        self.key, self.label, self.model, self.servers = key, label, model, servers
        self.fixed_workers = workers is not None
        self.workers = workers or int(os.environ.get('FOTOMATES_IMAGE_WORKERS', '1'))


def _backends() -> dict:
    choices = {'default': Backend('default', os.environ.get('FOTOMATES_OCR_LABEL', OCR_MODEL), OCR_MODEL, SERVERS)}
    for number, entry in enumerate(filter(None, map(str.strip, os.environ.get('FOTOMATES_MODELS', '').split(';'))), 1):
        parts = [part.strip() for part in entry.split('|')]
        if len(parts) not in (3, 4, 5):
            raise ValueError(f"FOTOMATES_MODELS entry {entry!r} is not 'label|url|model[|workers]'")
        label, urls, model = parts[:3]
        workers = int(parts[3]) if len(parts) >= 4 else None  # a 5th field, an old time limit, is ignored
        servers = [_Server(url.strip()) for url in urls.split(',') if url.strip()]
        choices[f'm{number}'] = Backend(f'm{number}', label, model, servers, workers)
    return choices


BACKENDS = _backends()
DEFAULT_BACKEND = BACKENDS['default']


def backend(key: str | None) -> Backend:
    """The model choice by its key; the default for None or an unknown key."""
    return BACKENDS.get(key or 'default', DEFAULT_BACKEND)

_MODEL_LOADING_WAIT = float(os.environ.get('FOTOMATES_MODEL_LOADING_WAIT', '900'))
_MODEL_LOADING_POLL = float(os.environ.get('FOTOMATES_MODEL_LOADING_POLL', '15'))


def _is_model_loading(error: Exception) -> bool:
    if getattr(error, 'status_code', None) != 503:
        return False
    return 'loading model' in str(error).lower()


def complete(server=None, **request):
    """One request per photo, no retries: a failed read is reported as failed.
    The one wait is for a model server still loading its model at start."""
    loading_deadline = time.monotonic() + _MODEL_LOADING_WAIT
    while True:
        try:
            return (server or SERVERS[0]).client.chat.completions.create(**request)
        except Exception as error:
            if _is_model_loading(error) and time.monotonic() < loading_deadline:
                time.sleep(_MODEL_LOADING_POLL)
                continue
            raise


# Timeout of the /slots request that counts a server's slots when a batch starts.
_SLOTS_COUNT_TIMEOUT = float(os.environ.get('FOTOMATES_SLOTS_COUNT_TIMEOUT', '5'))
# Time the warm-up request at start may take on each server.
_WARM_UP_TIMEOUT = float(os.environ.get('FOTOMATES_WARM_UP_TIMEOUT', '60'))
_TURN = itertools.count()  # which server of a model takes the next photo


# Requests a server without a /slots endpoint (vLLM) runs at once, set by hand:
# vLLM does not publish it, and its GPU memory, not --max-num-seqs, decides it
# (vllm:num_requests_running on /metrics under full load shows the number).
_SERVER_SLOTS = int(os.environ.get('FOTOMATES_SERVER_SLOTS', '0'))


def server_slots(servers: list | None = None) -> int | None:
    """How many requests the model servers can work on at once, all together:
    each server's /slots count (llama.cpp), else FOTOMATES_SERVER_SLOTS for it;
    None when neither says."""
    counts = []
    for server in servers or SERVERS:
        try:
            with urllib.request.urlopen(server.slots_url, timeout=_SLOTS_COUNT_TIMEOUT) as response:
                counts.append(len(json.load(response)))
        except Exception:
            counts.append(_SERVER_SLOTS)
    return sum(counts) or None


# Photos in flight per model slot, on a server that reports its slots: the
# second one is prepared (file share, decode, enhance, encode) and queued at
# the server while the slot is busy, so the slot never idles on file reading.
WORKERS_PER_SLOT = int(os.environ.get('FOTOMATES_WORKERS_PER_SLOT', '2'))


# The photo file types the model server reads as they are.
PHOTO_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}


@functools.lru_cache(maxsize=512)
def photo_bytes(path: Path) -> bytes:
    """A photo file's bytes, read once: judging its quality, sending it and a
    second, enlarged read all use them. On the file share a read costs seconds,
    and it was made two or three times per photo. The last 512 photos are kept
    (~35 MB), more than the photos in flight."""
    return path.read_bytes()


def photo_data_url(path: Path) -> str:
    """The photo file exactly as it is on disk, as a data URL: no decoding,
    no enhancement, no resizing, no re-encoding."""
    kind = PHOTO_TYPES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.suffix or 'no extension'}: the model server reads "
                         + ", ".join(PHOTO_TYPES) + " photos")
    return f"data:{kind};base64," + base64.b64encode(photo_bytes(path)).decode("ascii")


def png_data_url(pixels: np.ndarray) -> str:
    """An enhanced photo as lossless PNG: only a corrected photo is encoded."""
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError("Could not encode the enhanced photo")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_image(path: Path) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(photo_bytes(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("Could not decode image")
    return pixels


# Without a READING line, the longest run of this many digits in the answer is the reading.
_MAX_READING_DIGITS = int(os.environ.get('FOTOMATES_MAX_READING_DIGITS', '12'))
_MIN_READING_DIGITS = int(os.environ.get('FOTOMATES_MIN_READING_DIGITS', '6'))


def _digit_count(reading: str) -> int:
    return sum(ch.isdigit() for ch in reading)


# A register code: the small digits a register display shows beside the
# consumption (1.8.0, 15.8.0, 2.8.1, ...). It is never part of the reading.
# Three or more short groups joined by dots: a reading has at most one decimal
# point, so this shape at the front is always a code.
_DOTTED_CODE = re.compile(r"^(?:\d{1,2}[.,]){2,}\d{1,2}(?=[.,]|$)[.,]?")
# The same code copied without its dots. An energy register is A.8.T (the 8 is
# sometimes misread as 5-7), and the display pads the consumption with leading
# zeros, so such a prefix followed by a zero and five more digits is a code glued
# onto the reading (15800061724 -> 0061724), not the start of a real consumption.
_GLUED_CODE = re.compile(r"^(?:1\d|[1-9])[5-8]\d(?=0\d{5})")


def _strip_code(digits: str, code: str) -> str:
    """The reading without a register code the model glued onto its front."""
    code_digits = re.sub(r"\D", "", code)
    if len(code_digits) >= 2 and digits.startswith(code_digits) \
            and len(digits) - len(code_digits) >= 4:
        return digits[len(code_digits):]
    glued = _GLUED_CODE.match(digits)
    return digits[glued.end():] if glued else digits


def extract_reading(text: str) -> str:
    reading = _reading_digits(text)
    stripped = reading.lstrip("0")
    return stripped or (reading and "0")


def extract_device_type(text: str) -> str | None:
    types = re.findall(r"TYPE:[ \t]*\**[ \t]*(DIGITAL|MECHANICAL)", text, re.IGNORECASE)
    if not types:
        return None
    return 'digital' if types[-1].upper() == 'DIGITAL' else 'mechanical'


def extract_window_decimals(text: str) -> int | None:
    """Decimal wheels by the one rule: at the right end, a wheel whose digit is
    smaller (-small) or whose window has another colour than most of the wheels
    (WINDOWS: dark dark dark dark dark red gives 1; ... dark-small gives 1; a frame,
    dot or printed 0.1 alone gives 0). None without one window word per digit."""
    lines = re.findall(r'^WINDOWS:[ \t]*([^\n]*)', text, re.IGNORECASE | re.MULTILINE)
    windows = lines[-1].lower().split() if lines else []
    digits = _reading_digits(text)
    if not digits or len(windows) != len(digits):
        return None
    colours = [word.split('-')[0] for word in windows]
    colour = max(set(colours), key=colours.count)  # the integer wheels' colour
    marked = [word.split('-')[0] != colour or 'small' in word.split('-')[1:] for word in windows]
    decimals = len(marked) - next((i + 1 for i in range(len(marked) - 1, -1, -1) if not marked[i]), 0)
    # Only a short group after unmarked integer wheels: one odd wheel in the
    # middle does not make every wheel after it a decimal.
    return decimals if decimals < len(windows) and not any(marked[:len(marked) - decimals]) else 0


def extract_decimal_digits(text: str) -> int | None:
    """Mechanical decimal wheels identified visually, never from the saved value:
    from the WINDOWS line by the one rule (smaller digit or a box of another
    colour); the model's DECIMALS answer only when WINDOWS is missing or does
    not have one word per digit."""
    if extract_device_type(text) != 'mechanical':
        return None
    answers = re.findall(r'^DECIMALS:[ \t]*(\d+|UNKNOWN)[ \t]*$', text, re.IGNORECASE | re.MULTILINE)
    windows = extract_window_decimals(text)
    if windows is not None:
        count = windows
    elif not answers or answers[-1].upper() == 'UNKNOWN':
        count = None
    else:
        count = int(answers[-1])
    digits = _reading_digits(text)
    return count if count is not None and digits and 0 <= count < len(digits) else None


def extract_no_meter(text: str) -> bool:
    """True when the model answered that the photo shows no meter at all."""
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text or '', re.IGNORECASE)
    return bool(answers) and re.match(r"\s*NO[\s_-]*METER", answers[-1], re.IGNORECASE) is not None


def extract_rolling(text: str) -> list[dict]:

    if extract_device_type(text) != 'mechanical':
        return []
    lines = re.findall(r"ROLLING:[ \t]*\**[ \t]*([^\n*]*)", text, re.IGNORECASE)
    if not lines or re.fullmatch(r"\s*(NONE|NO|-|N/?A)?\s*\.?\s*", lines[-1], re.IGNORECASE):
        return []
    wheels = [dict(position=int(position), lower=int(lower), upper=int(upper))
              for position, lower, upper in re.findall(r"(\d+)\s*:\s*(\d)\s*[-/]\s*(\d)", lines[-1])]
    return wheels or [dict(position=None, lower=None, upper=None)]


def _reading_digits(text: str) -> str:
    """The register digits the model wrote, leading zeros kept, register code
    removed, no decimal point; "" when there is no usable reading."""
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text, re.IGNORECASE)
    if answers:
        reading = ""
        for token in answers[-1].split():
            token = _DOTTED_CODE.sub("", token).strip(".,")
            if not re.search(r"\d", token):  # a unit such as kWh, or UNREADABLE
                continue
            if not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", token):
                return ""
            reading += re.sub(r"[.,]", "", token)
        codes = re.findall(r"CODE:[ \t]*\**[ \t]*([^\n*]*)", text, re.IGNORECASE)
        # A wheel counter has no LCD register code. Its first wheels must not
        # disappear merely because their values resemble a glued register code.
        if extract_device_type(text) != 'mechanical':
            reading = _strip_code(reading, codes[-1] if codes else "")
    else:
        runs = []
        for match in re.finditer(r"\d[\d.,]*\d|\d", text):
            digits = re.sub(r"\D", "", match.group(0))
            if not _MIN_READING_DIGITS <= len(digits) <= _MAX_READING_DIGITS:
                continue
            runs.append(digits)
        reading = _strip_code(max(runs, key=len), "") if runs else ""
    if not re.fullmatch(r"[0-9]+", reading):
        return ""
    if extract_device_type(text) == 'mechanical':
        cells = re.findall(r'^CELLS:[ \t]*([^\n]*)', text, re.IGNORECASE | re.MULTILINE)
        if cells and re.fullmatch(r'[ \t]*\d(?:[ \t]+\d)*[ \t]*', cells[-1]):
            # Refuse a shortened or changed concatenation; the reread path can
            # try again rather than reporting a confident missing-wheel value.
            if ''.join(cells[-1].split()) != reading:
                return ""
    if _digit_count(reading) > _MAX_READING_DIGITS:
        return ""
    return reading


@dataclass
class OcrResult:
    text: str
    evidence_source: str = "unknown"
    confidence: float | None = None          # the weakest digit's probability, 0..1
    digit_confidence: list | None = None     # one probability per digit read
    digit_margin: list | None = None         # per digit: its probability minus the runner-up digit's
    digit_runner_up: list | None = None      # per digit: the most likely other digit, or None
    digit_alternatives: list | None = None   # per digit: {digit: probability} of the model's top alternatives
    rolling: list | None = None              # mechanical wheels caught between two numbers
    device_type: str | None = None           # 'mechanical', 'digital', or None when not said
    decimal_digits: int | None = None        # visually marked fractional wheels; 0 means none
    reading_seconds: float | None = None     # the model reading the photo plus digit extraction
    sent_png: bytes | None = None            # the enhanced photo the model read; None: it read the original


# A photo whose score S is below this is flagged for a retake, with the reasons.
RETAKE_SCORE = float(os.environ.get('FOTOMATES_RETAKE_SCORE', '0.5'))
# A photo whose score S is below this gets a note on its quality and speed.
QUALITY_SCORE = float(os.environ.get('FOTOMATES_QUALITY_SCORE', '0.9'))
# ... or whose least certain digit is below this. On 224 checked reads, 2 of the
# 40 whose every digit was at least 95% sure were wrong, against 7 of the 58
# with a score of at least 0.9: one doubtful digit hides in a high score.
QUALITY_DIGIT = float(os.environ.get('FOTOMATES_QUALITY_DIGIT', '0.95'))


def _digit_reason(parts: dict) -> list[str]:
    """The least certain digit, how sure the model was of it and what it could also be."""
    if not parts or parts.get('weakest') is None:
        return ['no digit probabilities from the model']
    position, value, rival = parts['weakest_position'], parts.get('weakest_digit'), parts.get('weakest_runner_up')
    if position:
        suffix = 'th' if 10 <= position % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(position % 10, 'th')
        which = f"the {position}{suffix} digit" + (f" ({value})" if value is not None else '')
    else:
        which = 'a leading zero'
    if rival is not None:
        return [f"{which} is only {parts['weakest']:.2f} ahead of a {rival}: check it on the meter"]
    return [f"{which} is only {100 * parts['weakest']:.0f}% sure: check it on the meter"]


def low_score_reasons(parts: dict) -> list[str]:
    """Why a score is low: its least certain digit."""
    return _digit_reason(parts)


def quality_reasons(parts: dict) -> list[str]:
    """What kept a score below 1: its least certain digit."""
    return _digit_reason(parts)


def score_reading(cells: str, per_digit: list | None, decimals: int | None,
                  margins: list | None = None, runner_ups: list | None = None,
                  alternatives: list | None = None, mechanical: bool = False) -> dict:

    cells, per_digit = cells or '', per_digit or []
    lead = len(cells) - len(cells.lstrip('0'))
    end = len(cells) - (decimals or 0)
    consumption = cells[lead:end] if lead < end else (cells and '0')
    aligned = len(per_digit) == len(cells)
    scored = []  # (margin, index, rival)
    if aligned and alternatives and len(alternatives) == len(cells):
        # The consumption's digits only; a consumption of 0 is its last whole cell.
        digits_counted = range(lead, end) if lead < end else range(max(end - 1, 0), max(end, 0))
        for index in digits_counted:
            written = cells[index]
            # At the first digit of the consumption a 0 changes nothing: it is a leading zero more.
            turning = mechanical and index == end - 1  # the last whole wheel of a counter
            changes = {digit: p for digit, p in (alternatives[index] or {}).items()
                       if digit != written and not (index == lead and digit == '0')
                       and not (turning and (int(digit) - int(written)) % 10 in (1, 9))}
            rival = max(changes, key=changes.get, default=None)
            scored.append((per_digit[index] - (changes[rival] if rival else 0.0), index, rival))
    elif aligned:
        values = margins if margins and len(margins) == len(per_digit) else per_digit
        rivals = runner_ups if runner_ups and len(runner_ups) == len(per_digit) else [None] * len(per_digit)
        for index in (range(lead, end) if lead < end else range(len(cells))):
            if index == lead and rivals[index] == '0':  # only a leading zero more: its probability alone
                scored.append((per_digit[index], index, None))
            else:
                scored.append((values[index], index, rivals[index]))
    weakest = position = digit = rival = None
    if scored:
        weakest, index, rival = min(scored)
        # Counted in the reading as shown, without its leading zeros; 0 for a leading zero.
        position, digit = (index + 1 - lead if index >= lead else 0), cells[index]
    elif per_digit:
        weakest = min(per_digit)
    total = 0.0 if not consumption else weakest
    return dict(total=None if total is None else round(max(total, 0.0), 3),
                weakest=None if weakest is None else round(max(weakest, 0.0), 3),
                weakest_position=position, weakest_digit=digit, weakest_runner_up=rival)


def _digit_confidence(choice) -> tuple[float | None, list, list, list, list]:
    """Per digit of the READING line: the probability of the digit written, its
    margin over the most likely other digit (the runner-up among the model's top
    alternatives), and that runner-up digit. A 5 at 78% with 9 at 22% has a
    margin of 0.56: the model was torn between the two. Returns (weakest
    probability, probabilities, margins, runner-ups and alternatives)."""
    tokens = getattr(getattr(choice, 'logprobs', None), 'content', None) or []
    joined = ''.join(token.token or '' for token in tokens)
    marker = joined.upper().rfind('READING:')
    if marker < 0:
        return None, [], [], [], []
    begin, offset = marker + len('READING:'), 0
    probabilities, margins, runner_ups, alternatives = [], [], [], []
    for token in tokens:
        piece, logprob = token.token or '', token.logprob
        end = offset + len(piece)
        if end > begin and logprob is not None:
            digits = sum(ch.isdigit() for ch in piece)
            if digits:
                probability = math.exp(logprob)
                options = {}
                for alt in getattr(token, 'top_logprobs', None) or []:
                    if alt.logprob is not None and alt.token.strip().isdigit() and len(alt.token.strip()) == 1:
                        options[alt.token.strip()] = max(options.get(alt.token.strip(), 0.0), math.exp(alt.logprob))
                others = [(p, d) for d, p in options.items() if d != piece.strip()]
                second, rival = max(others, default=(0.0, None))
                probabilities.extend([probability] * digits)
                margins.extend([max(0.0, probability - second)] * digits)
                runner_ups.extend([rival] * digits)
                alternatives.extend([options] * digits)
            elif probabilities and piece.strip() and piece.strip() not in '.,':
                break  # the number is over
        offset = end
    return (min(probabilities) if probabilities else None), probabilities, margins, runner_ups, alternatives


class OcrFailure(Exception):
    """A read that failed, carrying the per-stage timings up to the failure."""

    def __init__(self, message: str, breakdown: str):
        super().__init__(message)
        self.breakdown = breakdown


def _failure_reason(error: Exception) -> str:
    
    return f'{type(error).__name__}: {error}'


_SYSTEM_PROMPT = (
    "You transcribe meter readings from photographs. Preserve every visible digit cell, "
    "especially the first and last mechanical wheels. Never drop a wheel as a decimal. "
    "A last wheel set apart by its own coloured or white box, frame, or window, or "
    "labelled 0.1 above it, is a fractional wheel: keep it and count it in DECIMALS. "
    "For a mechanical counter, read only the printed digits on the number wheels inside "
    "the counter window; never read handwritten, painted, or sticker numbers. "
    "For a mechanical counter, first transcribe each physical wheel separately, "
    "then concatenate those same digits into READING without losing the final wheel. "
    "Reply with exactly these six "
    "lines and nothing else, no explanation:\n"
    "CELLS: <MECHANICAL: each wheel digit from left to right separated by spaces; DIGITAL: NONE>\n"
    "WINDOWS: <MECHANICAL: each wheel's window colour from left to right, with -box when "
    "it has its own frame; DIGITAL: NONE>\n"
    "READING: <every digit cell of the main row of large digits, UNREADABLE, or NO METER>\n"
    "TYPE: <DIGITAL for an LCD or LED display, MECHANICAL for a counter of turning number wheels>\n"
    "ROLLING: <only for MECHANICAL: wheels caught turning between two numbers, as "
    "position:lower-upper; NONE for DIGITAL or when no wheel is turning>\n"
    "DECIMALS: <MECHANICAL: visually marked fractional wheel count, 0 if none; "
    "UNKNOWN for DIGITAL or if fractional markings cannot be determined>")
_USER_PROMPT = (
    "Read the consumption register on this meter: the main row of large digits on the display.\n"
    "First find that row: the longest line of the largest digits, side by side. Then read "
    "every digit cell in it from the first to the very last, one digit per cell, including "
    "leading zeros and the last cell or two at the right end, even when that cell is boxed, "
    "in another colour, slightly smaller, faint, or under glare. Count the cells and make "
    "sure you wrote one digit for each: never skip a cell and never write a digit twice.\n"
    "Many displays also show a register code: a short group of small digits, usually two or "
    "three numbers joined by dots, set apart at the left or top edge of the display and smaller "
    "than the main row. Its numbers can be anything: recognise it by its size, its dots and its "
    "place, never by its value. Never write it in READING, even when it sits on the same line "
    "as the main row or its dots are hard to see. The main row's own leading zeros and first "
    "digits are never a code: they are as large as the rest of the row, so read them as part of it.\n"
    "Leave out every digit that is not in that row: the register code, a single small 1 or "
    "index before the row, a tariff or register number, units, the meter serial, the model code, the barcode, the "
    "year and every number printed outside the display. Never add a digit you cannot see.\n"
    "On a MECHANICAL counter the reading is only the printed digits on the turning number "
    "wheels seen through the counter window. Ignore every handwritten number: digits "
    "written in pen, marker, chalk, or paint on the meter face, the glass, the cover, a "
    "label, or a sticker, even when they are larger, darker, or closer to the camera than "
    "the wheels, and even when they look like a reading. Wheel digits are all the same "
    "printed font and size, evenly spaced, each in its own wheel slot; a number that is "
    "not inside those slots is never part of READING.\n"
    "Read each digit from its own strokes or segments, not from what you expect: 7 has no "
    "lower-left or bottom segment while 9 has both a top-left segment and a bottom; 8 has a "
    "middle segment that 0 lacks; 6 has a lower-left segment that 5 lacks; 1 is only one "
    "vertical stroke on the right.\n"
    "5 and 9 are easily confused: check the upper right. A 9 has an upper-right segment "
    "or stroke that closes its top into a loop; a 5 has none there, its top is a flat bar "
    "with a stroke down the left side only, so its upper half is open on the right. On a "
    "printed wheel, 5 has a straight top bar and a sharp corner at the upper left, 9 a "
    "round closed loop on top. Write 9 only when you see that closed upper loop; when the "
    "upper right is open, it is a 5, even if the lower half looks round. In the same way "
    "a 6 closes its lower half into a loop on the left, while a 5 is open at the lower "
    "left.\n"
    "Give the digits as one unbroken string with no dot and no comma. When the row cannot "
    "be read, write UNREADABLE; when the photo shows no meter at all, write NO METER.\n"
    "TYPE says what shows the register: DIGITAL for an LCD or LED display of segment "
    "digits, MECHANICAL for a counter of printed number wheels behind a window.\n"
    "For a MECHANICAL counter, locate every wheel aperture from the left edge to the "
    "right edge before reading. Transcribe one digit per wheel, including a partially "
    "visible or turning final wheel: six visible wheels require six digits, not five. "
    "Do not mistake the final wheel for a unit, serial digit, or decimal to omit. "
    "Before deciding DECIMALS, look at the window of every wheel and describe it on the "
    "WINDOWS line, one word per wheel from left to right, in the same order as CELLS: "
    "dark, white, red, orange, yellow, or other for the window's background colour, "
    "followed by -box when that wheel has a frame or border of its own, e.g. "
    "WINDOWS: dark dark dark dark dark white-box. Then DECIMALS is the number of "
    "wheels at the right end that the WINDOWS line, a separator, or a 0.1 scale sets "
    "apart from the wheels before them. "
    "A row of wheels whose windows all look alike, with no separator and no 0.1 scale, "
    "is an integer register: DECIMALS: 0. Neither position at the right end nor the "
    "number of digits in an expected reading is evidence of decimals; a wheel is not "
    "fractional merely because it is last. "
    "Wheels at the right end are a marked fractional section when they are set apart "
    "from the integer wheels by any of: a different background or window colour (red, "
    "pink, magenta, orange, yellow, or white/light where the other wheels are dark, or "
    "dark where they are light); a box, frame, or border of their own; a decimal point "
    "or comma before them; or a scale printed above the wheels that labels them 0.1, "
    "x0.1, or 1/10. This holds even when the digit is the same height; glare may make "
    "a coloured panel look pale or white. Include those wheels in READING and count "
    "them in DECIMALS. Example: black wheels 0 4 7 2 9 3 followed by one red wheel 6 gives "
    "CELLS: 0 4 7 2 9 3 6, READING: 0472936, DECIMALS: 1. White digits on a dark band "
    "0 4 2 9 6 followed by a 7 in a white box of its own give CELLS: 0 4 2 9 6 7, "
    "READING: 042967, DECIMALS: 1. Black wheels "
    "3 8 1 2 5 6 with no separator or marked fractional section give CELLS: "
    "3 8 1 2 5 6, READING: 381256, DECIMALS: 0. "
    "Keep fractional wheels in READING too; DECIMALS only describes them. If markings "
    "are unclear write DECIMALS: UNKNOWN, never invent a separator. For DIGITAL write "
    "DECIMALS: UNKNOWN.\n"
    "Only on a MECHANICAL counter can a wheel be caught turning, showing parts of two "
    "numbers at once. Write the lower of the two in READING (9 when it turns from 9 to 0), "
    "and report every such wheel on the ROLLING line as its position in READING counted "
    "from the left, then the two numbers, e.g. 6:6-7. For a DIGITAL display always write "
    "ROLLING: NONE.\n"
    "When the counter is too blurred, dim, or washed out to read every wheel, write "
    "UNREADABLE instead of guessing digits.\n"
    "Before replying, check the first and final wheel and verify that the READING "
    "digit count equals the visible wheel count. Never infer a missing digit from a "
    "stored reading, and never omit a visible wheel to make the length fit.\n"
    "Copy every digit from CELLS into READING in order; do not shorten it. "
    "A partially visible bottom or top of the final wheel still belongs to that "
    "wheel. If a required wheel cannot be read, use UNREADABLE rather than "
    "returning a shorter reading.\n"
    "Reply with exactly six lines:\n"
    "CELLS: mechanical wheel digits separated by spaces, or NONE for DIGITAL\n"
    "WINDOWS: one window word per wheel, as described above, or NONE for DIGITAL\n"
    "READING: every digit cell of the main row, UNREADABLE, or NO METER\n"
    "TYPE: DIGITAL or MECHANICAL\n"
    "ROLLING: position:lower-upper for each turning wheel, comma separated, or NONE\n"
    "DECIMALS: visually marked fractional wheel count, 0 when none, or UNKNOWN.")

# The same rules restructured (role and output contract in the system prompt;
# task, what to read, what to ignore, digit shapes, decimals, rolling wheels and
# when unsure in the user prompt), each said once: about half the tokens, and
# TYPE first so the lines after it follow from it. The default;
# FOTOMATES_PROMPT=full goes back to the prompt above.
_COMPACT_SYSTEM_PROMPT = (
    "You read electricity meter registers from photos. Answer with exactly these six "
    "lines and nothing else:\n"
    "TYPE: DIGITAL (LCD or LED segments) or MECHANICAL (printed number wheels)\n"
    "CELLS: MECHANICAL: one digit per wheel, left to right, space separated; DIGITAL: NONE\n"
    "WINDOWS: MECHANICAL: one word per wheel for its window colour (dark, white, red, "
    "orange, yellow, other), with -small when its digit is smaller than the other "
    "wheels' digits; DIGITAL: NONE\n"
    "READING: every digit of the main row as one string with no dot or comma, or "
    "UNREADABLE, or NO METER\n"
    "ROLLING: MECHANICAL wheels caught between two numbers as position:lower-upper, "
    "comma separated; otherwise NONE\n"
    "DECIMALS: MECHANICAL: number of wheels at the right end with a smaller digit or "
    "a box of another colour, 0 if none; DIGITAL: UNKNOWN")
_COMPACT_USER_PROMPT = (
    "TASK\n"
    "Read the consumption register: the main row of the largest digits on the display.\n"
    "\n"
    "WHAT TO READ\n"
    "- Every digit cell of that row, first to last, one digit per cell: leading zeros, and "
    "the last cell or two even when boxed, coloured, smaller, faint or under glare. Never "
    "skip or repeat a cell: READING has as many digits as the row has cells.\n"
    "- MECHANICAL: only the printed digits on the wheels inside the counter window, "
    "including a partly visible or turning last wheel; CELLS and READING hold the same "
    "digits in the same order.\n"
    "\n"
    "WHAT TO IGNORE\n"
    "- The register code: a short group of smaller digits, often joined by dots, at the "
    "left or top edge of the display. Recognise it by size, dots and place, never by "
    "value. The row's own leading zeros are as large as the row and belong to it.\n"
    "- A small 1 or index before the row, tariff or register numbers, units, serial, model "
    "code, barcode, year and anything printed outside the display.\n"
    "- Handwritten, painted or sticker numbers anywhere (face, glass, cover, label), even "
    "when larger, darker or closer to the camera than the wheels.\n"
    "\n"
    "DIGIT SHAPES (read each digit from its own strokes, not from what you expect)\n"
    "- 7 has no lower-left stroke and no bottom; 9 has a top-left stroke and a bottom. 8 "
    "has the middle bar that 0 lacks. 1 is a single stroke on the right.\n"
    "- 5 or 9: a 9 closes its top into a loop at the upper right; a 5 has a flat top bar "
    "and a stroke down the left only, open at the upper right. Write 9 only when that "
    "upper loop is closed.\n"
    "- 5 or 6: a 6 closes its lower half on the left; a 5 is open at the lower left.\n"
    "\n"
    "DECIMALS (MECHANICAL)\n"
    "- Write WINDOWS first. A wheel at the right end is a decimal ONLY when its digit is "
    "smaller than the other wheels' digits, or it sits in a box of another colour than "
    "the wheels before it (red, pink, orange, yellow, or white where the others are dark; "
    "glare can make red look pale). Nothing else makes a decimal: not a frame of the same "
    "colour, not a dot or comma, not a printed 0.1 label, not being the last wheel. "
    "Otherwise DECIMALS: 0. Keep decimal wheels in READING.\n"
    "- Examples: black wheels 0 4 7 2 9 3 then a 6 in a red box: WINDOWS: dark dark dark "
    "dark dark dark red, READING: 0472936, DECIMALS: 1. A dark band 0 4 2 9 6 then a 7 "
    "in a white box: WINDOWS: dark dark dark dark dark white, READING: 042967, DECIMALS: "
    "1. Wheels 1 2 3 4 5 then a smaller 6, same colour: WINDOWS: dark dark dark dark dark "
    "dark-small, READING: 123456, DECIMALS: 1. Black wheels 3 8 1 2 5 6, all the same "
    "size and colour: READING: 381256, DECIMALS: 0.\n"
    "\n"
    "ROLLING (MECHANICAL)\n"
    "- A wheel showing parts of two numbers: write the lower in READING (9 when it turns "
    "from 9 to 0) and list it as its position in READING from the left, then the two "
    "numbers, e.g. 6:6-7.\n"
    "\n"
    "WHEN UNSURE\n"
    "- Too blurred, dim or washed out to read every cell: UNREADABLE. No meter in the "
    "photo: NO METER. Never guess a digit and never shorten the reading to fit a length "
    "you expect.")
if os.environ.get('FOTOMATES_PROMPT', 'compact') == 'compact':
    _SYSTEM_PROMPT, _USER_PROMPT = _COMPACT_SYSTEM_PROMPT, _COMPACT_USER_PROMPT


def _request_ocr(image_path: Path, choice: Backend = DEFAULT_BACKEND, scale: float = 1.0) -> OcrResult:
    steps = []
    aids = []
    server = []
    model_seconds = []  # the server's own time per model request, None when it does not say
    overall = time.monotonic()

    def timed(label, work):
        """Run one stage, remember what it cost — including when it fails."""
        started = time.monotonic()
        try:
            return work()
        finally:
            steps.append((label, time.monotonic() - started))

    def reading_seconds():
      
        if model_seconds and None not in model_seconds:
            model = sum(model_seconds)
        else:
            model = sum(seconds for label, seconds in steps if label.startswith('model request'))
        return model + sum(seconds for label, seconds in steps if label == 'extract reading')

    def breakdown(*tail):
        total = time.monotonic() - overall
        lines = ['', '--- computation ---']
        merged = {} 
        for label, seconds in steps:
            if label.startswith('model request') or label == 'extract reading':
                merged[label] = merged.get(label, 0.0) + seconds
        for label, seconds in merged.items():
            lines.append('%-22s %6.2f s   %4.1f%%' % (label, seconds, 100 * seconds / max(total, 1e-6)))
        lines.append('%-22s %6.2f s' % ('reading time', reading_seconds()))
        lines.extend(tail)
        return '\n'.join(lines)

    try:
        content, choice, report, readings = _read_photo(image_path, timed, aids, server, model_seconds, choice,
                                                        scale)
    except Exception as error:
        reason = _failure_reason(error)
        raise OcrFailure(reason, breakdown(*server, 'FAILED: ' + reason)) from error
    confidence, per_digit, margins, runner_ups, alternatives = _digit_confidence(choice)
    # No score here: it is made once, at the end, on the consumption (score_reading).
    tail = list(server)
    if confidence is not None:
        tail.append('confidence %.1f%% (weakest digit) | per digit: %s' % (
            100 * confidence, ' '.join('%.0f%%' % (100 * p) for p in per_digit)))
    sent = report.get('sent_image')
    return OcrResult(content + breakdown(*tail), evidence_source='+'.join(aids),
                     reading_seconds=reading_seconds(),
                     sent_png=base64.b64decode(sent.split(',', 1)[1]) if sent else None,
                     confidence=confidence, digit_confidence=per_digit,
                     digit_margin=margins, digit_runner_up=runner_ups, digit_alternatives=alternatives,
                     rolling=extract_rolling(content),
                     device_type=extract_device_type(content),
                     decimal_digits=extract_decimal_digits(content))


def _read_photo(image_path: Path, timed, aids: list, server: list, model_seconds: list,
                backend_choice: Backend = DEFAULT_BACKEND, scale: float = 1.0):
    # The photo is only decoded to judge it: whether it needs enhancing.
    report = dict(blur_variance=0.0)
    needs = []
    enlarged = scale != 1.0
    if _ENHANCE_MODE != 'never' or enlarged:
        # Read at its own size, unless a second read asks for it enlarged
        # (scale): a small photo's digits are a few pixels wide there.
        original = timed('decode image', lambda: decode_image(image_path))
        if enlarged:
            original = timed('enlarge photo', lambda: cv2.resize(original, None, fx=scale, fy=scale,
                                                                 interpolation=cv2.INTER_CUBIC))
        report = timed('assess quality', lambda: assess(original))
        if _ENHANCE_MODE == 'always':
            needs = ['always']
        elif _ENHANCE_MODE == 'auto':
            needs = [flag for flag in _ENHANCE_WHEN if report.get(flag)]
    if needs or enlarged:
        photo = timed('enhance photo', lambda: enhance(original, report)) if needs else original
        encoded = timed('encode png + base64', lambda: png_data_url(photo))
        if enlarged:
            aids.append(f'enlarged({scale:g}x)')
        if needs:
            aids.append('enhanced(' + ','.join(needs) + ')')
    else:
        encoded = timed('read photo file', lambda: photo_data_url(image_path))
        aids.append('original')
    # The photo as prepared is what the model reads, never the original beside it.
    report = dict(report, sent_image=encoded if needs or enlarged else None)

    images = [
        {"type": "text", "text": "Meter photo."},
        {"type": "image_url", "image_url": {"url": encoded}},
    ]

    # Straight to the model server, no slot claimed or waited for: the server
    # queues what it cannot run yet. A model on several servers takes them in turn.
    servers = backend_choice.servers
    model_server = servers[next(_TURN) % len(servers)]
    if len(servers) > 1:
        aids.append(f'server{servers.index(model_server) + 1}')
    read_started = time.monotonic()
    extra = dict(NO_THINKING)

    def read(system: str, temperature: float = 0, seed: int = 0, timeout: float | None = None):
        response = complete(
            server=model_server,
            **({'timeout': timeout} if timeout is not None else {}),
            model=backend_choice.model, temperature=temperature, seed=seed, max_tokens=_READ_MAX_TOKENS,
            logprobs=True,  # per-digit probabilities, for the score
            top_logprobs=5,  # and each digit's runner-ups: a close second digit is a confusion
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": _USER_PROMPT}, *images]},
            ],
            extra_body=extra,
        )

        timings = (response.model_extra or {}).get('timings') or {}
        usage = response.usage
        model_seconds.append((timings.get('prompt_ms', 0) + timings.get('predicted_ms', 0)) / 1000
                             if timings else None)
        if timings or usage:
            server.append('server: prompt %s tok in %.2f s, answer %s tok in %.2f s' % (
                getattr(usage, 'prompt_tokens', timings.get('prompt_n')),
                timings.get('prompt_ms', 0) / 1000,
                getattr(usage, 'completion_tokens', timings.get('predicted_n')),
                timings.get('predicted_ms', 0) / 1000))
        return response.choices[0]

    choice = timed('model request', lambda: read(_SYSTEM_PROMPT))
    content = choice.message.content or ''
    readings = [timed('extract reading', lambda: extract_reading(content))]
    return content, choice, report, readings


def warm_up() -> None:
    for choice in BACKENDS.values():
        for server in choice.servers:
            try:
                complete(server=server, model=choice.model,
                         max_tokens=1, messages=[{"role": "user", "content": "ok"}], extra_body=NO_THINKING)
                print(f'model warm: {choice.label} at {server.url}', flush=True)
            except Exception as error:
                print(f'warm-up skipped for {server.url}: {type(error).__name__}: {error}', flush=True)


def ocr_image(image_path: Path, model: str | None = None, scale: float = 1.0) -> OcrResult:

    return _request_ocr(Path(image_path), backend(model), scale)











