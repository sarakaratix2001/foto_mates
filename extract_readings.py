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
from detect_quality import assess, enhance  # after .env: it reads FOTOMATES_ENHANCER on import


def _required_env(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise RuntimeError(f'{name} is not set; FotoMates has no default model server (see .env)')
    return value


# Enhancement before reading. auto: only photos assess() flags with one of
# ENHANCE_WHEN are corrected, every other photo is sent as its original file.
# always: every photo is corrected. never: every photo is sent as its original file.
# whole: as auto for a whole photo, and a crop is never corrected (a crop of a bright LCD is
# often flagged, and the correction can cost a digit a stroke at that size).
_ENHANCE_MODE = os.environ.get('FOTOMATES_ENHANCE', 'auto').strip().lower()
if _ENHANCE_MODE not in ('auto', 'always', 'never', 'whole'):
    raise ValueError(f"FOTOMATES_ENHANCE must be 'auto', 'always', 'never' or 'whole', not {_ENHANCE_MODE!r}")
_ENHANCE_WHEN = tuple(flag.strip() for flag in os.environ.get(
    'FOTOMATES_ENHANCE_WHEN', 'is_low_contrast,is_dark,is_bright,has_glare').split(',') if flag.strip())
# Shorter side, in pixels, a photo is enlarged to before it is read; 0 sends it at its own size.
_MIN_SIDE = float(os.environ.get('FOTOMATES_MIN_SIDE', '480'))

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
            max_retries=int(os.environ.get('FOTOMATES_API_RETRIES', '2')),
            http_client=httpx.Client(timeout=None, transport=httpx.HTTPTransport(socket_options=_KEEPALIVE),
                                     limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)),
        )
        self.slots_url = re.sub(r'/v1/?$', '', url.rstrip('/')) + '/slots'


SERVERS = [_Server(url) for url in OCR_URLS]


# Photos read at once when the server reports its slots: at most this many, and at most
# WORKERS_PER_SLOT per slot. Production value.
IMAGE_WORKERS = int(os.environ.get('FOTOMATES_IMAGE_WORKERS', '24'))


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
        self.workers = workers or IMAGE_WORKERS


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
    """SDK retries transient transport/status errors; wait for model loading too."""
    loading_deadline = time.monotonic() + _MODEL_LOADING_WAIT
    while True:
        try:
            return (server or SERVERS[0]).client.chat.completions.create(**request)
        except Exception as error:
            if _is_model_loading(error) and time.monotonic() < loading_deadline:
                time.sleep(_MODEL_LOADING_POLL)
                continue
            raise


def _on(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default).strip().lower() not in ('0', 'false', 'no', 'off', '')


# Task 6b. FOTOMATES_STREAM=1 (default off): every model request is streamed, so the client
# measures the time to the first token (queueing + prompt evaluation) and the time from the
# first to the last token (generation); the per-digit logprobs arrive with the stream.
STREAM = _on('FOTOMATES_STREAM')
# Task 6c. FOTOMATES_CACHE_PROMPT=1 (default off): "cache_prompt": true is sent with every request.
CACHE_PROMPT = _on('FOTOMATES_CACHE_PROMPT')
# Task 6c. FOTOMATES_SLOT_IDS (default empty: the server chooses): the server slots the program's
# requests are pinned to, e.g. 0-7 or 0,1,2,3; each request takes the next one in turn (id_slot).
def _slot_ids(text: str) -> list:
    ids = []
    for part in filter(None, (p.strip() for p in text.split(','))):
        low, _, high = part.partition('-')
        ids += list(range(int(low), int(high or low) + 1))
    return ids


SLOT_IDS = _slot_ids(os.environ.get('FOTOMATES_SLOT_IDS', ''))
_SLOT_TURN = itertools.count()

# What every model request of the photo being read cost (Task 11, 6b): kept per thread, as a
# worker reads one photo at a time. start_photo_stats() before the photo, photo_stats() after.
_LOCAL = threading.local()


def start_photo_stats() -> None:
    _LOCAL.calls = []


def photo_stats() -> dict:
    """The photo's model requests in sum: model_calls, prompt_tokens, completion_tokens,
    cached_tokens (None when the server never said), first_token_ms and generation_ms (None
    unless streamed)."""
    calls = getattr(_LOCAL, 'calls', None) or []

    def total(key):
        values = [call[key] for call in calls if call.get(key) is not None]
        return round(sum(values)) if values else None
    return dict(model_calls=len(calls), prompt_tokens=total('prompt_tokens'),
                completion_tokens=total('completion_tokens'), cached_tokens=total('cached_tokens'),
                first_token_ms=total('first_token_ms'), generation_ms=total('generation_ms'))


def _record(call: dict) -> None:
    calls = getattr(_LOCAL, 'calls', None)
    if calls is not None:
        calls.append(call)


def _streamed(server, request: dict):
    """The request streamed, put back together as the non-streamed response looks
    (choices[0].message.content, choices[0].logprobs.content, usage, model_extra.timings),
    with the client's own first_token_ms and generation_ms."""
    from types import SimpleNamespace as NS
    sent = time.monotonic()
    stream = complete(server=server, stream=True, stream_options={'include_usage': True}, **request)
    text, tokens, usage, timings, finish = [], [], None, None, None
    first = last = None
    for chunk in stream:
        usage = getattr(chunk, 'usage', None) or usage
        extra = getattr(chunk, 'model_extra', None) or {}
        timings = extra.get('timings') or timings
        for part in getattr(chunk, 'choices', None) or []:
            delta = getattr(part, 'delta', None)
            piece = getattr(delta, 'content', None)
            if not piece and delta is not None:
                # The model server streams the answer's text as delta.reasoning (thinking is off
                # on every request, so nothing else is written there); content stays empty.
                other = getattr(delta, 'model_extra', None) or {}
                piece = (other.get('reasoning') or other.get('reasoning_content')
                         or getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None))
            logprobs = getattr(getattr(part, 'logprobs', None), 'content', None)
            if piece or logprobs:
                now = time.monotonic()
                first = first if first is not None else now
                last = now
            if piece:
                text.append(piece)
            if logprobs:
                tokens.extend(logprobs)
            finish = getattr(part, 'finish_reason', None) or finish
    choice = NS(message=NS(content=''.join(text)), logprobs=NS(content=tokens), finish_reason=finish)
    response = NS(choices=[choice], usage=usage, model_extra={'timings': timings} if timings else {})
    response.first_token_ms = None if first is None else (first - sent) * 1000
    response.generation_ms = None if first is None else (last - first) * 1000
    return response


def model_call(server=None, **request):
    """One model request (a read or the box request), with the Task 6b/6c settings applied,
    its cost recorded for photo_stats(). Same answer as complete()."""
    extra = dict(request.pop('extra_body', None) or {})
    if CACHE_PROMPT:
        extra['cache_prompt'] = True
    if SLOT_IDS:
        extra['id_slot'] = SLOT_IDS[next(_SLOT_TURN) % len(SLOT_IDS)]
    if extra:
        request['extra_body'] = extra
    response = _streamed(server, request) if STREAM else complete(server=server, **request)
    usage = getattr(response, 'usage', None)
    timings = (getattr(response, 'model_extra', None) or {}).get('timings') or {}
    details = getattr(usage, 'prompt_tokens_details', None)
    cached = getattr(details, 'cached_tokens', None) if details is not None else None
    prompt = getattr(usage, 'prompt_tokens', None)
    if cached is None and prompt is not None and timings.get('prompt_n') is not None:
        cached = max(0, prompt - timings['prompt_n'])  # llama.cpp: prompt_n counts only what it evaluated
    _record(dict(prompt_tokens=prompt, completion_tokens=getattr(usage, 'completion_tokens', None),
                 cached_tokens=cached, first_token_ms=getattr(response, 'first_token_ms', None),
                 generation_ms=getattr(response, 'generation_ms', None)))
    return response


_SLOTS_COUNT_TIMEOUT = float(os.environ.get('FOTOMATES_SLOTS_COUNT_TIMEOUT', '5'))
_TURN = itertools.count()  # which server of a model takes the next photo



# The slot count assumed when a server's /slots endpoint does not answer. Production value.
_SERVER_SLOTS = int(os.environ.get('FOTOMATES_SERVER_SLOTS', '32'))
WORKERS_PER_SLOT = int(os.environ.get('FOTOMATES_WORKERS_PER_SLOT', '2'))


def server_slots(servers: list | None = None) -> int | None:
    counts = []
    for server in servers or SERVERS:
        try:
            with urllib.request.urlopen(server.slots_url, timeout=_SLOTS_COUNT_TIMEOUT) as response:
                counts.append(len(json.load(response)))
        except Exception:
            counts.append(_SERVER_SLOTS)
    return sum(counts) or None


def image_workers(choice: Backend, slots: int | None = None) -> int:
    """How many photos to read at once with a model choice: its own worker count when it
    has one or no slot count is known, otherwise WORKERS_PER_SLOT per slot, at most IMAGE_WORKERS."""
    if choice.fixed_workers or not slots:
        return choice.workers
    return min(IMAGE_WORKERS, slots * WORKERS_PER_SLOT)


# The photo file types the model server reads as they are.
PHOTO_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}


class PhotoFileError(OSError):
    """The photo's file could not be read (the share gone, a file not there): raised
    before any model request, so the photo itself was never judged."""


@functools.lru_cache(maxsize=512)
def photo_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PhotoFileError(error.errno, f'{type(error).__name__}: {error}', str(path)) from error


def photo_data_url(path: Path) -> str:
    kind = PHOTO_TYPES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.suffix or 'no extension'}: the model server reads "
                         + ", ".join(PHOTO_TYPES) + " photos")
    return f"data:{kind};base64," + base64.b64encode(photo_bytes(path)).decode("ascii")


def png_data_url(pixels: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError("Could not encode the enhanced photo")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_image(path: Path) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(photo_bytes(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("Could not decode image")
    return pixels


_MAX_READING_DIGITS = int(os.environ.get('FOTOMATES_MAX_READING_DIGITS', '12'))
_MIN_READING_DIGITS = int(os.environ.get('FOTOMATES_MIN_READING_DIGITS', '6'))


def _digit_count(reading: str) -> int:
    return sum(ch.isdigit() for ch in reading)


_DOTTED_CODE = re.compile(r"^(?:\d{1,2}[.,]){2,}\d{1,2}(?=[.,]|$)[.,]?")
_GLUED_CODE = re.compile(r"^(?:1\d|[1-9])[5-8]\d(?=0\d{5})")
# Register codes a display shows before its reading (3.8.0, 5.8.0, 15.8.0, ...): an
# 11-digit reading that starts with one has the code glued on, as no meter shows 11
# digits of its own. 1588 and 1589 are 15.8.0 with its last 0 misread.
_ELEVEN_DIGIT_CODES = ('1588', '1580', '1589', '380', '580', '680', '390', '300')
# Up to 9 digits a reading can be a real one that only looks like code + reading
# (178 064558 is 178064.558 on a 9-digit display), so there only these codes are cut.
_SHORT_GLUED_CODES = ('1580', '158', '180', '380', '580', '680')


def _strip_code(digits: str, code: str) -> str:
    if len(digits) == 11:
        known = next((c for c in _ELEVEN_DIGIT_CODES if digits.startswith(c)), None)
        if known:
            return digits[len(known):]
    code_digits = re.sub(r"\D", "", code)
    if len(code_digits) >= 2 and digits.startswith(code_digits) \
            and len(digits) - len(code_digits) >= 4:
        return digits[len(code_digits):]
    glued = _GLUED_CODE.match(digits)
    if glued and (len(digits) >= 10 or digits[:glued.end()] in _SHORT_GLUED_CODES):
        return digits[glued.end():]
    return digits

def extract_device_type(text: str) -> str | None:
    # Prompt v4 answers TYPE: D or M.
    pattern = (r"TYPE:[ \t]*\**[ \t]*(DIGITAL|MECHANICAL|D\b|M\b)" if PROMPT == 'v4'
               else r"TYPE:[ \t]*\**[ \t]*(DIGITAL|MECHANICAL)")
    types = re.findall(pattern, text, re.IGNORECASE)
    if not types:
        return None
    return 'digital' if types[-1].upper().startswith('D') else 'mechanical'


def extract_off(text: str) -> bool:
    """READING: OFF (prompt v3/v4): a meter is in the photo and its display is dark or blank."""
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text or '', re.IGNORECASE)
    return bool(answers) and re.match(r"\s*OFF\b", answers[-1], re.IGNORECASE) is not None


SOURCES = ('LCD', 'LED', 'WHEELS', 'HANDWRITTEN', 'PRINTED', 'NONE')
# A SOURCE that is not a meter's register: such a read gives no reading (status no_meter).
NOT_A_REGISTER = ('HANDWRITTEN', 'PRINTED', 'NONE')


def extract_source(text: str) -> str | None:
    """The SOURCE line (prompt v3/v4): where the CELLS digits are physically shown; None without one."""
    answers = re.findall(r"^SOURCE:[ \t]*\**[ \t]*([A-Z]+)", text or '', re.IGNORECASE | re.MULTILINE)
    value = answers[-1].upper() if answers else None
    return value if value in SOURCES else None


def with_reading_line(text: str) -> str:
    """A v4 answer has no READING line: CELLS carries the digits or one of the words (NO METER,
    UNREADABLE, BLANK, OFF). A READING line made from CELLS is added, so every rule that reads
    READING works on v4 answers too. Any other answer is returned as it is."""
    if PROMPT != 'v4' or re.search(r'^READING:', text or '', re.IGNORECASE | re.MULTILINE):
        return text
    cells = re.findall(r'^CELLS:[ \t]*([^\n]*)', text or '', re.IGNORECASE | re.MULTILINE)
    if not cells:
        return text
    value = cells[-1].strip()
    digits = ''.join(value.split()) if re.fullmatch(r'\d(?:[ \t]*\d)*', value) else value
    return text.rstrip('\n') + '\nREADING: ' + digits


def extract_no_meter(text: str) -> bool:
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text or '', re.IGNORECASE)
    return bool(answers) and re.match(r"\s*NO[\s_-]*METER", answers[-1], re.IGNORECASE) is not None


def extract_blank(text: str) -> bool:
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text or '', re.IGNORECASE)
    return bool(answers) and re.match(r"\s*BLANK\b", answers[-1], re.IGNORECASE) is not None


def _cells_digits(text: str) -> str | None:
    """The digits of the last CELLS line when it is one digit per cell, else None."""
    cells = re.findall(r'^CELLS:[ \t]*([^\n]*)', text or '', re.IGNORECASE | re.MULTILINE)
    # v4 may write the cells without spaces (when the tokenizer keeps one token per digit).
    one_per_cell = r'[ \t]*\d(?:[ \t]*\d)*[ \t]*' if PROMPT == 'v4' else r'[ \t]*\d(?:[ \t]+\d)*[ \t]*'
    if cells and re.fullmatch(one_per_cell, cells[-1]):
        return ''.join(cells[-1].split())
    return None


def _reading_line(text: str) -> str | None:
    """READING's digits as written (dotted codes dropped, a glued register code still
    on); "" when it holds anything but digits, None when there is no READING line."""
    answers = re.findall(r"READING:[ \t]*\**[ \t]*([^\n*]+)", text or '', re.IGNORECASE)
    if not answers:
        return None
    reading = ""
    for token in answers[-1].split():
        token = _DOTTED_CODE.sub("", token).strip(".,")
        if not re.search(r"\d", token):  # a unit such as kWh, or UNREADABLE
            continue
        if not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", token):
            return ""
        reading += re.sub(r"[.,]", "", token)
    return reading


def _reading_digits(text: str) -> str:
    reading = _reading_line(text)
    if reading is not None:
        codes = re.findall(r"CODE:[ \t]*\**[ \t]*([^\n*]*)", text, re.IGNORECASE)
        if extract_device_type(text) != 'mechanical':
            # A digital display read cell by cell (CELLS, prompt v2): READING only copies
            # those digits, so where the two differ the cells are kept.
            cells = _cells_digits(text)
            if reading and cells and cells != reading:
                reading = cells
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
        cells = _cells_digits(text)
        if cells is not None and cells != reading:
            return ""  # a mechanical READING that is not its wheels is not trusted
    if _digit_count(reading) > _MAX_READING_DIGITS:
        return ""
    return reading
 
def extract_decimal_digits(text: str) -> int | None:
    """The DECIMALS count the model wrote: on a mechanical counter, and with prompt v3 (which
    asks it for every meter) on a digital display too."""
    if extract_device_type(text) != 'mechanical' and PROMPT not in ('v3', 'v4'):
        return None
    answers = re.findall(r'^DECIMALS:[ \t]*(\d+|UNKNOWN)[ \t]*$', text, re.IGNORECASE | re.MULTILINE)
    count = int(answers[-1]) if answers and answers[-1].isdigit() else None
    digits = _reading_digits(text)
    return count if count is not None and digits and 0 <= count < len(digits) else None

def extract_reading(text: str) -> str:
    reading = _reading_digits(text)
    stripped = reading.lstrip("0")
    return stripped or (reading and "0")


@dataclass
class OcrResult:
    text: str
    evidence_source: str = "unknown"
    confidence: float | None = None          # the weakest digit's probability, 0..1
    digit_confidence: list | None = None     # one probability per digit read
    digit_margin: list | None = None         # per digit: its probability minus the runner-up digit's
    digit_runner_up: list | None = None      # per digit: the most likely other digit, or None
    digit_alternatives: list | None = None   # per digit: {digit: probability} of the model's top alternatives
    device_type: str | None = None           # 'mechanical', 'digital', or None when not said
    decimal_digits: int | None = None        # visually marked fractional wheels; 0 means none
    clear: float | None = None               # the model's probability that the photo is CLEAR, 0..1
    code_confidence: list | None = None      # per digit of a register code cut from the front of READING
    image_report: dict | None = None         # the photo's size and quality checks (blur, contrast, glare, ...)
    reading_seconds: float | None = None     # the model reading the photo plus digit extraction
    sent_png: bytes | None = None            # the enhanced photo the model read; None: it read the original


# A black frame: the whole photo's grey mean and grey standard deviation (0..255) are both
# below this. It is answered BLANK without any model request and is never brightened; 0 turns
# the check off. In October no photo below 5 on both was ever read.
BLANK_MAX = float(os.environ.get('FOTOMATES_BLANK_MAX', '5'))
# Task 4, a second rule (default off: FOTOMATES_BLANK_P99=0). A near-black frame: grey mean and
# standard deviation both below FOTOMATES_BLANK_DIM (default 20) and the 99th percentile of grey
# below FOTOMATES_BLANK_P99 (proposed 40). A lit display gives bright pixels even in a dark
# photo, so its 99th percentile is high; a black frame, or one with a flash spot under 1% of
# the photo, has none. Either rule makes the photo blank.
BLANK_P99 = float(os.environ.get('FOTOMATES_BLANK_P99', '0'))
BLANK_DIM = float(os.environ.get('FOTOMATES_BLANK_DIM', '20'))


def _grey_p99(pixels: np.ndarray) -> float:
    grey = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
    return float(np.percentile(grey, 99))


def blank_photo(image_path: Path) -> OcrResult | None:
    """The result of a black frame, made without a model request (READING: BLANK,
    photo_sent 'blank-check'); None when the photo is not one or both rules are off.
    It runs before any model request, so a photo the model was asked about is never made blank."""
    if BLANK_MAX <= 0 and BLANK_P99 <= 0:
        return None
    pixels = decode_image(Path(image_path))
    report = assess(pixels)
    black = BLANK_MAX > 0 and report['brightness'] < BLANK_MAX and report['contrast'] < BLANK_MAX
    p99 = None
    if not black and BLANK_P99 > 0 and report['brightness'] < BLANK_DIM and report['contrast'] < BLANK_DIM:
        p99 = _grey_p99(pixels)
        black = p99 < BLANK_P99
    if not black:
        return None
    if p99 is None:
        note = (f"black frame: brightness {report['brightness']:.1f}, contrast {report['contrast']:.1f}, "
                f"both below {BLANK_MAX:g}; no model request")
    else:
        note = (f"near-black frame: brightness {report['brightness']:.1f}, contrast {report['contrast']:.1f}, "
                f"both below {BLANK_DIM:g}, 99th percentile {p99:.0f} below {BLANK_P99:g}; no model request")
    return OcrResult('TYPE: BLANK\nREADING: BLANK\n--- ' + note + ' ---', evidence_source='blank-check',
                     reading_seconds=0.0,
                     image_report=dict(report, width=int(pixels.shape[1]), height=int(pixels.shape[0])))


RETAKE_SCORE = float(os.environ.get('FOTOMATES_RETAKE_SCORE', '0.5'))
# A reading with this many digits or more gets score 0: no meter shows 12 digits.
TOO_MANY_DIGITS = int(os.environ.get('FOTOMATES_TOO_MANY_DIGITS', '12'))
# QUALITY: CLEAR/POOR is only recorded (its chance of CLEAR is saved with each photo);
# it never rejects a photo unless this is set above 0: retake below this chance of CLEAR.
RETAKE_CLEAR = float(os.environ.get('FOTOMATES_RETAKE_CLEAR', '0'))


def test_pattern(cells: str) -> bool:
    """A digital display's segment test: every cell the same, 0 or 8 (00000000, 88888888)."""
    return len(cells) >= 4 and len(set(cells)) == 1 and cells[0] in '08'


def photo_retake_reason(cells: str, clear: float | None) -> str | None:
    """Why the photo itself, whatever its digits' confidence, must be retaken; None when it need not."""
    if test_pattern(cells):
        return 'the display shows its test pattern, not the reading'
    if clear is not None and clear < RETAKE_CLEAR:
        return f'the photo is not clear enough to check the digits ({100 * clear:.0f}% clear)'
    return None
QUALITY_SCORE = float(os.environ.get('FOTOMATES_QUALITY_SCORE', '0.9'))
QUALITY_DIGIT = float(os.environ.get('FOTOMATES_QUALITY_DIGIT', '0.95'))


def _digit_reason(parts: dict) -> list[str]:
    """The least certain digit, how sure the model was of it and what it could also be."""
    if parts and parts.get('too_long'):
        return [f"{parts['digit_count']} digits were read, and no meter shows that many: part of the "
                "display that is not the reading (a register code or a second line) was read with it"]
    if not parts or parts.get('weakest') is None:
        return ['no digit probabilities from the model']
    position, value, rival = parts['weakest_position'], parts.get('weakest_digit'), parts.get('weakest_runner_up')
    if position:
        suffix = 'th' if 10 <= position % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(position % 10, 'th')
        which = f"the {position}{suffix} digit" + (f" ({value})" if value is not None else '')
    else:
        which = 'a leading zero'
    could_be = f" (it could be a {rival})" if rival is not None else ''
    return [f"{which} is only {100 * parts['weakest']:.0f}% sure{could_be}: check it on the meter"]


def low_score_reasons(parts: dict) -> list[str]:
    """Why a score is low: its least certain digit."""
    return _digit_reason(parts)


def quality_reasons(parts: dict) -> list[str]:
    """What kept a score below 1: its least certain digit."""
    return _digit_reason(parts)


def score_reading(cells: str, per_digit: list | None, decimals: int | None,
                  margins: list | None = None, runner_ups: list | None = None,
                  alternatives: list | None = None, mechanical: bool = False) -> dict:

    """The probability that every digit read from the photo is right: the product of
    each digit's probability, leading zeros and decimals included. It is made from the
    photo alone: the saved reading in the file name plays no part, so it can be
    compared with the reading afterwards. decimals is what the model saw on the meter
    (None when it did not say), never a count derived from the saved reading. The one
    alternative that counts for the digit written is, on the last whole wheel of a
    counter, a neighbouring number it is turning between. The model's digit
    probabilities already carry what the photo lets it see: blur, glare or a dim
    display make the digits it reads uncertain."""
    cells, per_digit = cells or '', per_digit or []
    lead = len(cells) - len(cells.lstrip('0'))  # only to name a weak digit in messages
    end = len(cells) - (decimals or 0)
    aligned = len(per_digit) == len(cells)
    scored = []  # (probability right, index, most likely wrong digit)
    if aligned and alternatives and len(alternatives) == len(cells):
        for index, written in enumerate(cells):
            # The last whole wheel of a counter, known only when the model saw its decimals.
            turning = mechanical and decimals is not None and index == end - 1
            options = alternatives[index] or {}
            same = {digit for digit in options if digit != written and turning
                    and (int(digit) - int(written)) % 10 in (1, 9)}
            changes = {digit: p for digit, p in options.items() if digit != written and digit not in same}
            rival = max(changes, key=changes.get, default=None)
            scored.append((min(1.0, per_digit[index] + sum(options[digit] for digit in same)), index, rival))
    elif aligned:
        rivals = runner_ups if runner_ups and len(runner_ups) == len(per_digit) else [None] * len(per_digit)
        for index in range(len(cells)):
            scored.append((per_digit[index], index, rivals[index]))
    weakest = position = digit = rival = None
    total = None
    if scored:
        weakest, index, rival = min(scored)
        # Counted in the reading as shown, without its leading zeros; 0 for a leading zero.
        position, digit = (index + 1 - lead if index >= lead else 0), cells[index]
        total = math.prod(p for p, _, _ in scored)
    elif per_digit:
        weakest = total = min(per_digit)
    total = 0.0 if not cells else total
    # No meter shows 12 digits: such a reading includes something that is not the reading.
    too_long = len(cells) >= TOO_MANY_DIGITS
    if too_long:
        total = 0.0
    return dict(total=None if total is None else round(max(total, 0.0), 3),
                weakest=None if weakest is None else round(max(weakest, 0.0), 3),
                weakest_position=position, weakest_digit=digit, weakest_runner_up=rival,
                too_long=too_long, digit_count=len(cells))


def _digit_confidence(choice) -> tuple[float | None, list, list, list, list]:
    """Each digit's probability where the model first read it off the photo: the
    CELLS line, on every meter (prompt v2; compact writes cells for mechanical
    counters only). READING repeats those digits, so there the model is copying and
    is always near 100% sure. An answer without a CELLS row of one digit per cell
    (CELLS: NONE) is first read at READING."""
    tokens = getattr(getattr(choice, 'logprobs', None), 'content', None) or []
    content = getattr(getattr(choice, 'message', None), 'content', None) or ''
    written = _cells_digits(content)
    cells = _line_digits(tokens, 'CELLS:')
    probabilities, margins, runner_ups, alternatives = (
        cells if written and len(cells[0]) == len(written) else _line_digits(tokens, 'READING:'))
    return (min(probabilities) if probabilities else None), probabilities, margins, runner_ups, alternatives


def _clear_probability(choice) -> float | None:
    """How clear the model judged the photo, 0..1, from the first word it wrote on the
    QUALITY line and its alternatives: for CLEAR or POOR (compact), the probability of
    CLEAR; for a number 0-10 (v2), its expected value over the alternatives, divided
    by 10 (a 1 followed by a 0 is 10). None when it wrote no QUALITY line."""
    tokens = getattr(getattr(choice, 'logprobs', None), 'content', None) or []
    joined = ''.join(token.token or '' for token in tokens)
    marker = joined.upper().rfind('QUALITY:')
    if marker < 0:
        return None
    begin, offset = marker + len('QUALITY:'), 0
    for index, token in enumerate(tokens):
        piece = token.token or ''
        end = offset + len(piece)
        word = piece[max(0, begin - offset):].strip().upper()
        if end > begin and word and token.logprob is not None:
            options = {word: math.exp(token.logprob)}
            for alt in getattr(token, 'top_logprobs', None) or []:
                key = alt.token.strip().upper()
                if alt.logprob is not None and key:
                    options[key] = max(options.get(key, 0.0), math.exp(alt.logprob))
            if word[:1].isdigit():
                return _expected_quality(word, options, tokens[index + 1] if index + 1 < len(tokens) else None)
            clear = sum(p for key, p in options.items() if key[:1] == 'C')
            poor = sum(p for key, p in options.items() if key[:1] == 'P')
            return clear / (clear + poor) if clear + poor else None
        offset = end
    return None


def _expected_quality(word, options, following):
    """The expected QUALITY 0-10, divided by 10, from the first token's alternatives.
    Digits are often one token each, so a written 1 is 10 with the chance that the
    next token is 0."""
    ten = 0.0
    if word == '1' and following is not None and following.logprob is not None:
        nexts = {(following.token or '').strip(): math.exp(following.logprob)}
        for alt in getattr(following, 'top_logprobs', None) or []:
            if alt.logprob is not None:
                key = alt.token.strip()
                nexts[key] = max(nexts.get(key, 0.0), math.exp(alt.logprob))
        ten = nexts.get('0', 0.0) / (sum(nexts.values()) or 1.0)
    total = weight = 0.0
    for key, p in options.items():
        if not key.isdigit():
            continue
        value = 1 + 9 * ten if key == '1' else min(int(key), 10)
        total, weight = total + p * value, weight + p
    return total / weight / 10 if weight else None


def _line_digits(tokens, label: str) -> tuple[list, list, list, list]:
    """The probability, margin, runner-up and alternatives of each digit written
    after label, up to the end of the number or of its line."""
    joined = ''.join(token.token or '' for token in tokens)
    marker = joined.upper().rfind(label)
    if marker < 0:
        return [], [], [], []
    begin, offset = marker + len(label), 0
    probabilities, margins, runner_ups, alternatives = [], [], [], []
    for token in tokens:
        piece, logprob = token.token or '', token.logprob
        end = offset + len(piece)
        if end > begin and logprob is not None:
            part = piece[max(0, begin - offset):]  # the token's text after label
            written = part.split('\n')[0]  # and on its line
            digits = sum(ch.isdigit() for ch in written)
            if digits:
                probability = math.exp(logprob)
                options = {}
                for alt in getattr(token, 'top_logprobs', None) or []:
                    if alt.logprob is not None and alt.token.strip().isdigit() and len(alt.token.strip()) == 1:
                        options[alt.token.strip()] = max(options.get(alt.token.strip(), 0.0), math.exp(alt.logprob))
                others = [(p, d) for d, p in options.items() if d != written.strip()]
                second, rival = max(others, default=(0.0, None))
                probabilities.extend([probability] * digits)
                margins.extend([max(0.0, probability - second)] * digits)
                runner_ups.extend([rival] * digits)
                alternatives.extend([options] * digits)
            elif probabilities and written.strip() and written.strip() not in '.,':
                break  # the number is over
            if '\n' in part:
                break  # the line is over
        offset = end
    return probabilities, margins, runner_ups, alternatives


class OcrFailure(Exception):
    """A read that failed, carrying the per-stage timings up to the failure."""

    def __init__(self, message: str, breakdown: str):
        super().__init__(message)
        self.breakdown = breakdown


def _failure_reason(error: Exception) -> str:
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        message = str(current)
        key = globals().get('OCR_API_KEY', '')
        if key and key != 'EMPTY':
            message = message.replace(key, '[redacted]')
        message = re.sub(r'(https?://)[^/\s@]+@', r'\1[redacted]@', message)
        chain.append(f'{type(current).__name__}: {message[:300]}')
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return ' <- '.join(chain)


_COMPACT_SYSTEM_PROMPT = (
    "You read electricity meter registers from photos. Answer with exactly these seven "
    "lines and nothing else:\n"
    "TYPE: DIGITAL (LCD or LED segments) or MECHANICAL (printed number wheels)\n"
    "CELLS: MECHANICAL: one digit per wheel, left to right, space separated; DIGITAL: NONE\n"
    "WINDOWS: MECHANICAL: one word per wheel for its window colour (dark, white, red, "
    "orange, yellow, other), with -small when its digit is smaller than the other "
    "wheels' digits; DIGITAL: NONE\n"
    "READING: every digit of the main row as one string with no dot or comma, or "
    "UNREADABLE, or NO METER, or BLANK\n"
    "ROLLING: MECHANICAL wheels caught between two numbers as position:lower-upper, "
    "comma separated; otherwise NONE\n"
    "DECIMALS: MECHANICAL: number of wheels at the right end with a smaller digit or "
    "a box of another colour, 0 if none; DIGITAL: UNKNOWN\n"
    "QUALITY: CLEAR or POOR")
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
    "- Each digit is made of up to 7 strokes: top, upper-left, upper-right, middle, "
    "lower-left, lower-right, bottom. Printed wheel digits follow the same layout, curved. "
    "On an LCD, faint unlit segments do not count.\n"
    "  0: all but middle | 1: upper-right, lower-right only | 2: top, upper-right, middle, "
    "lower-left, bottom | 3: top, upper-right, middle, lower-right, bottom | 4: upper-left, "
    "upper-right, middle, lower-right | 5: top, upper-left, middle, lower-right, bottom | "
    "6: top, upper-left, middle, lower-left, lower-right, bottom | 7: top, upper-right, "
    "lower-right (some displays add upper-left) | 8: all seven | 9: all but lower-left\n"
    "- Compare each digit with the other digits of the same row: they share one font, so "
    "a stroke that is lit or printed in one digit looks the same in the next.\n"
    "- The most frequent mistakes are 7 read as 1 and 6 read as 5. Avoid them:\n"
    "  1 or 7: a top bar across the top of the cell makes it 7, however short or faint. "
    "A 1 has no top bar at all. Write 1 only when the top of the cell is empty.\n"
    "  5 or 6: a stroke on the lower left, closing the lower loop, makes it 6. Write 5 "
    "only when the lower left is clearly empty.\n"
    "- 8 or 6: 8 closes its upper loop on the right; 6 is open at the upper right. "
    "8 or 9: 9 is open at the lower left. 8 or 0: 0 has no middle stroke. 8 or 3: 3 has "
    "no left strokes.\n"
    "- 3 or 5: 3 has the upper-right stroke and no upper-left; 5 has the upper-left and "
    "no upper-right. 3 or 9: 9 has the upper-left stroke; 3 does not.\n"
    "- 2 or 5: mirror images. 2 goes down the upper right and the lower left; 5 goes down "
    "the upper left and the lower right. 2 or 7: 2 has a middle stroke and a bottom bar. "
    "2 or 4: 4 has an upper-left stroke and no top or bottom bar. 1 or 4: 4 has an "
    "upper-left stroke and a middle bar.\n"
    "- A faint but present stroke still counts; glare or a reflection never adds one.\n"
    "- Before replying, look again at every 1, 5, 3 and 6 you wrote: is there a top bar "
    "(then 7), a lower-left stroke (then 6), an upper-right stroke that closes the top "
    "(then 9 or 8)?\n"
    "\n"
    "DECIMALS (MECHANICAL)\n"
    "- Write WINDOWS first. A wheel at the right end is a decimal ONLY when its digit is "
    "smaller than the other wheels' digits, or it sits in a box of another colour than "
    "the wheels before it (red, pink, orange, yellow, or white where the others are dark; "
    "glare can make red look pale). Nothing else makes a decimal: not a frame of the same "
    "colour, not a dot or comma, not a printed 0.1 label, not being the last wheel. "
    "Otherwise DECIMALS: 0. Keep decimal wheels in READING.\n"
    "- Examples (window patterns only, never copy digits from them): six black wheels "
    "then one wheel in a red box: WINDOWS: dark dark dark dark dark dark red, DECIMALS: 1. "
    "Five wheels on a dark band then one in a white box: WINDOWS: dark dark dark dark "
    "dark white, DECIMALS: 1. Five wheels then a smaller digit, same colour: WINDOWS: "
    "dark dark dark dark dark dark-small, DECIMALS: 1. Six wheels all the same size and "
    "colour: DECIMALS: 0.\n"
    "\n"
    "ROLLING (MECHANICAL)\n"
    "- A wheel showing parts of two numbers: write the lower in READING (9 when it turns "
    "from 9 to 0) and list it as its position in READING from the left, then the two "
    "numbers, e.g. 6:6-7.\n"
    "\n"
    "NO METER OR BLANK PHOTO\n"
    "- BLANK: the photo shows nothing recognisable: all or nearly all black, white, grey "
    "or one colour, the lens covered, or so blurred that no object can be made out.\n"
    "- NO METER: the photo shows something, but no meter register: no meter LCD or LED "
    "display and no counter window of number wheels. Numbers alone are not a meter: "
    "numbers on a sticker, label, paper, document, nameplate, serial plate, barcode, "
    "phone screen, wall, door, box or cable are NO METER, even when they look like a "
    "reading. Never write digits that are not on a meter's register.\n"
    "- A meter is visible but its display is off, dark or cannot be read: UNREADABLE, "
    "not NO METER and not BLANK.\n"
    "\n"
    "WHEN UNSURE\n"
    "- Too blurred, dim or washed out to read every cell: UNREADABLE. Never guess a digit "
    "and never shorten the reading to fit a length you expect.\n"
    "\n"
    "QUALITY (always, last)\n"
    "- After reading, judge the photo: could a person check every digit you wrote from "
    "this photo alone, without guessing?\n"
    "- CLEAR: every digit of the row is sharp and fully visible.\n"
    "- POOR: any digit is blurred, faint, washed out, under glare or a reflection, cut off, "
    "too small to make out its strokes, or seen through a scratched or misted cover; the "
    "photo is of a screen or of another photo; or the display shows a test pattern (all "
    "segments lit, or every cell the same) instead of the reading. When in doubt: POOR.")
# v2: five lines (TYPE, CELLS, READING, DECIMALS, QUALITY 0-10): CELLS for every meter,
# no WINDOWS or ROLLING, and the DIGIT SHAPES cut to the pairs that are confused.
_V2_SYSTEM_PROMPT = (
    "You read electricity meter registers from photos. Answer with exactly these five "
    "lines and nothing else:\n"
    "TYPE: DIGITAL or MECHANICAL\n"
    "CELLS: one digit per cell of the main row, left to right, space separated; NONE when "
    "there is no row to read\n"
    "READING: the CELLS digits as one string with no space, dot or comma, or UNREADABLE, "
    "or NO METER, or BLANK\n"
    "DECIMALS: on a mechanical counter, the number of wheels at the right end with a "
    "smaller digit or a box of another colour, 0 if none; on a digital display, UNKNOWN\n"
    "QUALITY: <0-10>: 10 when every digit could be checked by a person from this photo, "
    "0 when none")
_V2_USER_PROMPT = (
    "TASK\n"
    "Read the consumption register: the main row of the largest digits on the display.\n"
    "\n"
    "WHAT TO READ\n"
    "- Every digit cell of that row, first to last, one digit per cell, in CELLS: leading "
    "zeros, and the last cell or two even when boxed, coloured, smaller, faint or under "
    "glare. Never skip or repeat a cell: CELLS has as many digits as the row has cells, "
    "and READING copies them.\n"
    "- On a mechanical counter: only the printed digits on the wheels inside the counter "
    "window, including a partly visible or turning last wheel. A wheel caught between two "
    "numbers: write the lower one (9 when it turns from 9 to 0).\n"
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
    "- A digit has up to 7 strokes: top, upper-left, upper-right, middle, lower-left, "
    "lower-right, bottom. On an LCD, faint unlit segments do not count; a faint but "
    "present stroke still counts, and glare or a reflection never adds one.\n"
    "- 5 or 6: a lower-left stroke closing the lower loop makes it 6. Write 5 only when "
    "the lower left is clearly empty.\n"
    "- 1 or 7: a top bar across the top of the cell makes it 7, however short or faint. "
    "Write 1 only when the top of the cell is empty.\n"
    "- 0 or 8: 8 has a middle stroke; 0 has none.\n"
    "- 6 or 8: 8 closes its upper loop on the right; 6 is open at the upper right.\n"
    "- 2 or 7: 2 has a middle stroke and a bottom bar; 7 has neither.\n"
    "- 5 or 3: 3 has the upper-right stroke and no upper-left; 5 has the upper-left and "
    "no upper-right.\n"
    "- 5 or 9: 9 has the upper-right stroke, closing its upper loop; 5 is open at the "
    "upper right.\n"
    "- Printed wheels: a 1 has a flag and a base; a 7 has a straight top bar.\n"
    "- Printed wheels: 6 and 9 are closed loops; 5 and 3 are open.\n"
    "\n"
    "DECIMALS (mechanical counters)\n"
    "- A wheel at the right end is a decimal only when its digit is smaller than the "
    "other wheels' digits, or it sits in a box of another colour than the wheels before "
    "it (red, pink, orange, yellow, or white where the others are dark; glare can make "
    "red look pale). Nothing else makes a decimal: not a frame of the same colour, not a "
    "dot or comma, not a printed 0.1 label, not being the last wheel. Otherwise "
    "DECIMALS: 0. Keep decimal wheels in CELLS and READING.\n"
    "- Examples: six dark wheels then one wheel in a red box: DECIMALS: 1. Five wheels "
    "then a smaller digit of the same colour: DECIMALS: 1. Six wheels all the same size "
    "and colour: DECIMALS: 0.\n"
    "\n"
    "PHOTOS WITHOUT A METER\n"
    "- BLANK: the photo shows nothing recognisable: all or nearly all black, white, grey "
    "or one colour, the lens covered, or so blurred that no object can be made out.\n"
    "- NO METER: the photo shows something, but no meter register: no meter LCD or LED "
    "display and no counter window of number wheels. Numbers alone are not a meter: "
    "numbers on a sticker, label, paper, document, nameplate, serial plate, barcode, "
    "phone screen, wall, door, box or cable mean no meter, even when they look like a "
    "reading. Never write digits that are not on a meter's register.\n"
    "- A meter is visible but its display is off, dark or cannot be read: UNREADABLE.\n"
    "\n"
    "WHEN UNSURE\n"
    "- Too blurred, dim or washed out to read every cell: UNREADABLE. Never guess a digit "
    "and never shorten the reading to fit a length you expect.\n"
    "\n"
    "QUALITY (always, last)\n"
    "- After reading, judge the photo: how many of the digits you wrote could a person "
    "check from this photo alone, without guessing? 10: every digit is sharp and fully "
    "visible. Lower it for each digit that is blurred, faint, washed out, under glare or a "
    "reflection, cut off, too small to make out its strokes, or seen through a scratched "
    "or misted cover. 0: none can be checked, the photo is of a screen or of another "
    "photo, or the display shows a test pattern (all segments lit, or every cell the "
    "same) instead of the reading.")



def _swap(text: str, old: str, new: str) -> str:
    """text with old replaced by new; an error when old is not there exactly once."""
    if text.count(old) != 1:
        raise ValueError(f'prompt text not found once: {old[:50]!r}')
    return text.replace(old, new)


# v3: v2 with DECIMALS asked for every meter. On a digital display the model counts the
# smaller cell(s) at the right end, the cell it otherwise drops (0009261.9 read as 000926).
_V3_SYSTEM_PROMPT = _swap(
    _V2_SYSTEM_PROMPT,
    "DECIMALS: on a mechanical counter, the number of wheels at the right end with a "
    "smaller digit or a box of another colour, 0 if none; on a digital display, UNKNOWN\n",
    "DECIMALS: the number of cells at the right end of the row that are smaller than the "
    "others, in a box of another colour, or after a dot or comma; 0 if none (on a mechanical "
    "counter: the wheels with a smaller digit or a box of another colour)\n")
_V3_USER_PROMPT = _swap(_swap(
    _V2_USER_PROMPT, "DECIMALS (mechanical counters)\n", "DECIMALS\n"),
    "and colour: DECIMALS: 0.\n",
    "and colour: DECIMALS: 0.\n"
    "- On a digital display: the cells at the right end of the row that are smaller than the "
    "others or come after a dot or comma. Example: a digital display of seven large cells "
    "then one smaller cell: DECIMALS: 1; keep that cell in CELLS and READING.\n")
# v3, Task 1: a sixth line, SOURCE: where the CELLS digits are physically shown (a number
# written in marker on a casing was read at 99-100% although WHAT TO IGNORE names it).
# v3, Task 2: OFF beside UNREADABLE, for a meter whose display is dark (no power).
_SOURCE_LINE = ("SOURCE: LCD, LED, WHEELS, HANDWRITTEN, PRINTED or NONE: where the CELLS digits are "
                "physically shown")
_V3_SYSTEM_PROMPT = _swap(_swap(
    _V3_SYSTEM_PROMPT, "exactly these five lines", "exactly these six lines"),
    "or UNREADABLE, or NO METER, or BLANK\n", "or UNREADABLE, or NO METER, or BLANK, or OFF\n") + "\n" + _SOURCE_LINE
_V3_USER_PROMPT = _swap(_swap(
    _V3_USER_PROMPT,
    "- A meter is visible but its display is off, dark or cannot be read: UNREADABLE.\n",
    "- OFF: a meter is in the photo and its display is dark or blank (switched off, no power): "
    "READING: OFF.\n"
    "- UNREADABLE: a meter's display is lit but cannot be read.\n"),
    "QUALITY (always, last)\n", "QUALITY (always)\n") + (
    "\n\nSOURCE (always, last)\n"
    "- Where the CELLS digits are physically shown: LCD or LED (a meter's own display), WHEELS (a "
    "counter's number wheels), HANDWRITTEN (by hand or marker, on a casing, cover or label), PRINTED "
    "(a sticker, label, nameplate, paper or screen) or NONE (no digits read).\n"
    "- Example: a number written in marker on the side of a meter casing, no display in the photo: "
    "SOURCE: HANDWRITTEN, READING: NO METER.")

# v4 (Task 6a): v3 with a shorter answer, as the call time follows the answer's length (about
# 0.16 s per token). No READING line: CELLS carries the digits or one of the words; TYPE is D or M.
# FOTOMATES_V4_CELLS_SPACES=0 writes CELLS without spaces: only when the tokenizer still gives
# one token per digit (the per-digit probabilities need it); the default keeps the spaces.
V4_CELLS_SPACES = _on('FOTOMATES_V4_CELLS_SPACES', '1')
_V4_SYSTEM_PROMPT = _swap(_swap(_swap(_swap(
    _V3_SYSTEM_PROMPT, "exactly these six lines", "exactly these five lines"),
    "TYPE: DIGITAL or MECHANICAL\n", "TYPE: D (digital) or M (mechanical)\n"),
    "CELLS: one digit per cell of the main row, left to right, space separated; NONE when "
    "there is no row to read\n",
    "CELLS: one digit per cell of the main row, left to right, "
    + ("space separated" if V4_CELLS_SPACES else "with no space between them")
    + "; or UNREADABLE, or NO METER, or BLANK, or OFF\n"),
    "READING: the CELLS digits as one string with no space, dot or comma, or UNREADABLE, "
    "or NO METER, or BLANK, or OFF\n", "")
_V4_USER_PROMPT = _swap(_swap(_swap(_swap(_swap(
    _V3_USER_PROMPT,
    "as the row has cells, and READING copies them.\n", "as the row has cells.\n"),
    "Keep decimal wheels in CELLS and READING.\n", "Keep decimal wheels in CELLS.\n"),
    "keep that cell in CELLS and READING.\n", "keep that cell in CELLS.\n"),
    "READING: OFF.\n", "CELLS: OFF.\n"),
    "SOURCE: HANDWRITTEN, READING: NO METER.", "SOURCE: HANDWRITTEN, CELLS: NO METER.")

# Which prompt reads the photos: v2 (the default since the 4 Nov test: as accurate as compact, ~30% faster), compact,
# v3 (v2 with DECIMALS on digital displays too, SOURCE and OFF) or v4 (v3 with a shorter answer). v3 and v4 are not
# the default until their test passes. Anything else is an error, so a typo cannot select another prompt silently.
PROMPTS = {'compact': (_COMPACT_SYSTEM_PROMPT, _COMPACT_USER_PROMPT),
           'v2': (_V2_SYSTEM_PROMPT, _V2_USER_PROMPT),
           'v3': (_V3_SYSTEM_PROMPT, _V3_USER_PROMPT),
           'v4': (_V4_SYSTEM_PROMPT, _V4_USER_PROMPT)}
PROMPT = os.environ.get('FOTOMATES_PROMPT', 'v2').strip().lower()
if PROMPT not in PROMPTS:
    raise ValueError(f"FOTOMATES_PROMPT must be 'compact', 'v2', 'v3' or 'v4', not {PROMPT!r}")
_SYSTEM_PROMPT, _USER_PROMPT = PROMPTS[PROMPT]


# FOTOMATES_REPEAT_HINT=1 (default off): one more line under WHAT TO READ, against a digit read
# one cell short: a lone 1 (0717182 read 077182) as often as a run (a saved 111 read as 11).
REPEAT_HINT = _on('FOTOMATES_REPEAT_HINT')
_REPEAT_LINE = ("- Count the cells before reading. A 1 fills a whole cell even though it is a narrow bar "
                "with a wide gap on each side; adjacent identical digits are separate cells: write every one.\n")
if REPEAT_HINT:
    _USER_PROMPT = _swap(_USER_PROMPT, "\nWHAT TO IGNORE\n", _REPEAT_LINE + "\nWHAT TO IGNORE\n")

# FOTOMATES_FEWSHOT=1 (default off): example photos with their CELLS line, sent in the user
# message before the photo to read. They are the rows of fewshot/examples.csv (file, cells):
# small crops of real displays, a few hundred prompt tokens in all.
FEWSHOT = _on('FOTOMATES_FEWSHOT')
FEWSHOT_DIR = Path(os.environ.get('FOTOMATES_FEWSHOT_DIR', '') or Path(__file__).resolve().parent / 'fewshot')


def load_fewshots(folder: Path = FEWSHOT_DIR) -> list:
    """The example photos as message parts: for each row of examples.csv a caption, the
    image and its CELLS line. An error when the list or a file is missing or a row is bad:
    a run asked to use examples must not go without them silently."""
    import csv
    parts = []
    with open(Path(folder) / 'examples.csv', encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.DictReader(file) if (row.get('file') or '').strip()]
    for number, row in enumerate(rows, 1):
        cells = re.sub(r'\D', '', row.get('cells') or '')
        if not cells:
            raise ValueError(f"fewshot/examples.csv: no digits in cells for {row['file']}")
        data = (Path(folder) / row['file'].strip()).read_bytes()
        kind = 'jpeg' if row['file'].lower().endswith(('.jpg', '.jpeg')) else 'png'
        parts += [{'type': 'text', 'text': f'Example {number} (another meter, already read):'},
                  {'type': 'image_url', 'image_url': {
                      'url': f'data:image/{kind};base64,' + base64.b64encode(data).decode('ascii')}},
                  {'type': 'text', 'text': 'CELLS: ' + ' '.join(cells)}]
    if not parts:
        raise ValueError('fewshot/examples.csv has no examples')
    return parts


_FEWSHOTS = load_fewshots() if FEWSHOT else []
# Task 10. With FOTOMATES_FEWSHOT=1, which reads get the example photos (about 230 prompt tokens
# each): third (the default: only the read after an unsure whole photo) or all (every read).
FEWSHOT_ON = os.environ.get('FOTOMATES_FEWSHOT_ON', 'third').strip().lower()
if FEWSHOT_ON not in ('third', 'all'):
    raise ValueError(f"FOTOMATES_FEWSHOT_ON must be 'third' or 'all', not {FEWSHOT_ON!r}")


class _Flag(threading.local):
    third = False


_READ_KIND = _Flag()


class third_read:
    """`with third_read():` around the read after an unsure whole photo (FOTOMATES_FEWSHOT_ON=third)."""

    def __enter__(self):
        self.before, _READ_KIND.third = _READ_KIND.third, True

    def __exit__(self, *exc):
        _READ_KIND.third = self.before


def fewshots_for_this_read() -> list:
    return _FEWSHOTS if FEWSHOT_ON == 'all' or _READ_KIND.third else []


# Task 6c. FOTOMATES_PREFIX=1 (default off): every fixed word goes before the image, so the
# server can keep it as a cached prefix: all instructions in the system message, the user message
# holds the image only (and per-photo lines such as the cell count after it). The box request
# starts with the same system message, then the image, then its question.
PREFIX = _on('FOTOMATES_PREFIX')

# Crop first (the default): every photo is first read cut to its display, the box asked in
# a request of its own; FOTOMATES_CROP_FIRST=0 reads the whole photo first, as before.
CROP_FIRST = os.environ.get('FOTOMATES_CROP_FIRST', '1').strip() not in ('0', 'false', 'no', 'off', '')
# How an unsure photo is read once more after the whole photo: enlarge (the whole frame, 3x)
# or crop (cut to the box the whole-photo read gives around the display, then enlarged).
# The default is crop with crop first (3x adds no detail to a photo already enlarged; the
# whole read's box is a second estimate, independent of the box finder's), else enlarge.
REREAD_MODE = (os.environ.get('FOTOMATES_REREAD_MODE', '').strip().lower()
               or ('crop' if CROP_FIRST else 'enlarge'))
if REREAD_MODE not in ('enlarge', 'crop'):
    raise ValueError(f"FOTOMATES_REREAD_MODE must be 'enlarge' or 'crop', not {REREAD_MODE!r}")
# The crop: this share of the box's longer side added on every side, then enlarged so it is
# CROP_WIDTH px wide (its shorter side never below FOTOMATES_MIN_SIDE).
CROP_MARGIN = float(os.environ.get('FOTOMATES_CROP_MARGIN', '0.25'))
CROP_WIDTH = int(os.environ.get('FOTOMATES_CROP_WIDTH', '650'))
# The margin of the crop read after the whole photo, with crop first: wider than CROP_MARGIN,
# so a cell cut off at the edge of the first crop is inside this one.
THIRD_MARGIN = float(os.environ.get('FOTOMATES_THIRD_MARGIN', '0.5'))
# A box is used only when it looks like a display: its share of the photo's width and
# height, and its own width / height (in pixels). Width up to 100%: a display can fill it.
BOX_WIDTH, BOX_HEIGHT, BOX_ASPECT = (0.15, 1.0), (0.04, 0.60), (1.2, 8.0)
# FOTOMATES_ROTATED=1 (default off): a box taller than wide (a display photographed sideways)
# is accepted when its height / width is within BOX_ASPECT; the cut is turned 90 degrees
# before it is read.
ROTATED = _on('FOTOMATES_ROTATED')
_BOX_LINE = ("BOX: x1 y1 x2 y2: the box around the register display or counter window, every "
             "digit of the row inside it, as numbers from 0 to 1000 of the photo (x from the left "
             "edge, y from the top edge); NONE when there is none\n")
# With REREAD_MODE crop the read of the whole photo also says where the display is. A read
# of a crop is not asked: its box is of no use, and its prompt stays the one without the line.
_at = _SYSTEM_PROMPT.rindex('QUALITY:')
_LINE_COUNT = {'five': 'six', 'six': 'seven', 'seven': 'eight'}
_BOX_SYSTEM_PROMPT = (re.sub(r'exactly these (five|six|seven) lines',
                             lambda m: f'exactly these {_LINE_COUNT[m[1]]} lines', _SYSTEM_PROMPT[:_at], count=1)
                      + _BOX_LINE + _SYSTEM_PROMPT[_at:])


def system_prompt(crop: bool = False) -> str:
    """The system prompt of a read: with the BOX line for a whole photo when REREAD_MODE is crop.
    With FOTOMATES_PREFIX the instructions of the user message follow it."""
    system = _BOX_SYSTEM_PROMPT if REREAD_MODE == 'crop' and not crop else _SYSTEM_PROMPT
    return system + '\n\n' + _USER_PROMPT if PREFIX else system


BOX_PROMPT = ('Locate the meter\'s register display: the LCD/LED screen or the window of number wheels that '
              'shows the consumption. Answer only with JSON: {"bbox_2d": [x1, y1, x2, y2]}')
# Task 6a. FOTOMATES_BOX_ANSWER=numbers (default json): the box is asked as four integers, a
# quarter of the answer tokens; parse_box takes both forms.
BOX_ANSWER = os.environ.get('FOTOMATES_BOX_ANSWER', 'json').strip().lower()
if BOX_ANSWER not in ('json', 'numbers'):
    raise ValueError(f"FOTOMATES_BOX_ANSWER must be 'json' or 'numbers', not {BOX_ANSWER!r}")
if BOX_ANSWER == 'numbers':
    BOX_PROMPT = _swap(BOX_PROMPT, 'Answer only with JSON: {"bbox_2d": [x1, y1, x2, y2]}',
                       'Answer with four integers from 0 to 1000 separated by commas: x1,y1,x2,y2, nothing '
                       'else; NONE when there is no display.')
# 32, not 16: the model writes [321, 393, 714, 538] with a bracket and spaces as often as
# 321,393,714,538, and digits are one token each; at 16 the fourth number was cut off.
BOX_MAX_TOKENS = 32 if BOX_ANSWER == 'numbers' else 60
if PROMPT in ('v3', 'v4'):  # a tight box can cut the small last cell off
    BOX_PROMPT = _swap(BOX_PROMPT, 'shows the consumption.',
                       'shows the consumption, including any smaller decimal cell at the right end of the row.')
# Task 7. The shorter side, in pixels, the box request's photo is enlarged to (default 480 =
# FOTOMATES_MIN_SIDE, as before); 0 sends the photo as it is (about 95 image tokens instead of 375).
BOX_SIDE = float(os.environ.get('FOTOMATES_BOX_SIDE', str(_MIN_SIDE)))


def parse_box(text):
    """The four numbers of {"bbox_2d": [...]}, or of the first four numbers written; None if fewer."""
    try:
        data = json.loads(re.search(r'\{.*\}', text or '', re.S)[0])
        box = data.get('bbox_2d') if isinstance(data, dict) else None
        if isinstance(box, list) and len(box) >= 4 and isinstance(box[0], list):
            box = box[0]  # [[x1, y1, x2, y2]]
        if isinstance(box, list) and len(box) >= 4:
            return [float(v) for v in box[:4]]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text or '')]
    return numbers[:4] if len(numbers) >= 4 else None


def find_box(image_path: Path, model: str | None = None) -> dict:
    """Ask the model, in a request of its own, where the photo's display is: the photo as the
    first read sends it (shorter side FOTOMATES_MIN_SIDE). Returns box (0-1000, None when
    there is none or it does not look like a display), why (when not), answer and seconds."""
    started = time.monotonic()
    image = decode_image(Path(image_path))
    height, width = image.shape[:2]
    scale = max(1.0, BOX_SIDE / min(height, width))
    sent = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else image
    choice = backend(model)
    servers = choice.servers
    question = [{'type': 'image_url', 'image_url': {'url': png_data_url(sent)}},
                {'type': 'text', 'text': BOX_PROMPT}]
    # FOTOMATES_PREFIX: the same start as a crop read (its system message), so both share the cached prefix.
    messages = ([{'role': 'system', 'content': system_prompt(crop=True)}] if PREFIX else []) + [
        {'role': 'user', 'content': question}]
    answered = model_call(server=servers[next(_TURN) % len(servers)], model=choice.model, temperature=0,
                          max_tokens=BOX_MAX_TOKENS, extra_body=dict(NO_THINKING),
                          messages=messages).choices[0]
    answer = (answered.message.content or '').strip()
    box = parse_box(answer)
    if box and BOX_ANSWER == 'numbers' and getattr(answered, 'finish_reason', None) == 'length':
        # the answer ran into max_tokens: its last number may have lost a digit, so it is no box
        box, rect, why = None, None, 'box answer cut off'
    else:
        rect, why = crop_rect(box, width, height) if box else (None, 'no box')
    return dict(box=tuple(box) if rect else None, why=why, answer=answer, seconds=time.monotonic() - started)


def extract_box(text: str) -> tuple | None:
    """The four numbers of the last BOX line (0-1000 of the photo), or None."""
    lines = re.findall(r'^BOX:[ \t]*([^\n]*)', text or '', re.IGNORECASE | re.MULTILINE)
    numbers = re.findall(r'\d+(?:\.\d+)?', lines[-1]) if lines else []
    return tuple(float(n) for n in numbers[:4]) if len(numbers) >= 4 else None


def box_turned(box, width: int, height: int) -> bool:
    """With FOTOMATES_ROTATED: the box (0-1000) is taller than wide in pixels and its
    height / width is within BOX_ASPECT, the shape of a display photographed sideways."""
    if not ROTATED or box is None:
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return False
    tall = (y2 - y1) * height / ((x2 - x1) * width)
    return BOX_ASPECT[0] <= tall <= BOX_ASPECT[1]


def crop_rect(box, width: int, height: int, margin: float | None = None):
    """The pixels (x1, y1, x2, y2) of the photo to cut for box (0-1000), with margin (by
    default CROP_MARGIN) and clipped to the photo, and None; or None and why the box is not used."""
    if box is None:
        return None, 'no box'
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None, 'box outside the photo'
    share_w, share_h = (x2 - x1) / 1000, (y2 - y1) / 1000
    aspect = share_w * width / (share_h * height)
    if box_turned(box, width, height):  # a display on its side: its long side runs down the photo
        if not BOX_WIDTH[0] <= share_h <= BOX_WIDTH[1]:
            return None, f'box height {100 * share_h:.0f}%'
        if not BOX_HEIGHT[0] <= share_w <= BOX_HEIGHT[1]:
            return None, f'box width {100 * share_w:.0f}%'
    else:
        if not BOX_WIDTH[0] <= share_w <= BOX_WIDTH[1]:
            return None, f'box width {100 * share_w:.0f}%'
        if not BOX_HEIGHT[0] <= share_h <= BOX_HEIGHT[1]:
            return None, f'box height {100 * share_h:.0f}%'
        if not BOX_ASPECT[0] <= aspect <= BOX_ASPECT[1]:
            return None, f'box aspect {aspect:.2f}'
    px1, py1, px2, py2 = x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height
    margin = (CROP_MARGIN if margin is None else margin) * max(px2 - px1, py2 - py1)
    return (max(0, int(px1 - margin)), max(0, int(py1 - margin)),
            min(width, int(round(px2 + margin))), min(height, int(round(py2 + margin)))), None


def _request_ocr(image_path: Path, choice: Backend = DEFAULT_BACKEND, scale: float = 1.0,
                 crop: bool = False, box=None, margin: float | None = None, hint: str | None = None) -> OcrResult:
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
                                                        scale, crop, box, margin, hint=hint)
    except Exception as error:
        reason = _failure_reason(error)
        raise OcrFailure(reason, breakdown(*server, 'FAILED: ' + reason)) from error
    confidence, per_digit, margins, runner_ups, alternatives = _digit_confidence(choice)
    written, cells_read = _reading_line(content), _cells_digits(content)
    if extract_device_type(content) == 'digital' and written and cells_read and cells_read != written:
        print(f'CELLS {cells_read} differ from READING {written}: CELLS kept ({Path(image_path).name})', flush=True)
    # A register code glued to the front of READING (380..., 158...) is cut from the
    # digits by _reading_digits; cut its probabilities too, so each stays with its digit.
    code_confidence = []
    kept = len(_reading_digits(content) or '')
    if kept and len(per_digit) > kept:
        cut = len(per_digit) - kept
        code_confidence = per_digit[:cut]
        per_digit, margins, runner_ups, alternatives = (
            values[cut:] for values in (per_digit, margins, runner_ups, alternatives))
        confidence = min(per_digit)
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
                     device_type=extract_device_type(content),
                     decimal_digits=extract_decimal_digits(content),
                     clear=_clear_probability(choice), code_confidence=code_confidence,
                     image_report={k: v for k, v in report.items() if k != 'sent_image'})


def _read_photo(image_path: Path, timed, aids: list, server: list, model_seconds: list,
                backend_choice: Backend = DEFAULT_BACKEND, scale: float = 1.0, crop: bool = False, box=None,
                margin: float | None = None, turn: int = 90, hint: str | None = None):
    # The photo is decoded to judge it: whether it is too small or needs enhancing.
    report = dict(blur_variance=0.0)
    needs = []
    original = timed('decode image', lambda: decode_image(image_path))
    size = dict(width=int(original.shape[1]), height=int(original.shape[0]))
    cropped = skipped = turned = False
    if crop:  # the re-read of a crop: cut the box from the original photo, then enlarge it
        rect, why = crop_rect(box, size['width'], size['height'], margin)
        if rect:
            x1, y1, x2, y2 = rect
            original = original[y1:y2, x1:x2]
            turned = box_turned(box, size['width'], size['height'])
            if turned:  # a display on its side: turned upright, clockwise first (turn 90)
                original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE if turn == 90
                                      else cv2.ROTATE_90_COUNTERCLOCKWISE)
            scale = max(CROP_WIDTH / original.shape[1], _MIN_SIDE / min(original.shape[:2]))
            cropped = True
        else:
            aids.append(f'crop-fallback({why})')  # the whole frame, enlarged by scale, instead
    # A small photo is enlarged until its shorter side reaches _MIN_SIDE: on a
    # 240x400 photo a digit is ~10 px, smaller than one image patch of the model.
    scale = max(scale, _MIN_SIDE / min(original.shape[:2]))
    enlarged = scale > 1.0
    if _ENHANCE_MODE != 'never' or enlarged or cropped:
        if enlarged or cropped:
            original = timed('enlarge photo', lambda: cv2.resize(original, None, fx=scale, fy=scale,
                                                                 interpolation=cv2.INTER_CUBIC))
        report = timed('assess quality', lambda: assess(original))
        if _ENHANCE_MODE == 'always':
            needs = ['always']
        elif _ENHANCE_MODE == 'auto' or (_ENHANCE_MODE == 'whole' and not cropped):
            needs = [flag for flag in _ENHANCE_WHEN if report.get(flag)]
            if needs and report.get('is_dark') and report.get('contrast', 255.0) < BLANK_MAX:
                needs = []  # a black cut: brightening it only turns noise into shapes
                skipped = True
    if needs or enlarged or cropped:
        photo = timed('enhance photo', lambda: enhance(original, report)) if needs else original
        encoded = timed('encode png + base64', lambda: png_data_url(photo))
        if cropped:
            aids.append(f'crop({scale:.3g}x,rot{turn})' if turned else f'crop({scale:.3g}x)')
        elif enlarged:
            aids.append(f'enlarged({scale:g}x)')
        if needs:
            aids.append('enhanced(' + ','.join(needs) + ')')
        elif skipped:
            aids.append('enhance-skipped(black)')
    else:
        encoded = timed('read photo file', lambda: photo_data_url(image_path))
        aids.append('original')
    # The photo as prepared is what the model reads, never the original beside it.
    report = dict(report, **size, sent_image=encoded if needs or enlarged else None)

    shots = fewshots_for_this_read()  # FOTOMATES_FEWSHOT: example photos with their CELLS line, else nothing
    images = [
        *shots,
        *([{"type": "text", "text": "The meter photo to read:" if shots else "Meter photo."}]
          if shots or not PREFIX else []),
        {"type": "image_url", "image_url": {"url": encoded}},
        # A line for this photo only (FOTOMATES_CELL_HISTORY: "This display has N cells."), after the image.
        *([{"type": "text", "text": hint}] if hint else []),
    ]
    if hint:
        aids.append('cells-hint')
    # FOTOMATES_PREFIX: the instructions are in the system message; the user message is the image.
    user = images if PREFIX else [{"type": "text", "text": _USER_PROMPT}, *images]

    servers = backend_choice.servers
    model_server = servers[next(_TURN) % len(servers)]
    if len(servers) > 1:
        aids.append(f'server{servers.index(model_server) + 1}')
    read_started = time.monotonic()
    extra = dict(NO_THINKING)

    def read(system: str, temperature: float = 0, seed: int = 0, timeout: float | None = None):
        response = model_call(
            server=model_server,
            **({'timeout': timeout} if timeout is not None else {}),
            model=backend_choice.model, temperature=temperature, seed=seed, max_tokens=_READ_MAX_TOKENS,
            logprobs=True,  # per-digit probabilities, for the score
            top_logprobs=5,  # and each digit's runner-ups: a close second digit is a confusion
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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

    choice = timed('model request', lambda: read(system_prompt(crop)))
    content = with_reading_line(choice.message.content or '')  # v4: a READING line made from CELLS
    readings = [timed('extract reading', lambda: extract_reading(content))]
    if turned and turn == 90 and (not readings[0] or extract_device_type(content) is None):
        # Nothing read: the display may be on its other side. Read once more turned the other
        # way and keep that read when it has digits.
        other_aids = []
        other = _read_photo(image_path, timed, other_aids, server, model_seconds, backend_choice,
                            scale, crop, box, margin, turn=270, hint=hint)
        if other[3][0]:
            aids[:] = other_aids
            return other
    return content, choice, report, readings


def settings_note() -> str:
    """The settings of Round 2 that are not at their default, for a run's first log line."""
    from detect_quality import ENHANCER
    notes = [f'enhance {_ENHANCE_MODE}' if _ENHANCE_MODE != 'auto' else '',
             f'enhancer {ENHANCER}' if ENHANCER != 'light' else '',
             'rotated boxes accepted' if ROTATED else '',
             'repeated-digit line' if REPEAT_HINT else '',
             f'{len(_FEWSHOTS) // 3} example photos on {FEWSHOT_ON} reads' if _FEWSHOTS else '',
             'box as four numbers' if BOX_ANSWER != 'json' else '',
             f'box photo side {BOX_SIDE:g}' if BOX_SIDE != _MIN_SIDE else '',
             'instructions as a cached prefix' if PREFIX else '',
             'cache_prompt' if CACHE_PROMPT else '',
             f'slots {",".join(map(str, SLOT_IDS))}' if SLOT_IDS else '',
             'streamed' if STREAM else '',
             f'near-black check below {BLANK_DIM:g} with 99th percentile under {BLANK_P99:g}' if BLANK_P99 > 0 else '',
             f'black frame below {BLANK_MAX:g}' if BLANK_MAX != 5 else '']
    return ''.join('; ' + note for note in notes if note)


def warm_up() -> None:
    for choice in BACKENDS.values():
        for server in choice.servers:
            try:
                complete(server=server, model=choice.model,
                         max_tokens=1, messages=[{"role": "user", "content": "ok"}], extra_body=NO_THINKING)
                print(f'model warm: {choice.label} at {server.url}', flush=True)
            except Exception as error:
                print(f'warm-up skipped for {server.url}: {type(error).__name__}: {error}', flush=True)


def ocr_image(image_path: Path, model: str | None = None, scale: float = 1.0,
              crop: bool = False, box=None, margin: float | None = None, hint: str | None = None) -> OcrResult:
    """Read a photo. crop: read the box (0-1000, from the box request or a whole-photo read's
    BOX line) cut from the photo with margin (by default CROP_MARGIN); when the box is missing
    or does not look like a display, the whole photo enlarged by scale, labelled crop-fallback."""
    return _request_ocr(Path(image_path), backend(model), scale, crop, box, margin, hint)











