import base64
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from openai import APIConnectionError, APITimeoutError, OpenAI
from detect_quality import assess, enhance_for_ocr


OCR_URL = os.environ.get("FOTOMATES_OCR_URL", "http://4.232.241.238:28000/v1")
OCR_MODEL = os.environ.get("FOTOMATES_OCR_MODEL", "qwen3.5-122b")
OCR_API_KEY = os.environ.get("FOTOMATES_OCR_API_KEY", "EMPTY")

_THINKING = False
_READ_MAX_TOKENS = 1024


BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "segmented_readings"
SOURCE_DIR = BASE_DIR / "foto mates"
HTML_PATH = BASE_DIR / "extracted_readings.html"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

client = OpenAI(
    base_url=OCR_URL,
    api_key=OCR_API_KEY,
    timeout=1800.0,
)
_MODEL_LOADING_WAIT = 900.0
_MODEL_LOADING_POLL = 15.0


_TRANSIENT_ATTEMPTS = 4
_TRANSIENT_BACKOFF = 4.0


def _is_model_loading(error: Exception) -> bool:
    if getattr(error, 'status_code', None) != 503:
        return False
    return 'loading model' in str(error).lower()


def _is_transient(error: Exception) -> bool:
    return isinstance(error, (APIConnectionError, APITimeoutError)) or \
        getattr(error, 'status_code', None) in (500, 502, 503, 504)


def complete(**request):
    deadline = time.monotonic() + _MODEL_LOADING_WAIT
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**request)
        except Exception as error:
            if _is_model_loading(error) and time.monotonic() < deadline:
                time.sleep(_MODEL_LOADING_POLL)
                continue
            attempt += 1
            if not _is_transient(error) or attempt >= _TRANSIENT_ATTEMPTS:
                raise
            time.sleep(_TRANSIENT_BACKOFF * attempt)


_MIN_LONG_SIDE = 1600


def encode_image(image: Path | np.ndarray) -> str:
    pixels = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, Path) else image
    scale = _MIN_LONG_SIDE / max(pixels.shape[:2])
    if scale > 1:
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, encoded_image = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError("Could not encode OCR image")
    encoded = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _digit_count(reading: str) -> int:
    return sum(ch.isdigit() for ch in reading)


def extract_reading(text: str) -> str:
    answers = re.findall(r"READING:\s*\**\s*([^\s*]+)", text, re.IGNORECASE)
    reading = answers[-1].rstrip(".,") if answers else text.strip()
    return reading if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", reading) else ""


@dataclass
class OcrResult:
    text: str
    avg_confidence: float | None
    min_confidence: float | None
    min_margin: float | None
    evidence_source: str = "unknown"


def _final_answer_tokens(tokens):
    text, start = '', 0
    for i, tok in enumerate(tokens):
        text += tok.token
        if tok.token == '</think>' or text.rstrip().upper().endswith('READING:'):
            start = i + 1
    return tokens[start:] or tokens


def _confidence_from_logprobs(logprobs) -> tuple[float | None, float | None]:
    if logprobs is None or not logprobs.content:
        return None, None
    probs = [math.exp(tok.logprob) for tok in _final_answer_tokens(logprobs.content)]
    return sum(probs) / len(probs), min(probs)


def _confusability_margin(logprobs) -> float | None:
    if logprobs is None or not logprobs.content:
        return None

    margins = []
    for tok in _final_answer_tokens(logprobs.content):
        alts = tok.top_logprobs or []
        if len(alts) < 2:
            continue
        top1, top2 = sorted((alt.logprob for alt in alts), reverse=True)[:2]
        margins.append(math.exp(top1) - math.exp(top2))

    return min(margins) if margins else None


_SYSTEM_PROMPT = "You transcribe meter readings from photographs."
_FINAL_LINE_REMINDER = (
    " End your reply with one final line: READING: followed by the reading, or "
    "READING: UNREADABLE."
)
_USER_PROMPT = (
    "Read the main consumption register: the row of digits that gives the energy "
    "reading.\n\n"
    "Return every digit of that row in order, including leading zeros, any "
    "fractional separator and the digits after it. On a mechanical counter the "
    "fractional wheels are often set apart by colour and are easy to miss, but "
    "they are part of the reading. A wheel turning between two numbers still "
    "counts: give the smaller one, or 9 between 9 and 0. Return UNREADABLE only if "
    "a digit cannot be made out.\n\n"
    "Decimal separator: a reading with no separator is normal, because many "
    "registers show whole units only. Put one only where the register itself "
    "shows it:\n"
    "- Digital display: a dot or comma on the baseline, level with the bottom of "
    "the digits, between two digits of the reading row. It can be tiny and can "
    "touch the digits.\n"
    "- Mechanical counter: before the first fractional wheel. That wheel is set "
    "apart from the whole-number wheels by colour in any hue (a coloured box or "
    "background, coloured digits, or a coloured frame or outline around its "
    "window; judge colour from the original photo), by a comma or dot printed just "
    "before it, or by a fraction label such as 0,1 over it.\n"
    "Nothing else is a separator, even when it lines up with the gap between two "
    "digits: arrows, triangles, pins and other icons above or below the row; marks "
    "before the first digit or after the last; ticks or dividers repeated between "
    "wheels; screws, seals, dirt, glare, and reflections; dots inside other "
    "numbers such as a 1.8.0 code; blank gaps; and how many digits the meter has. "
    "If you cannot point to the mark itself, give no separator.\n\n"
    "Read digit and separator strokes from the enhanced image. The original photo "
    "is for colour and for telling the register apart from nearby text and "
    "symbols; never take a stroke from it.\n\n"
    "Check each digit and any separator mark against these rules briefly, then end "
    "your reply with one final line: READING: followed by the reading, or "
    "READING: UNREADABLE."
)


def _request_ocr(image_path: Path) -> OcrResult:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError('Could not decode image')


    report = assess(original)

    corrected = enhance_for_ocr(original, report, force_gmm=True)
    
    images = [
        {"type": "text", "text": "Enhanced image — read the register from this image."},
        {"type": "image_url", "image_url": {"url": encode_image(corrected)}},
        {"type": "text", "text": "Original photo — for colour and context only; do not read digit or decimal strokes from it."},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]
    aids = ['enhanced', 'original_context']

    def read(system: str):
        response = complete(
            model=OCR_MODEL, temperature=0, seed=0, max_tokens=_READ_MAX_TOKENS,
            logprobs=True, top_logprobs=2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": _USER_PROMPT}, *images]},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": _THINKING}},
        )
        return response.choices[0]

    choice = read(_SYSTEM_PROMPT)
    if getattr(choice, 'finish_reason', None) == 'length':
        raise ValueError('Model response was truncated; no complete reading returned')
    content = choice.message.content or ''
    main_reading = extract_reading(content)
    if not main_reading and 'UNREADABLE' not in content.upper():
        retry = read(_SYSTEM_PROMPT + _FINAL_LINE_REMINDER)
        if getattr(retry, 'finish_reason', None) != 'length':
            retry_reading = extract_reading(retry.message.content or '')
            if retry_reading:
                choice, content, main_reading = retry, retry_reading, retry_reading
                aids.append('reread')
    
    logprobs = getattr(choice, 'logprobs', None)
    average, minimum = _confidence_from_logprobs(logprobs)
    evidence_source = '+'.join(aids)

    return OcrResult(content, average, minimum, _confusability_margin(logprobs),
                     evidence_source=evidence_source)


def ocr_image(image_path: Path) -> OcrResult:
    return _request_ocr(Path(image_path))


def write_html_report(rows, image_dir_name="segmented_readings"):
    cards = []
    for file_name, reading, raw_text, digit_count in rows:
        metrics = f"digits: {digit_count}"
        cards.append(f"""
        <figure>
            <img src="{image_dir_name}/{file_name}" alt="{file_name}">
            <figcaption>
                {file_name}<br>
                <strong>{reading}</strong>
                <div class="metrics">{metrics}</div>
                <small>{raw_text}</small>
            </figcaption>
        </figure>
        """)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Extracted meter readings</title>

<style>
body {{
    font-family: Arial, sans-serif;
    background:#111;
    color:#eee;
    margin:30px;
}}

.grid {{
    display:flex;
    flex-wrap:wrap;
    gap:20px;
}}

figure {{
    width:260px;
    background:#1f1f1f;
    padding:10px;
    border-radius:8px;
    margin:0;
}}

img {{
    width:100%;
    border-radius:5px;
}}

figcaption {{
    margin-top:8px;
    font-size:14px;
}}

strong {{
    color:#67d667;
    font-size:18px;
}}

.metrics {{
    margin-top:4px;
    color:#9cc2ff;
    font-size:11px;
}}

small {{
    color:#888;
    white-space:pre-wrap;
}}
</style>

</head>

<body>

<h1>Extracted Meter Readings</h1>

<div class="grid">
{''.join(cards)}
</div>

</body>
</html>
"""

    HTML_PATH.write_text(html, encoding="utf-8")


def main():
    rows = []

    if SOURCE_DIR.is_dir():
        active_input_dir = SOURCE_DIR
    elif INPUT_DIR.is_dir():
        active_input_dir = INPUT_DIR
    else:
        raise FileNotFoundError(f'No input images directory: {SOURCE_DIR} or {INPUT_DIR}')

    image_files = sorted(
        p for p in active_input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    print(f"Found {len(image_files)} images.\n")

    for path in image_files:
        reading = ""
        raw_text = ""
        try:
            result = ocr_image(path)
            raw_text = result.text
            reading = extract_reading(raw_text)
            status = reading if reading else "(no digits found)"

        except Exception as e:
            status = f"ERROR: {e}"


        digit_count = _digit_count(reading)
        rows.append((path.name, status, raw_text, digit_count))
        print(f"{path.name:<35} {status:<16} digits={digit_count}")

    write_html_report(rows, active_input_dir.name)

    print(f"\nHTML report written to:\n{HTML_PATH}")

if __name__ == "__main__":
    main()
