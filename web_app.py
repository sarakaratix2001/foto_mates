"""Local upload UI for FotoMates, served by FastAPI. Run: python web_app.py"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Path as PathParam, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from audit_readings import (AUDIT_DB, LIST_KINDS, connect as audit_connect, list_query, parse_name, read_best,
                            split_decimals, summary as audit_summary)
from evaluation import LABELS as EVALUATION_LABELS, evaluate, lines as evaluation_lines
from extract_readings import (BACKENDS as MODEL_BACKENDS, OCR_MODEL, PHOTO_TYPES, QUALITY_DIGIT, QUALITY_SCORE,
                              RETAKE_SCORE, _reading_digits, photo_retake_reason, backend as model_backend, extract_blank, extract_no_meter, score_reading,
                              WORKERS_PER_SLOT, low_score_reasons, ocr_image, quality_reasons,
                              server_slots, warm_up)

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ.get('FOTOMATES_RUNS', BASE_DIR / 'web_runs'))
MAX_PHOTOS = int(os.environ.get('FOTOMATES_MAX_PHOTOS', '1000'))
MAX_UPLOAD_MB = int(os.environ.get('FOTOMATES_MAX_UPLOAD_MB', '5000'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED = set(PHOTO_TYPES)  # kept as uploaded, so only types the model server reads


JOB_WORKERS = int(os.environ.get('FOTOMATES_JOB_WORKERS', '1'))
WORKER = ThreadPoolExecutor(max_workers=JOB_WORKERS)

IMAGE_WORKERS = int(os.environ.get('FOTOMATES_IMAGE_WORKERS', '1'))
LOCK = threading.RLock()

JOB_ID = r'^[a-f0-9]{32}$'
IMAGE_NAME = r'^[a-f0-9]{32}\.(?:jpg|jpeg|png|bmp)$'
NO_STORE = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
IMMUTABLE = {'Cache-Control': 'public, max-age=31536000, immutable',
             'X-Content-Type-Options': 'nosniff'}


def answer(payload, status=200):
    return JSONResponse(payload, status_code=status, headers=NO_STORE)


def save_job(job):
    with LOCK:
        root = RUNS_DIR / job['id']
        temporary = root / 'results.tmp'
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(root / 'results.json')


def read_job(job_id):
    with LOCK:
        job = json.loads((RUNS_DIR / job_id / 'results.json').read_text(encoding='utf-8'))
    return job


def quality_note(score, parts, seconds=None):

    weakest = (parts or {}).get('weakest')
    if score is None or (score >= QUALITY_SCORE and (weakest is None or weakest >= QUALITY_DIGIT)):
        return None
    note = f"score {score:.2f}: " + (
        '; '.join(quality_reasons(parts)) if parts else 'the overall confidence is low')
    if seconds is not None:
        note += f". Read in {seconds:.1f} s"
    return note


def verdict(reading, score=None, parts=None, seconds=None, text=None, clear=None, cells=None):

    if not reading and extract_blank(text):
        return dict(status='blank', retake=True, quality_note=None,
                    retake_reason='the photo is blank: nothing recognisable in it')
    if not reading and extract_no_meter(text):
        return dict(status='no_meter', retake=True, quality_note=None,
                    retake_reason='no meter found in the photo')
    if not reading:
        return dict(status='no_digits', retake=True, quality_note=None,
                    retake_reason='no reading could be taken from the photo')
    note = quality_note(score, parts, seconds)
    photo = photo_retake_reason(cells or '', clear)
    if photo:
        return dict(status='done', retake=True, quality_note=note, retake_reason=(
            f'{photo}: retake the photo straight on, sharp and without glare, with the whole display in the frame'))
    if score is not None and score < RETAKE_SCORE:
        reasons = '; '.join(low_score_reasons(parts)) if parts else 'the overall confidence is low'
        return dict(status='done', retake=True, quality_note=note, retake_reason=(
            f'confidence score {score:.2f}: {reasons}. A digit may be wrong or missing: '
            'retake the photo straight on, sharp and without glare, with the whole display in the frame'))
    return dict(status='done', retake=False, retake_reason=None, quality_note=note)


def consumption(filename, cells, per_digit=None, margins=None, runner_ups=None, alternatives=None,
                device_type=None, decimal_digits=None):

    cells = re.sub(r'\D', '', cells or '')
    stored = parse_name(Path(filename or '').name)['saved_reading']
    visual = decimal_digits if device_type == 'mechanical' else None
    # The one score, from the photo alone: every digit read, nothing from the file name.
    parts = score_reading(cells, per_digit, visual, margins, runner_ups, alternatives,
                          device_type == 'mechanical') if cells else None
    # Only then the comparison with the saved reading in the file name.
    whole, decimals, match = split_decimals(cells or None, stored, visual)
    return dict(reading=whole or '', all_digits=cells, digits=list(whole or ''), whole_reading=whole,
                stored_reading=stored, decimals=decimals, match=match if stored and cells else None,
                score=parts['total'] if parts else None, score_parts=parts)


def with_verdicts(job):
    for item in job['images']:
        if item.get('status') in ('done', 'no_digits', 'no_meter', 'blank'):
            item.update(consumption(item.get('filename'), item.get('all_digits') or item.get('reading'),
                                    item.get('digit_confidence'), item.get('digit_margin'),
                                    item.get('digit_runner_up'), item.get('digit_alternatives'),
                                    item.get('device_type'), item.get('decimal_digits')))
            item.update(verdict(item.get('reading'), item.get('score'), item.get('score_parts'),
                                item.get('seconds'), item.get('raw_text'), item.get('clear'), item.get('all_digits')))
    job['evaluation'] = batch_evaluation(job)
    job['evaluation_lines'] = evaluation_lines(job['evaluation'])
    return job


def batch_evaluation(job):
    return evaluate(dict(name=item.get('filename'), saved=item.get('stored_reading'), extracted=item.get('reading'),
                         all_digits=item.get('all_digits'), score=item.get('score'), status=item.get('status'),
                         no_meter=extract_no_meter(item.get('raw_text')), blank=extract_blank(item.get('raw_text')))
                    for item in job['images'] if item.get('status') in ('done', 'no_digits', 'no_meter', 'blank', 'error'))


def process_image(job, item):
    item['status'] = 'processing'
    save_job(job)
    started = time.monotonic()
    reading_seconds = None
    try:
        image_path = RUNS_DIR / job['id'] / item['image']
        best = read_best(image_path, parse_name(item['filename'])['saved_reading'], job.get('model'))
        result = best['result']
        read = consumption(item['filename'], _reading_digits(result.text), result.digit_confidence,
                           result.digit_margin, result.digit_runner_up, result.digit_alternatives,
                           result.device_type, result.decimal_digits)
        reading = read['reading']
        item.pop('error', None)
        # The enhanced photo the model read, kept to show beside the original.
        if item.get('enhanced_image'):
            (image_path.parent / item.pop('enhanced_image')).unlink(missing_ok=True)
        if result.sent_png:
            item['enhanced_image'] = uuid.uuid4().hex + '.png'
            (image_path.parent / item['enhanced_image']).write_bytes(result.sent_png)
        reading_seconds = best['seconds']
        item.update(raw_text=result.text, **read,
                    evidence_source=result.evidence_source,
                    confidence=result.confidence,
                    digit_confidence=result.digit_confidence, reread=best['reread'],
                    digit_margin=result.digit_margin, digit_runner_up=result.digit_runner_up,
                    digit_alternatives=result.digit_alternatives,
                    rolling=result.rolling, device_type=result.device_type, decimal_digits=result.decimal_digits,
                    clear=result.clear, code_confidence=result.code_confidence,
                    **verdict(reading, read['score'], read['score_parts'], text=result.text,
                              clear=result.clear, cells=read['all_digits']))
    except Exception as exc:
        item.update(status='error', error=str(exc),
                    raw_text=getattr(exc, 'breakdown', '').lstrip('\n'))
    # The model reading the photo plus digit extraction; a failed read has no
    # reading, so it shows how long it took to fail.
    item['seconds'] = round(time.monotonic() - started if reading_seconds is None else reading_seconds, 1)
    if item['status'] != 'error':
        item.update(verdict(item.get('reading'), item.get('score'), item.get('score_parts'), item['seconds'],
                            item.get('raw_text')))
    print(f"{item['filename']}: {item.get('all_digits') or '-'} -> {item.get('reading') or '-'} "
          f"({item['seconds']}s)", flush=True)
    if item.get('quality_note'):
        print(f"  quality: {item['quality_note']}", flush=True)
    save_job(job)


def image_workers(choice):
 
    slots = server_slots(choice.servers)
    if choice.fixed_workers or not slots:
        workers = choice.workers
    else:
        workers = min(IMAGE_WORKERS, slots * WORKERS_PER_SLOT)
    print(f'reading with {workers} worker(s) on {choice.label}; model server slots: {slots or "not reported"}',
          flush=True)
    return workers


def process_job(job):
    job['status'] = 'processing'
    job['started_at'] = time.time()  # the page counts the batch's elapsed time from it
    save_job(job)
    started = time.monotonic()
    pending = [item for item in job['images'] if item['status'] in ('queued', 'processing')]
    choice = model_backend(job.get('model'))
    job['model_label'] = choice.label
    with ThreadPoolExecutor(max_workers=image_workers(choice)) as pool:
        list(pool.map(lambda item: process_image(job, item), pending))
    job['seconds'] = round(time.monotonic() - started, 1)
    job['status'] = 'done'
    print(f"batch of {len(job['images'])} photos: {job['seconds']}s "
          f"({job['seconds'] / max(len(job['images']), 1):.2f}s per photo)", flush=True)
    save_job(job)


def readable(payload):
    picture = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) if payload else None
    return picture is not None and min(picture.shape[:2]) >= 3


def create_job(uploads, model=None):
    job = dict(id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
               status='queued', images=[], model=model_backend(model).key,
               model_label=model_backend(model).label)
    root = RUNS_DIR / job['id']
    root.mkdir(parents=True)
    try:
        for filename, stream in uploads:
            if Path(filename).suffix.lower() not in ALLOWED:
                raise ValueError(f'Unsupported image type: {filename}')
            payload = stream.read()
            if not readable(payload):
                raise ValueError(f'Cannot read image: {filename}')
            # Stored byte for byte as uploaded, under its own extension.
            image = uuid.uuid4().hex + Path(filename).suffix.lower()
            (root / image).write_bytes(payload)
            job['images'].append(dict(filename=filename, image=image, status='queued'))
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    save_job(job)
    WORKER.submit(process_job, job)
    return job['id']


def delete_image(job_id, image_name):
    with LOCK:
        job = read_job(job_id)
        remaining = [i for i in job['images'] if i['image'] != image_name]
        if len(remaining) == len(job['images']):
            raise ValueError('Image not found in this upload')
        root = RUNS_DIR / job_id
        if not remaining:
            shutil.rmtree(root, ignore_errors=True)
            return None
        for item in job['images']:
            if item['image'] == image_name:
                for name in (image_name, item.get('enhanced_image')):
                    if name:
                        (root / name).unlink(missing_ok=True)
        job['images'] = remaining
        save_job(job)
        return job


def delete_job(job_id):
    with LOCK:
        root = RUNS_DIR / job_id
        if not (root / 'results.json').is_file():
            raise ValueError('Upload not found')
        shutil.rmtree(root, ignore_errors=True)


def resume_jobs():
    for result in sorted(RUNS_DIR.glob('*/results.json'), key=lambda path: path.stat().st_mtime):
        job = read_job(result.parent.name)
        if job['status'] not in ('queued', 'processing'):
            continue
        for item in job['images']:
            if item['status'] == 'processing':
                item['status'] = 'queued'
        job['status'] = 'queued'
        save_job(job)
        WORKER.submit(process_job, job)


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    resume_jobs()
    threading.Thread(target=warm_up, daemon=True).start()
    yield
    WORKER.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title='FotoMates', summary='Read meter readings from photos.',
              lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return answer({'error': exc.detail}, exc.status_code)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    if request.url.path == '/api/upload':
        return answer({'error': 'Expected an image upload'}, 400)
    return answer({'error': 'Not found'}, 404)


@app.get('/', include_in_schema=False)
def page():
    return FileResponse(BASE_DIR / 'web_index.html', media_type='text/html; charset=utf-8',
                        headers=NO_STORE)


@app.get('/api/jobs')
def list_jobs():
    jobs = []
    for result in RUNS_DIR.glob('*/results.json'):
        try:
            jobs.append(with_verdicts(read_job(result.parent.name)))
        except (OSError, ValueError):
            continue
    return answer(sorted(jobs, key=lambda j: j['created_at'], reverse=True))

@app.get('/api/jobs/{job_id}')
def show_job(job_id: str = PathParam(pattern=JOB_ID)):
    try:
        return answer(with_verdicts(read_job(job_id)))
    except (OSError, ValueError):
        return answer({'error': 'Upload not found'}, 404)


@app.get('/api/jobs/{job_id}/evaluation.csv')
def evaluation_csv(job_id: str = PathParam(pattern=JOB_ID)):
    """Every photo of the batch with its evaluation group, as CSV: the file
    names behind each count on the page."""
    try:
        evaluation = batch_evaluation(with_verdicts(read_job(job_id)))
    except (OSError, ValueError):
        return answer({'error': 'Upload not found'}, 404)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['group', 'file_name', 'saved_reading', 'extracted_rules_applied', 'all_digits_read', 'score'])
    for key, listed in evaluation['files'].items():
        for row in listed:
            writer.writerow([EVALUATION_LABELS[key], row['name'], row['saved'], row['extracted'],
                             row['all_digits'], '' if row['score'] is None else f"{row['score']:.3f}"])
    return Response('﻿' + out.getvalue(), media_type='text/csv; charset=utf-8', headers={
        **NO_STORE, 'Content-Disposition': f'attachment; filename="fotomates-{job_id}-evaluation.csv"'})


LIVE_MAX_BYTES = int(os.environ.get('FOTOMATES_LIVE_MAX_MB', '20')) * 1024 * 1024



@app.post('/api/read')
def read_now(image: UploadFile = File(...)):
    started = time.monotonic()
    payload = image.file.read()
    if not 0 < len(payload) <= LIVE_MAX_BYTES:
        return answer({'error': f'Send one photo up to {LIVE_MAX_BYTES // 1024 // 1024} MB.'}, 413)
    suffix = Path(image.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        return answer({'error': f'Send a {", ".join(sorted(ALLOWED))} photo.'}, 400)
    if not readable(payload):
        return answer({'error': 'Cannot read this file as an image.'}, 400)

    # Kept only while it is read, byte for byte as sent.
    live = RUNS_DIR / 'live'
    live.mkdir(parents=True, exist_ok=True)
    path = live / (uuid.uuid4().hex + suffix)
    path.write_bytes(payload)
    try:
        best = read_best(path, parse_name(image.filename or '')['saved_reading'])
        result = best['result']
    except Exception as exc:
        detail = {'error': str(exc), 'raw_text': getattr(exc, 'breakdown', '').lstrip('\n'),
                  'seconds': round(time.monotonic() - started, 2)}
        return answer(detail, 502)
    finally:
        path.unlink(missing_ok=True)

    read = consumption(image.filename, _reading_digits(result.text), result.digit_confidence,
                       result.digit_margin, result.digit_runner_up, result.digit_alternatives,
                       result.device_type, result.decimal_digits)
    reading = read['reading']
    seconds = round(best['seconds'] or time.monotonic() - started, 2)
    return answer({
        'filename': image.filename,
        **read,
        'reading': reading or None,
        'reread': best['reread'],
        **verdict(reading, read['score'], read['score_parts'], seconds,
                  clear=result.clear, cells=read['all_digits']),
        'clear': result.clear,
        'seconds': seconds,
        'evidence_source': result.evidence_source,
        'confidence': result.confidence,
        'digit_confidence': result.digit_confidence,
        'rolling': result.rolling,
        'device_type': result.device_type,
        'decimal_digits': result.decimal_digits,
        'raw_text': result.text,
    })


AUDIT_PAGE_SIZE = int(os.environ.get('FOTOMATES_AUDIT_PAGE_SIZE', '100'))
AUDIT_KIND = '^(' + '|'.join(LIST_KINDS) + ')$'
AUDIT_MONTH = r'^(\d{4}-\d{2}|unknown)?$'


def audit_db():

    if not AUDIT_DB.is_file():
        return None
    return audit_connect(AUDIT_DB, check_same_thread=False)  # a CSV streams over several threads


@app.get('/audit', include_in_schema=False)
def audit_page():
    return FileResponse(BASE_DIR / 'audit.html', media_type='text/html; charset=utf-8', headers=NO_STORE)


@app.get('/api/audit')
def audit_overview():
    """Per month: photos, read correctly, mismatched, missing, failed, % correct."""
    info = {'model': OCR_MODEL, 'database': str(AUDIT_DB), 'kinds': LIST_KINDS}
    db = audit_db()
    if db is None:
        return answer({**info, 'available': False, 'months': []})
    try:
        cursor = audit_summary(db)
        columns = [column[0] for column in cursor.description]
        months = [dict(zip(columns, row)) for row in cursor]
    finally:
        db.close()
    return answer({**info, 'available': True, 'months': months})


@app.get('/api/audit/rows')
def audit_rows(kind: str = Query('mismatches', pattern=AUDIT_KIND),
               month: str = Query('', pattern=AUDIT_MONTH),
               contract: str = Query('', max_length=100),
               page: int = Query(0, ge=0)):
    """One page of a list (mismatches, missing, duplicates or all)."""
    db = audit_db()
    if db is None:
        return answer({'error': 'No audit results yet: run audit_readings.py read <folder>'}, 404)
    sql, params = list_query(kind, month or None, contract.strip() or None)
    try:
        cursor = db.execute(sql + ' LIMIT ? OFFSET ?', params + [AUDIT_PAGE_SIZE + 1, page * AUDIT_PAGE_SIZE])
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    finally:
        db.close()
    return answer({'columns': columns, 'rows': rows[:AUDIT_PAGE_SIZE], 'page': page,
                   'page_size': AUDIT_PAGE_SIZE, 'has_more': len(rows) > AUDIT_PAGE_SIZE})


@app.get('/api/audit/export')
def audit_export(kind: str = Query('mismatches', pattern=AUDIT_KIND),
                 month: str = Query('', pattern=AUDIT_MONTH),
                 contract: str = Query('', max_length=100)):
    """The whole list as CSV, streamed: it can hold millions of rows."""
    db = audit_db()
    if db is None:
        return answer({'error': 'No audit results yet'}, 404)
    sql, params = list_query(kind, month or None, contract.strip() or None)

    def lines():
        try:
            cursor = db.execute(sql, params)
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            buffer.write('\ufeff')  # Excel opens it as UTF-8
            writer.writerow([column[0] for column in cursor.description])
            for row in cursor:
                writer.writerow(row)
                if buffer.tell() > 1 << 16:
                    yield buffer.getvalue()
                    buffer.seek(0)
                    buffer.truncate()
            yield buffer.getvalue()
        finally:
            db.close()

    name = f"{kind}_{month or 'all-months'}.csv"
    return StreamingResponse(lines(), media_type='text/csv; charset=utf-8',
                             headers={**NO_STORE, 'Content-Disposition': f'attachment; filename="{name}"'})


@app.get('/api/models')
def models():
    """The models a batch can be read with, the default first, each with what
    its servers say now: slots (llama.cpp), or none reported (vLLM)."""
    listed = []
    for choice in MODEL_BACKENDS.values():
        listed.append(dict(key=choice.key, label=choice.label, model=choice.model,
                           urls=[server.url for server in choice.servers],
                           slots=server_slots(choice.servers), workers=choice.workers,
                           fixed_workers=choice.fixed_workers))
    return answer(listed)


@app.get('/api/limits')
def limits():
    return answer({'photos': MAX_PHOTOS, 'mb': MAX_UPLOAD_MB})


@app.post('/api/upload', status_code=202)
def upload(request: Request, images: list[UploadFile] = File(...), model: str | None = Form(None)):
    length = int(request.headers.get('content-length', 0))
    if not 0 < length <= MAX_UPLOAD_BYTES:
        return answer({'error': f'Choose a batch smaller than {MAX_UPLOAD_MB} MB.'}, 413)
    if not 1 <= len(images) <= MAX_PHOTOS:
        return answer({'error': f'Choose between 1 and {MAX_PHOTOS} images.'}, 400)
    try:
        return answer({'id': create_job(((image.filename or 'image', image.file) for image in images),
                                        model)}, 202)
    except ValueError as exc:
        return answer({'error': str(exc)}, 400)
    except Exception as exc:
        return answer({'error': str(exc)}, 500)


@app.get('/images/{job_id}/{image_name}')
def original_photo(job_id: str = PathParam(pattern=JOB_ID),
                   image_name: str = PathParam(pattern=IMAGE_NAME)):
    image = RUNS_DIR / job_id / image_name
    if not image.is_file():
        return answer({'error': 'Not found'}, 404)
    return Response(image.read_bytes(), media_type=PHOTO_TYPES[image.suffix.lower()], headers=IMMUTABLE)


@app.delete('/api/jobs/{job_id}/images/{image_name}')
def remove_image(job_id: str = PathParam(pattern=JOB_ID),
                 image_name: str = PathParam(pattern=IMAGE_NAME)):
    try:
        delete_image(job_id, image_name)
        return answer({'ok': True})
    except (OSError, ValueError) as exc:
        return answer({'error': str(exc)}, 404)


@app.post('/api/jobs/{job_id}/retry-failed', status_code=202)
def retry_failed(job_id: str = PathParam(pattern=JOB_ID)):
    """Read the batch's failed photos again, and only those, with the batch's model."""
    with LOCK:
        try:
            job = read_job(job_id)
        except (OSError, ValueError):
            return answer({'error': 'Upload not found'}, 404)
        if job['status'] in ('queued', 'processing'):
            return answer({'error': 'This batch is still being read.'}, 409)
        failed = [item for item in job['images'] if item.get('status') == 'error']
        for item in failed:
            item.update(status='queued')
            item.pop('error', None)
        if failed:
            job['status'] = 'queued'
            save_job(job)
            WORKER.submit(process_job, job)
    return answer({'id': job_id, 'retrying': len(failed)}, 202)


@app.delete('/api/jobs/{job_id}')
def remove_job(job_id: str = PathParam(pattern=JOB_ID)):
    try:
        delete_job(job_id)
        return answer({'ok': True})
    except (OSError, ValueError) as exc:
        return answer({'error': str(exc)}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.environ.get('FOTOMATES_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('FOTOMATES_PORT', '8081')))
    args = parser.parse_args()
    print(f'FotoMates: http://{args.host}:{args.port}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
