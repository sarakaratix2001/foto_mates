"""Local upload UI for FotoMates, served by FastAPI. Run: python web_app.py"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Path as PathParam, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from extract_readings import ocr_image, extract_reading
from detect_quality import enhance_for_ocr

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ.get('FOTOMATES_RUNS', BASE_DIR / 'web_runs'))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
WORKER = ThreadPoolExecutor(max_workers=1)
IMAGE_WORKERS = 4
LOCK = threading.RLock()

JOB_ID = r'^[a-f0-9]{32}$'
IMAGE_NAME = r'^[a-f0-9]{32}\.png$'
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


def process_image(job, item):
    item['status'] = 'processing'
    save_job(job)
    try:
        image_path = RUNS_DIR / job['id'] / item['image']
        result = ocr_image(image_path)
        reading = extract_reading(result.text)
        item.update(reading=reading, raw_text=result.text,
                    digits=[c for c in reading if c.isdigit()],
                    avg_confidence=result.avg_confidence,
                    min_confidence=result.min_confidence,
                    min_margin=result.min_margin,
                    evidence_source=result.evidence_source,
                    status='done' if reading else 'no_digits')
    except Exception as exc:
        item.update(status='error', error=str(exc))
    save_job(job)


def process_job(job):
    job['status'] = 'processing'
    save_job(job)
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        list(pool.map(lambda item: process_image(job, item), job['images']))
    job['status'] = 'done'
    save_job(job)


def create_job(files):
    decoded = []
    for filename, payload in files:
        if Path(filename).suffix.lower() not in ALLOWED:
            raise ValueError(f'Unsupported image type: {filename}')
        picture = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) if payload else None
        if picture is None or min(picture.shape[:2]) < 3:
            raise ValueError(f'Cannot read image: {filename}')
        ok, encoded = cv2.imencode('.png', picture)
        if not ok:
            raise ValueError(f'Cannot prepare image: {filename}')
        decoded.append((filename, encoded.tobytes()))
    job = dict(id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
               status='queued', images=[])
    root = RUNS_DIR / job['id']
    root.mkdir(parents=True)
    for filename, payload in decoded:
        image = uuid.uuid4().hex + '.png'
        (root / image).write_bytes(payload)
        job['images'].append(dict(filename=filename, image=image, status='queued'))
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
        (root / image_name).unlink(missing_ok=True)
        for cache_dir in root.glob('enhanced_v*'):
            (cache_dir / image_name).unlink(missing_ok=True)
        job['images'] = remaining
        save_job(job)
        return job


def delete_job(job_id):
    with LOCK:
        root = RUNS_DIR / job_id
        if not (root / 'results.json').is_file():
            raise ValueError('Upload not found')
        shutil.rmtree(root, ignore_errors=True)


def mark_interrupted_jobs():
    for result in RUNS_DIR.glob('*/results.json'):
        job = read_job(result.parent.name)
        if job['status'] in ('queued', 'processing'):
            job['status'] = 'interrupted'
            for item in job['images']:
                if item['status'] in ('queued', 'processing'):
                    item.update(status='error', error='Server restarted. Re-upload to try again.')
            save_job(job)


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mark_interrupted_jobs()
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
            jobs.append(read_job(result.parent.name))
        except (OSError, ValueError):
            continue
    return answer(sorted(jobs, key=lambda j: j['created_at'], reverse=True))


@app.get('/api/jobs/{job_id}')
def show_job(job_id: str = PathParam(pattern=JOB_ID)):
    try:
        return answer(read_job(job_id))
    except (OSError, ValueError):
        return answer({'error': 'Upload not found'}, 404)


@app.post('/api/upload', status_code=202)
def upload(request: Request, images: list[UploadFile] = File(...)):
    length = int(request.headers.get('content-length', 0))
    if not 0 < length <= MAX_UPLOAD_BYTES:
        return answer({'error': 'Choose a batch smaller than 50 MB.'}, 413)
    if not 1 <= len(images) <= 30:
        return answer({'error': 'Choose between 1 and 30 images.'}, 400)
    try:
        files = [(image.filename or 'image', image.file.read()) for image in images]
        return answer({'id': create_job(files)}, 202)
    except ValueError as exc:
        return answer({'error': str(exc)}, 400)
    except Exception as exc:
        return answer({'error': str(exc)}, 500)


@app.get('/enhanced/{job_id}/{image_name}')
def enhanced_photo(job_id: str = PathParam(pattern=JOB_ID),
                   image_name: str = PathParam(pattern=IMAGE_NAME)):
    source = RUNS_DIR / job_id / image_name
    if not source.is_file():
        return answer({'error': 'Original photo not found'}, 404)
    cached = source.parent / 'enhanced_v10' / source.name
    try:
        if not cached.is_file():
            original = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError('Could not decode photo for enhancement')
            enhanced = enhance_for_ocr(original, force_gmm=True)
            cached.parent.mkdir(exist_ok=True)
            with LOCK:
                if not cv2.imwrite(str(cached), enhanced):
                    raise ValueError('Could not save enhanced photo')
        return Response(cached.read_bytes(), media_type='image/png', headers=IMMUTABLE)
    except (ValueError, OSError, cv2.error):
        return answer({'error': 'Could not enhance this photo'}, 500)


@app.get('/images/{job_id}/{image_name}')
def original_photo(job_id: str = PathParam(pattern=JOB_ID),
                   image_name: str = PathParam(pattern=IMAGE_NAME)):
    image = RUNS_DIR / job_id / image_name
    if not image.is_file():
        return answer({'error': 'Not found'}, 404)
    return Response(image.read_bytes(), media_type='image/png', headers=IMMUTABLE)


@app.delete('/api/jobs/{job_id}/images/{image_name}')
def remove_image(job_id: str = PathParam(pattern=JOB_ID),
                 image_name: str = PathParam(pattern=IMAGE_NAME)):
    try:
        delete_image(job_id, image_name)
        return answer({'ok': True})
    except (OSError, ValueError) as exc:
        return answer({'error': str(exc)}, 404)


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
