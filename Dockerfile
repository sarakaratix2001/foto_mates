
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py web_index.html ./


ENV FOTOMATES_RUNS=/data/web_runs \
    FOTOMATES_HOST=0.0.0.0 \
    FOTOMATES_PORT=8081
RUN mkdir -p /data/web_runs


RUN useradd --create-home --uid 10001 fotomates && chown -R fotomates /data
USER fotomates

EXPOSE 8081


HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ['FOTOMATES_PORT']+'/api/jobs',timeout=4)"

CMD ["python", "web_app.py"]
