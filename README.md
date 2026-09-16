# FotoMates

Start the local upload frontend:

```powershell
python -m pip install -r requirements.txt
python web_app.py
```

Open http://127.0.0.1:8081. Upload up to 30 meter photos per batch (50 MB total).

Or run it in Docker. Put `FOTOMATES_OCR_URL`, `FOTOMATES_OCR_MODEL` and
`FOTOMATES_OCR_API_KEY` in a `.env` file next to `docker-compose.yml`, then:

```powershell
docker compose up -d --build
```

That serves the same frontend on http://127.0.0.1:8081. If something already
holds that port, publish another one — the container still listens on 8081
internally:

```powershell
$env:FOTOMATES_PUBLISH_PORT='8082'; docker compose up -d --build
docker compose logs -f          # follow the server
docker compose down             # stop it
```

Recognition runs on a separate OCR server reached over HTTP, so the image needs
no GPU, CUDA, or model weights. Uploads and their results live in the
`fotomates-data` volume and survive a rebuild; `docker compose down -v` discards
them. Compose publishes the port on `127.0.0.1` only — drop that prefix in
`docker-compose.yml` to expose it on the network.

`FOTOMATES_OCR_URL` must keep its `/v1` suffix, and inside a container
`localhost` means the container itself: use `host.docker.internal` to reach an
OCR server on your own machine.

## How a photo is read

1. `detect_quality.assess` measures blur, contrast, brightness, noise and glare.
   `enhance_for_ocr` then corrects the photo: denoising, glare inpainting, GMM
   contrast normalisation, gamma, CLAHE and sharpening, as the measurements call
   for.
2. The model receives the enhanced image, which it reads digits and separators
   from, and the original photo, which it uses only for colour and to tell the
   register apart from nearby text and symbols. Both are enlarged so the long
   side is at least 1600 px: on a small photo a decimal point is only a few
   pixels wide.
3. The model may reason briefly, then ends its reply with a `READING:` line, and
   only that line is used. If the line is missing, the model is asked once more.

The prompt places a decimal separator only where the register shows one. On a
digital display that is a dot or comma on the baseline between two digits of the
reading row. On a mechanical counter it comes before the first fractional wheel,
which is set apart by colour, a printed comma or dot, or a fraction label such as
0,1. Arrows, triangles, pins and other icons never count, even when they line up
with a gap between digits, and a reading with no separator is normal.

Every upload is read fresh; nothing is cached. The model runs with
`temperature=0, seed=0`, so an unchanged photo usually gets the same reading
again. Readings already in the upload history are not changed; re-upload a
photo to read it again.

The page shows the original and enhanced photos side by side. The enhanced
preview is the same image the model reads.

`detect_quality` refuses to import if `cv2.ml` is missing, rather than reading
digits from uncorrected pixels. OpenCV 5.0 removed `cv2.ml`, which is why
`requirements.txt` pins `opencv-python<5`.

Syntax check:

```powershell
python -m py_compile extract_readings.py detect_quality.py web_app.py
```

Recognition accuracy depends on the model and the photo.
