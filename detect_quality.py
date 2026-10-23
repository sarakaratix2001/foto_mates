from __future__ import annotations

import cv2
import numpy as np
import os
import sys
import threading

if not hasattr(cv2, "ml") or not hasattr(cv2.ml, "EM_create"):
    raise ImportError(
        f"OpenCV {cv2.__version__} provides no cv2.ml, so the GMM contrast "
        "normalisation that every reading is built on cannot run. OpenCV 5.0 "
        "removed cv2.ml; install opencv-python>=4.10,<5. Refusing to start "
        "rather than read digits from uncorrected pixels."
    )
def _setting(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


# What makes a photo blurry, low-contrast (grey std), dark or bright (grey mean, 0..255).
BLUR_THRESHOLD = _setting('FOTOMATES_BLUR_THRESHOLD', '120')
LOW_CONTRAST_THRESHOLD = _setting('FOTOMATES_LOW_CONTRAST_THRESHOLD', '40')
DARK_THRESHOLD = _setting('FOTOMATES_DARK_THRESHOLD', '85')
BRIGHT_THRESHOLD = _setting('FOTOMATES_BRIGHT_THRESHOLD', '200')

# Noisy: sharpness above this, that falls below this share after a median filter.
NOISE_ABSOLUTE_MIN_VARIANCE = _setting('FOTOMATES_NOISE_MIN_VARIANCE', '800')
NOISE_VARIANCE_RATIO_THRESHOLD = _setting('FOTOMATES_NOISE_RATIO', '0.35')

# Glare: pixels at least this bright (0..255) and at most this saturated, covering
# this share of the photo, or one patch of them covering this share.
GLARE_MIN_VALUE = _setting('FOTOMATES_GLARE_MIN_VALUE', '235')
GLARE_MAX_SATURATION = _setting('FOTOMATES_GLARE_MAX_SATURATION', '55')
GLARE_FRACTION = _setting('FOTOMATES_GLARE_FRACTION', '0.025')
GLARE_PATCH_FRACTION = _setting('FOTOMATES_GLARE_PATCH_FRACTION', '0.012')


_GMM_SAMPLE_CAP = 5_000
_GMM_TERM_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.1)
_GMM_LOCK = threading.Lock()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raw_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    denoised_blur = float(cv2.Laplacian(cv2.medianBlur(gray, 3), cv2.CV_64F).var())
    noise_ratio = denoised_blur / raw_blur if raw_blur > 1.0 else 1.0
    is_noisy = raw_blur > NOISE_ABSOLUTE_MIN_VARIANCE and noise_ratio < NOISE_VARIANCE_RATIO_THRESHOLD
    blur = denoised_blur if is_noisy else raw_blur
    histogram = np.bincount(gray.reshape(-1), minlength=256)
    extreme_fraction = float(histogram[:16].sum() + histogram[240:].sum()) / gray.size
    midtone_fraction = float(histogram[32:224].sum()) / gray.size
    is_pixelated = (
        brightness > 185.0 and extreme_fraction > 0.78
        and midtone_fraction < 0.14 and raw_blur > 500.0
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    glare_mask = ((hsv[:, :, 2] >= GLARE_MIN_VALUE) & (hsv[:, :, 1] <= GLARE_MAX_SATURATION)).astype(np.uint8)
    glare_fraction = float(glare_mask.mean())
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(glare_mask, 8)
    largest_glare = (
        float(stats[1:, cv2.CC_STAT_AREA].max()) / gray.size
        if n_labels > 1 else 0.0
    )
    has_glare = glare_fraction >= GLARE_FRACTION or largest_glare >= GLARE_PATCH_FRACTION

    return {
        "blur_variance": round(blur, 1),
        "contrast": round(contrast, 1),
        "brightness": round(brightness, 1),
        "is_blurry": blur < BLUR_THRESHOLD,
        "is_low_contrast": contrast < LOW_CONTRAST_THRESHOLD,
        "is_dark": brightness < DARK_THRESHOLD,
        "is_bright": brightness > BRIGHT_THRESHOLD,
        "is_noisy": is_noisy,
        "is_pixelated": is_pixelated,
        "has_glare": has_glare,
        "glare_fraction": round(glare_fraction, 3),
        "extreme_pixel_fraction": round(extreme_fraction, 3),
    }


def _glare_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= GLARE_MIN_VALUE) & (hsv[:, :, 1] <= GLARE_MAX_SATURATION)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def _corrected_channel(channel, bgr, report, force_gmm):
    """The correction chain, applied to one channel of light."""
    if report["is_noisy"]:
        channel = cv2.fastNlMeansDenoising(channel, None, 7, 7, 21)
    elif report.get("is_pixelated"):
        channel = cv2.GaussianBlur(channel, (3, 3), 0.65)

    if report.get("has_glare"):
        mask = _glare_mask(bgr)
        if 0 < cv2.countNonZero(mask) < channel.size * 0.22:
            channel = cv2.inpaint(channel, mask, 3, cv2.INPAINT_TELEA)

    if (force_gmm or report["is_low_contrast"] or report["is_dark"]
            or report["is_bright"] or report.get("has_glare")):
        channel = _gmm_contrast_normalize(channel)

    if report["is_dark"]:
        channel = cv2.LUT(channel, np.array([((i / 255.0) ** 0.72) * 255 for i in range(256)], dtype=np.uint8))
    elif report["is_bright"]:
        channel = cv2.LUT(channel, np.array([((i / 255.0) ** 1.35) * 255 for i in range(256)], dtype=np.uint8))

    channel = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(channel)

    severity = _clip01(1.0 - report["blur_variance"] / BLUR_THRESHOLD)
    amount = 0.5 + 2.5 * severity
    blurred = cv2.GaussianBlur(channel, (0, 0), 1.0 + severity)
    return cv2.addWeighted(channel, 1.0 + amount, blurred, -amount, 0)


def adjust_light(bgr: np.ndarray) -> np.ndarray:
    """Brightness and contrast set from this photo's own light, colours kept.

    Each region is brightened or darkened by how far its local brightness sits
    from mid-grey, so a dark LCD next to a white label is lifted while the label
    is left alone. The lightness is then stretched so its 1st..99th percentiles
    span the full range. A well-exposed, evenly lit photo changes little.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32) / 255
    # About the height of a meter display, so the display is judged by itself.
    sigma = max(lightness.shape) / 30
    local = np.clip(cv2.GaussianBlur(lightness, (0, 0), sigma), 0.03, 0.97)
    # Per pixel gamma that maps the local mean to mid-grey, limited so noise in
    # near-black corners is not blown up.
    gamma = np.clip(np.log(0.5) / np.log(local), 0.35, 1.6)
    lifted = lightness ** gamma
    lo, hi = np.percentile(lifted, (1, 99))
    if hi - lo >= 0.07:
        lifted = np.clip((lifted - lo) / (hi - lo), 0, 1)
    lab[:, :, 0] = (lifted * 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_for_ocr(
    bgr: np.ndarray, report: dict | None = None, *, force_gmm: bool = False,
) -> np.ndarray:
    report = report or assess(bgr)
    gray = _corrected_channel(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr, report, force_gmm)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# Which correction enhance() applies: 'light' sets brightness and contrast from
# the photo's own light and keeps colour (adjust_light); 'gmm' is the grayscale
# denoise / glare inpaint / contrast / sharpen chain (enhance_for_ocr).
ENHANCER = os.environ.get('FOTOMATES_ENHANCER', 'light').strip().lower()
if ENHANCER not in ('light', 'gmm'):
    raise ValueError(f"FOTOMATES_ENHANCER must be 'light' or 'gmm', not {ENHANCER!r}")


def enhance(bgr: np.ndarray, report: dict | None = None) -> np.ndarray:
    """The photo corrected by the chosen ENHANCER."""
    if ENHANCER == 'gmm':
        return enhance_for_ocr(bgr, report)
    return adjust_light(bgr)



_GMM_WARNED = set()


def _warn_gmm(where: str, error: Exception) -> None:
    reason = f'{where}: {type(error).__name__}: {error}'
    if reason not in _GMM_WARNED:
        _GMM_WARNED.add(reason)
        print(f'WARNING: GMM contrast normalisation unavailable - {reason}',
              file=sys.stderr, flush=True)


def _gmm_contrast_normalize(gray: np.ndarray) -> np.ndarray:
    if int(gray.max()) - int(gray.min()) < 18:
        return gray.copy()
    samples = gray.reshape(-1, 1)
    samples = np.ascontiguousarray(samples[::max(1, len(samples) // _GMM_SAMPLE_CAP)], dtype=np.float32)
    if len(samples) < 100:
        return gray
    try:
        em = cv2.ml.EM_create()
        em.setClustersNumber(3)
        em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
        em.setTermCriteria(_GMM_TERM_CRITERIA)
        with _GMM_LOCK:
            cv2.setRNGSeed(0)
            ok, _, _, _ = em.trainEM(samples)
        if not ok:
            return gray
        means = em.getMeans().reshape(-1)
        covariances = np.asarray([cov.reshape(-1)[0] for cov in em.getCovs()])
        deviations = np.sqrt(np.maximum(covariances, 1.0))
        order = np.argsort(means)
        lo, hi = float(means[order[0]] - deviations[order[0]]), float(means[order[-1]] + deviations[order[-1]])
        if hi - lo < 18.0:
            return gray
        lut = np.clip((np.arange(256, dtype=np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        return cv2.LUT(gray, lut)
    except cv2.error as error:
        _warn_gmm('_gmm_contrast_normalize', error)
        return gray
