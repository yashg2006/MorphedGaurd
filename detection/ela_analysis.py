"""
Error Level Analysis (ELA) — detects edited regions by re-saving
the image at a known JPEG quality and comparing pixel differences.
"""
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import config


def perform_ela(image_path: str) -> dict:
    """
    Perform Error Level Analysis on an image.

    Returns:
        dict with keys: ela_image (numpy array), score (0-100), details (str)
    """
    original = Image.open(image_path).convert("RGB")

    # ── Re-save at known quality ──────────────────────
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=config.ELA_QUALITY)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # ── Compute pixel-level difference ────────────────
    diff = ImageChops.difference(original, resaved)

    # ── Amplify differences ───────────────────────────
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff * config.ELA_AMPLIFICATION
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(scale)

    # ── Convert to numpy for heatmap ──────────────────
    ela_array = np.array(ela_image)
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # ── Compute suspicion score ───────────────────────
    mean_diff = np.mean(gray)
    std_diff = np.std(gray)
    # Use generous divisor to avoid false positives on documents/certificates
    # which naturally have high variance due to mixed content (text + graphics)
    ratio = std_diff / (mean_diff + 1e-8)
    score = min(100, int(min(std_diff / 75.0, ratio / 2.5) * 100))

    return {
        "ela_image": heatmap,
        "score": score,
        "mean_diff": round(float(mean_diff), 2),
        "std_diff": round(float(std_diff), 2),
        "details": f"ELA mean={mean_diff:.2f}, std={std_diff:.2f}. "
                   f"{'High variance suggests possible editing.' if score > config.SUSPICION_THRESHOLD else 'Low variance — consistent compression.'}"
    }
