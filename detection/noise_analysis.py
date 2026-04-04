"""
Noise Analysis — examines noise patterns across the image.
Edited regions often have different noise characteristics.
"""
import numpy as np
import cv2
import config


def perform_noise_analysis(image_path: str) -> dict:
    """
    Analyze noise patterns to detect inconsistencies.

    Returns:
        dict with keys: noise_map (numpy array), score (0-100), details (str)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"noise_map": None, "score": 0, "details": "Failed to load image."}

    # ── Resize for consistent analysis ────────────────
    h, w = img.shape[:2]
    if max(h, w) > config.IMAGE_MAX_DIM:
        scale = config.IMAGE_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── High-pass filter to extract noise residual ────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = np.abs(gray - blurred)

    # ── Block-based noise variance analysis ───────────
    block_size = 32
    h, w = noise_residual.shape
    variances = []
    variance_map = np.zeros_like(noise_residual)

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise_residual[y:y + block_size, x:x + block_size]
            var = np.var(block)
            variances.append(var)
            variance_map[y:y + block_size, x:x + block_size] = var

    if len(variances) == 0:
        return {"noise_map": None, "score": 0, "details": "Image too small for analysis."}

    # ── Normalize variance map for visualization ──────
    var_array = np.array(variances)
    global_var = np.var(var_array)
    mean_var = np.mean(var_array)

    norm_map = ((variance_map - variance_map.min()) /
                (variance_map.max() - variance_map.min() + 1e-8) * 255).astype(np.uint8)
    noise_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_HOT)

    # Use moderate scaling — documents/certificates with mixed content
    # (text, graphics, borders) naturally have higher CV (~1.5-2.0)
    cv_val = (np.std(var_array) / (mean_var + 1e-8))
    score = min(100, int(cv_val * 22))  # Reduced multiplier for documents

    return {
        "noise_map": noise_heatmap,
        "score": score,
        "global_variance": round(float(global_var), 4),
        "blocks_analyzed": len(variances),
        "details": f"Analyzed {len(variances)} blocks. Noise CV={cv_val:.3f}. "
                   f"{'Inconsistent noise pattern detected.' if score > config.SUSPICION_THRESHOLD else 'Noise pattern is consistent.'}"
    }
