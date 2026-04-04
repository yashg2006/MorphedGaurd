"""
Copy-Move Forgery Detection — detects duplicated (copy-pasted)
regions within an image using block-based DCT matching.
"""
import numpy as np
import cv2
import config


def perform_copy_move_detection(image_path: str) -> dict:
    """
    Detect copy-move forgery using block matching with DCT features.

    Returns:
        dict with keys: result_image (numpy array), score (0-100),
                        matched_pairs (int), details (str)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"result_image": None, "score": 0, "matched_pairs": 0,
                "details": "Failed to load image."}

    # ── Resize for performance ────────────────────────
    h, w = img.shape[:2]
    if max(h, w) > config.IMAGE_MAX_DIM:
        scale = config.IMAGE_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    result_img = img.copy()
    h, w = gray.shape

    block_size = 32
    stride = 16
    min_distance = 80  # Minimum distance between matching blocks

    # ── Extract blocks and compute DCT features ──────
    blocks = []
    positions = []

    for y in range(0, h - block_size, stride):
        for x in range(0, w - block_size, stride):
            block = gray[y:y + block_size, x:x + block_size]
            dct_block = cv2.dct(block)
            # Use top-left 16x16 DCT coefficients as feature
            feature = dct_block[:16, :16].flatten()
            blocks.append(feature)
            positions.append((x, y))

    if len(blocks) < 2:
        return {"result_image": result_img, "score": 0, "matched_pairs": 0,
                "details": "Image too small for copy-move analysis."}

    blocks = np.array(blocks)
    positions = np.array(positions)

    # ── Lexicographic sort for efficient matching ─────
    indices = np.lexsort(blocks.T)
    sorted_blocks = blocks[indices]
    sorted_positions = positions[indices]

    # ── Find matching blocks ──────────────────────────
    matched_pairs = 0
    match_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(sorted_blocks) - 1):
        diff = np.sum(np.abs(sorted_blocks[i] - sorted_blocks[i + 1]))

        if diff < 15:  # Tight similarity threshold to avoid false positives
            p1 = sorted_positions[i]
            p2 = sorted_positions[i + 1]
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            if distance > min_distance:
                matched_pairs += 1
                # Mark matched regions
                cv2.rectangle(match_mask,
                              (p1[0], p1[1]),
                              (p1[0] + block_size, p1[1] + block_size), 255, -1)
                cv2.rectangle(match_mask,
                              (p2[0], p2[1]),
                              (p2[0] + block_size, p2[1] + block_size), 255, -1)

    # ── Highlight matched regions on result image ─────
    if matched_pairs > 0:
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        match_mask = cv2.morphologyEx(match_mask, cv2.MORPH_CLOSE, kernel)
        match_mask = cv2.morphologyEx(match_mask, cv2.MORPH_OPEN, kernel)

        # Draw contours
        contours, _ = cv2.findContours(match_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)

        # Create overlay
        overlay = result_img.copy()
        overlay[match_mask > 0] = [0, 0, 255]
        result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)

    # ── Score based on matched pairs ──────────────────
    # Require significant number of matches (100+) before flagging as suspicious
    # Documents naturally have some repeating patterns (borders, backgrounds)
    score = min(100, int((matched_pairs / 100.0) * 100))

    return {
        "result_image": result_img,
        "score": score,
        "matched_pairs": matched_pairs,
        "details": f"Found {matched_pairs} matching block pairs. "
                   f"{'Copy-move forgery likely detected!' if score > config.SUSPICION_THRESHOLD else 'No significant copy-move patterns found.'}"
    }
