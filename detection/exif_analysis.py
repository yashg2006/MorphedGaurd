"""
EXIF Metadata Analysis — extracts and inspects image metadata
for inconsistencies indicating tampering.
"""
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# Known editing software signatures
EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "snapseed", "pixlr",
    "canva", "affinity", "paint.net", "corel", "capture one",
    "adobe", "luminar", "darktable", "rawtherapee", "fotor",
    "befunky", "picmonkey", "inpixio", "photoscape", "irfanview"
]


def perform_exif_analysis(image_path: str) -> dict:
    """
    Extract EXIF data and flag suspicious metadata.

    Returns:
        dict with keys: metadata (dict), flags (list), score (0-100), details (str)
    """
    flags = []
    metadata = {}
    score = 0

    try:
        img = Image.open(image_path)
    except Exception as e:
        return {
            "metadata": {}, "flags": [f"Cannot open image: {e}"],
            "score": 50, "details": "Image could not be opened for EXIF analysis."
        }

    # ── Extract EXIF data ─────────────────────────────
    exif_data = img._getexif() if hasattr(img, '_getexif') else None

    if exif_data is None:
        flags.append("NO_EXIF_DATA")
        score += 15  # Reduced from 25: digital certificates legitimately lack EXIF
        metadata["warning"] = "No EXIF data found — common for digital certificates or screenshots"
    else:
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            try:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                metadata[tag_name] = str(value)[:200]  # Truncate long values
            except Exception:
                metadata[tag_name] = "<unreadable>"

    # ── Check for editing software signatures ─────────
    software = metadata.get("Software", "").lower()
    processing_software = metadata.get("ProcessingSoftware", "").lower()
    combined = software + " " + processing_software

    for editor in EDITING_SOFTWARE:
        if editor in combined:
            flags.append(f"EDITING_SOFTWARE_DETECTED: {editor.title()}")
            score += 30
            break

    # ── Check image dimensions vs EXIF dimensions ────
    actual_w, actual_h = img.size
    exif_w = metadata.get("ExifImageWidth", metadata.get("ImageWidth"))
    exif_h = metadata.get("ExifImageHeight", metadata.get("ImageLength"))

    if exif_w and exif_h:
        try:
            ew, eh = int(exif_w), int(exif_h)
            if (ew != actual_w or eh != actual_h):
                flags.append(f"SIZE_MISMATCH: EXIF({ew}x{eh}) vs Actual({actual_w}x{actual_h})")
                score += 20
        except (ValueError, TypeError):
            pass

    # ── Check for missing camera info ─────────────────
    if exif_data and not metadata.get("Make") and not metadata.get("Model"):
        flags.append("NO_CAMERA_INFO")
        score += 5  # Reduced from 10: digital certificates don't have camera models

    # ── Check thumbnail presence ──────────────────────
    if exif_data and not metadata.get("JPEGThumbnail") and not metadata.get("ThumbnailImage"):
        pass  # Not always suspicious

    # ── File format check ─────────────────────────────
    ext = os.path.splitext(image_path)[1].lower()
    img_format = (img.format or "").lower()
    if ext in [".jpg", ".jpeg"] and img_format not in ["jpeg", "jpg"]:
        flags.append(f"FORMAT_MISMATCH: Extension={ext}, Actual={img_format}")
        score += 15

    score = min(100, score)

    details_parts = []
    if flags:
        details_parts.append(f"Found {len(flags)} suspicious flag(s): {', '.join(flags)}.")
    else:
        details_parts.append("No suspicious metadata flags detected.")

    return {
        "metadata": metadata,
        "flags": flags,
        "score": score,
        "image_size": f"{actual_w}x{actual_h}",
        "format": img_format,
        "details": " ".join(details_parts)
    }
