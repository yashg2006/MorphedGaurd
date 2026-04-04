"""
Configuration constants for the Morphing & Fake Detection System.
"""
import os

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Upload limits ───────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "pdf"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# ─── Process / Thread limits ────────────────────────────
MAX_PROCESSES = 4          # Multiprocessing pool size
MAX_THREADS = 8            # Thread pool size per stage
SEMAPHORE_LIMIT = 3        # Max concurrent image analyses
LOCK_TIMEOUT = 10          # Seconds before deadlock detection

# ─── Memory / Cache ─────────────────────────────────────
CACHE_MAX_SIZE = 32        # LRU cache entries
BUFFER_POOL_SIZE = 8       # Pre-allocated image buffers
IMAGE_MAX_DIM = 1024       # Resize large images for processing

# ─── ELA settings ────────────────────────────────────────
ELA_QUALITY = 90           # JPEG re-save quality for ELA
ELA_AMPLIFICATION = 10     # Difference amplification factor

# ─── Detection thresholds ────────────────────────────────
SUSPICION_THRESHOLD = 60   # Score above which image is suspicious
FAKE_THRESHOLD = 80        # Score above which image is classified fake
