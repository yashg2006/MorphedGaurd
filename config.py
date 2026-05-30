"""Configuration constants for the Morphing & Fake Detection System."""
import os

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MORPHGUARD_DATA_DIR", BASE_DIR)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(DATA_DIR, "results"))
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(DATA_DIR, "logs"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
MODEL_WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    os.path.join(MODEL_DIR, "cnn_weights.h5"),
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(DATA_DIR, 'morphguard.db')}",
)
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")

# ─── Upload limits ───────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/x-ms-bmp",
}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 36_000_000))

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
ALLOW_UNTRAINED_CNN = os.environ.get("ALLOW_UNTRAINED_CNN", "").lower() == "true"
