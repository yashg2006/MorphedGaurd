"""
Morphing & Fake Detection System — Flask Web Server
====================================================
Main application that ties together detection engines, OS concepts,
and serves the web UI.
"""
import os
import sys
import uuid
import time
import json
import logging
import threading
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image

import config
from file_manager import FileManager
from detection.ela_analysis import perform_ela
from detection.noise_analysis import perform_noise_analysis
from detection.exif_analysis import perform_exif_analysis
from detection.copy_move_detect import perform_copy_move_detection
from detection.cnn_classifier import perform_cnn_classification

from os_concepts.thread_manager import ThreadManager
from os_concepts.ipc_manager import IPCManager
from os_concepts.sync_manager import SyncManager
from os_concepts.memory_manager import MemoryManager
from os_concepts.deadlock_handler import DeadlockHandler

# ── Logging Setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "system.log"),
                            mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("App")

# ── Flask App ─────────────────────────────────────────────
app = Flask(__name__,
            static_folder="static",
            template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# ── Initialize OS Concepts Managers ──────────────────────
file_mgr = FileManager()
thread_mgr = ThreadManager()
ipc_mgr = IPCManager()
sync_mgr = SyncManager()
memory_mgr = MemoryManager()
deadlock_handler = DeadlockHandler()

# Register resources for deadlock handler demonstration
deadlock_handler.register_resource("image_buffer", order=1)
deadlock_handler.register_resource("results_store", order=2)
deadlock_handler.register_resource("log_file", order=3)

# ── In-memory results store ──────────────────────────────
analysis_results = {}  # task_id -> results dict
results_lock = threading.Lock()


def encode_image_base64(img_array) -> str:
    """Convert a numpy image array to base64 string for JSON transport."""
    if img_array is None:
        return ""
    success, buffer = cv2.imencode('.png', img_array)
    if success:
        return base64.b64encode(buffer).decode('utf-8')
    return ""


def analyze_image(image_path: str, task_id: str) -> dict:
    """
    Run the full detection pipeline on a single image.

    Pipeline stages (using OS concepts):
    1. Preprocessing (Thread stage 1)
    2. Feature extraction — ELA, Noise, EXIF, Copy-Move in parallel (Thread stage 2)
    3. CNN Classification (Thread stage 3)
    4. Result aggregation with synchronization
    """
    start_time = time.time()
    logger.info(f"[Pipeline] Starting analysis for task {task_id}")

    # ── Update IPC shared memory ──────────────────────
    ipc_mgr.shared_memory.increment_active()
    ipc_mgr.task_queue.send({"task_id": task_id, "action": "analyze", "path": image_path})

    # ── Acquire semaphore slot (Synchronization) ──────
    if not sync_mgr.acquire_analysis_slot():
        return {"error": "System busy — semaphore timeout", "task_id": task_id}

    # ── Deadlock-safe resource acquisition ────────────
    resources_acquired = deadlock_handler.acquire_resources(
        ["image_buffer", "results_store", "log_file"]
    )
    if not resources_acquired:
        sync_mgr.release_analysis_slot()
        return {"error": "Resource acquisition failed", "task_id": task_id}

    try:
        # ── Stage 1: Preprocessing (Thread) ───────────
        def preprocess(path):
            """Resize and prepare image for analysis."""
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            h, w = img.shape[:2]
            if max(h, w) > config.IMAGE_MAX_DIM:
                scale = config.IMAGE_MAX_DIM / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            logger.info(f"[Preprocess] Image resized to {img.shape[1]}x{img.shape[0]}")
            return img

        preprocess_future = thread_mgr.submit_preprocessing(preprocess, image_path)
        preprocessed_img = preprocess_future.result(timeout=30)

        # ── Stage 2: Parallel Feature Extraction (Threads) ─
        extraction_tasks = [
            {"name": "ela", "func": perform_ela, "args": (image_path,)},
            {"name": "noise", "func": perform_noise_analysis, "args": (image_path,)},
            {"name": "exif", "func": perform_exif_analysis, "args": (image_path,)},
            {"name": "copy_move", "func": perform_copy_move_detection,
             "args": (image_path,)},
        ]
        extraction_results = thread_mgr.run_parallel_extraction(extraction_tasks)

        # ── Stage 3: CNN Classification (Thread) ──────
        cnn_future = thread_mgr.submit_detection(
            perform_cnn_classification, image_path
        )
        cnn_result = cnn_future.result(timeout=60)

        # ── Stage 4: Aggregate Results ────────────────
        ela_score = extraction_results.get("ela", {}).get("score", 0)
        noise_score = extraction_results.get("noise", {}).get("score", 0)
        exif_score = extraction_results.get("exif", {}).get("score", 0)
        copy_move_score = extraction_results.get("copy_move", {}).get("score", 0)
        cnn_score = cnn_result.get("score", 0)

        # Weighted average
        weights = {"ela": 0.25, "noise": 0.15, "exif": 0.20,
                   "copy_move": 0.15, "cnn": 0.25}
        overall_score = (
            ela_score * weights["ela"] +
            noise_score * weights["noise"] +
            exif_score * weights["exif"] +
            copy_move_score * weights["copy_move"] +
            cnn_score * weights["cnn"]
        )

        # Classification
        if overall_score >= config.FAKE_THRESHOLD:
            verdict = "FAKE"
        elif overall_score >= config.SUSPICION_THRESHOLD:
            verdict = "SUSPICIOUS"
        else:
            verdict = "REAL"

        elapsed = round(time.time() - start_time, 2)

        # ── Save result images ────────────────────────
        result_images = {}
        ela_img = extraction_results.get("ela", {}).get("ela_image")
        if ela_img is not None:
            file_mgr.save_result_image(ela_img, task_id, "ela")
            result_images["ela"] = encode_image_base64(ela_img)

        noise_img = extraction_results.get("noise", {}).get("noise_map")
        if noise_img is not None:
            file_mgr.save_result_image(noise_img, task_id, "noise")
            result_images["noise"] = encode_image_base64(noise_img)

        copy_move_img = extraction_results.get("copy_move", {}).get("result_image")
        if copy_move_img is not None:
            file_mgr.save_result_image(copy_move_img, task_id, "copy_move")
            result_images["copy_move"] = encode_image_base64(copy_move_img)

        # ── Check cache (Memory Management) ───────────
        cache_key = f"result_{task_id}"
        result = {
            "task_id": task_id,
            "verdict": verdict,
            "overall_score": round(overall_score, 1),
            "confidence": round(100 - overall_score, 1) if verdict == "REAL"
                         else round(overall_score, 1),
            "elapsed_seconds": elapsed,
            "analyses": {
                "ela": {
                    "score": ela_score,
                    "details": extraction_results.get("ela", {}).get("details", ""),
                    "image": result_images.get("ela", "")
                },
                "noise": {
                    "score": noise_score,
                    "details": extraction_results.get("noise", {}).get("details", ""),
                    "image": result_images.get("noise", "")
                },
                "exif": {
                    "score": exif_score,
                    "details": extraction_results.get("exif", {}).get("details", ""),
                    "metadata": extraction_results.get("exif", {}).get("metadata", {}),
                    "flags": extraction_results.get("exif", {}).get("flags", [])
                },
                "copy_move": {
                    "score": copy_move_score,
                    "details": extraction_results.get("copy_move", {}).get("details", ""),
                    "matched_pairs": extraction_results.get("copy_move", {}).get(
                        "matched_pairs", 0),
                    "image": result_images.get("copy_move", "")
                },
                "cnn": {
                    "score": cnn_score,
                    "label": cnn_result.get("label", "Unknown"),
                    "confidence": cnn_result.get("confidence", {}),
                    "details": cnn_result.get("details", "")
                }
            },
            "status": "complete",
            "timestamp": datetime.now().isoformat()
        }

        # ── Cache the result (Memory Management) ──────
        memory_mgr.cache.put(cache_key, result)

        # ── Write log (File System Management) ────────
        file_mgr.write_analysis_log(task_id, result)

        # ── Update shared results (with mutex) ────────
        with results_lock:
            analysis_results[task_id] = result

        # ── Send result via IPC queue ─────────────────
        ipc_mgr.result_queue.send({"task_id": task_id, "verdict": verdict})
        ipc_mgr.shared_memory.increment_completed()

        logger.info(f"[Pipeline] Task {task_id} complete: {verdict} "
                    f"(score={overall_score:.1f}, time={elapsed}s)")

        return result

    except Exception as e:
        logger.error(f"[Pipeline] Task {task_id} failed: {e}")
        ipc_mgr.shared_memory.increment_errors()
        error_result = {
            "task_id": task_id,
            "verdict": "ERROR",
            "overall_score": 0,
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
        with results_lock:
            analysis_results[task_id] = error_result
        return error_result

    finally:
        # ── Always release resources ──────────────────
        deadlock_handler.release_resources(
            ["image_buffer", "results_store", "log_file"]
        )
        sync_mgr.release_analysis_slot()
        ipc_mgr.shared_memory.decrement_active()


# ══════════════════════════════════════════════════════════
#  API Routes
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_single():
    """Upload and analyze a single image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file_mgr.is_allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Accepted: {config.ALLOWED_EXTENSIONS}"}), 400

    # Save upload
    filepath = file_mgr.save_upload(file, file.filename)
    task_id = str(uuid.uuid4())[:8]

    # Set initial status
    with results_lock:
        analysis_results[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }

    # Run analysis in a thread
    def run():
        analyze_image(filepath, task_id)

    thread = threading.Thread(target=run, name=f"Analysis-{task_id}")
    thread.start()

    return jsonify({"task_id": task_id, "status": "processing",
                    "message": f"Analysis started for {file.filename}"}), 202


@app.route("/api/batch", methods=["POST"])
def analyze_batch():
    """Upload and analyze multiple images."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    sync_mgr.reset_batch()
    batch_id = str(uuid.uuid4())[:8]
    task_ids = []

    for file in files:
        if file.filename and file_mgr.is_allowed_file(file.filename):
            filepath = file_mgr.save_upload(file, file.filename)
            task_id = f"{batch_id}_{str(uuid.uuid4())[:4]}"
            task_ids.append(task_id)

            with results_lock:
                analysis_results[task_id] = {
                    "task_id": task_id,
                    "status": "processing",
                    "filename": file.filename,
                    "batch_id": batch_id
                }

    # Run batch in threads
    def run_batch():
        threads = []
        for i, task_id in enumerate(task_ids):
            filepath = None
            with results_lock:
                info = analysis_results.get(task_id, {})

            file = files[i] if i < len(files) else None
            if file and file.filename:
                # File already saved above, get the path
                import glob
                pattern = os.path.join(config.UPLOAD_DIR, f"*_{file_mgr._sanitize_filename(file.filename)}")
                matches = glob.glob(pattern)
                if matches:
                    filepath = matches[-1]

            if filepath:
                t = threading.Thread(
                    target=analyze_image, args=(filepath, task_id),
                    name=f"Batch-{task_id}"
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=120)

        sync_mgr.signal_batch_complete()

    batch_thread = threading.Thread(target=run_batch, name=f"BatchManager-{batch_id}")
    batch_thread.start()

    return jsonify({
        "batch_id": batch_id,
        "task_ids": task_ids,
        "count": len(task_ids),
        "status": "processing"
    }), 202


@app.route("/api/status/<task_id>")
def get_status(task_id):
    """Get the status of an analysis task."""
    # Check cache first (Memory Management)
    cached = memory_mgr.cache.get(f"result_{task_id}")
    if cached:
        return jsonify(cached)

    with results_lock:
        result = analysis_results.get(task_id)

    if result is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify(result)


@app.route("/api/results/<task_id>")
def get_results(task_id):
    """Get the full results of an analysis task."""
    with results_lock:
        result = analysis_results.get(task_id)

    if result is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify(result)


@app.route("/api/os-stats")
def get_os_stats():
    """Get live OS concepts statistics."""
    stats = {
        "threads": thread_mgr.get_stats(),
        "ipc": ipc_mgr.get_stats(),
        "synchronization": sync_mgr.get_stats(),
        "memory": memory_mgr.get_stats(),
        "deadlock": deadlock_handler.get_stats(),
        "file_system": file_mgr.get_stats(),
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(stats)


@app.route("/api/result-image/<task_id>/<analysis_type>")
def get_result_image(task_id, analysis_type):
    """Serve a result image file."""
    filename = f"{task_id}_{analysis_type}.png"
    filepath = os.path.join(config.RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="image/png")
    return jsonify({"error": "Image not found"}), 404


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Morphing & Fake Detection System")
    logger.info("  Starting server on http://localhost:5000")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
