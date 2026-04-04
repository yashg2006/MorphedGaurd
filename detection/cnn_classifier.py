"""
CNN-based Image Classifier — classifies images as Original, Edited, or Morphed/Fake
using a lightweight Convolutional Neural Network.

NOTE: On first run, this builds a small CNN and uses random weights for demo.
For production, replace with a properly trained model.
"""
import numpy as np
import cv2
import os
import config

# Lazy-load TensorFlow to avoid slow startup
_model = None


def _build_model():
    """Build a lightweight CNN for three-class image classification."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Check for pre-trained weights
        weights_path = os.path.join(config.MODEL_DIR, "cnn_weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        return model

    except ImportError:
        return None


def _get_model():
    """Get or create the CNN model (singleton)."""
    global _model
    if _model is None:
        _model = _build_model()
    return _model


def perform_cnn_classification(image_path: str) -> dict:
    """
    Classify an image using the CNN model.

    Returns:
        dict with keys: label (str), confidence (dict), score (0-100), details (str)
    """
    classes = ["Original", "Edited", "Morphed/Fake"]

    model = _get_model()

    # ── Read and preprocess image ─────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return {
            "label": "Unknown", "confidence": {}, "score": 50,
            "details": "Failed to load image for CNN classification."
        }

    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    if model is not None:
        try:
            predictions = model.predict(img_batch, verbose=0)[0]
        except Exception:
            # Fallback to heuristic-based classification
            predictions = _heuristic_classify(image_path)
    else:
        # TensorFlow not available — fallback
        predictions = _heuristic_classify(image_path)

    # ── Build result ──────────────────────────────────
    predicted_class = int(np.argmax(predictions))
    label = classes[predicted_class]
    confidence = {
        classes[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(classes))
    }

    # Score: how suspicious (inverse of "Original" confidence)
    score = int(100 - confidence["Original"])

    return {
        "label": label,
        "confidence": confidence,
        "score": score,
        "details": f"CNN classification: {label} "
                   f"(Original: {confidence['Original']}%, "
                   f"Edited: {confidence['Edited']}%, "
                   f"Morphed: {confidence['Morphed/Fake']}%)"
    }


def _heuristic_classify(image_path: str) -> np.ndarray:
    """
    Fallback heuristic when TensorFlow is unavailable.
    Uses basic statistical features to estimate classification.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.array([0.33, 0.34, 0.33])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── Feature extraction ────────────────────────────
    # Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Noise level estimation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.std(gray - blurred)

    # Edge density
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Color channel correlation
    b, g, r = cv2.split(img)
    rg_corr = np.corrcoef(r.flatten()[:1000], g.flatten()[:1000])[0, 1]

    # ── Simple scoring heuristic ──────────────────────
    original_score = 0.5
    edited_score = 0.25
    morphed_score = 0.25

    # Low noise = might be processed, but digital certificates naturally have very low noise
    if noise < 3:
        edited_score += 0.05  # Reduced from 0.15 to avoid false positives on clean docs
        original_score -= 0.05

    # Very high edge density = potential artifacts
    if edge_density > 0.15:
        edited_score += 0.1
        original_score -= 0.1

    # Low color correlation = possible manipulation
    if abs(rg_corr) < 0.7:
        morphed_score += 0.1
        original_score -= 0.1

    # Normalize to probabilities
    total = original_score + edited_score + morphed_score
    return np.array([original_score / total, edited_score / total, morphed_score / total])
