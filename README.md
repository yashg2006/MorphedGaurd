# MorphGuard

MorphGuard is an image analysis and fake detection tool. I built this mainly to analyze digital certificates and detect if they've been tampered with or morphed. It was also designed to showcase some core OS concepts handling the backend processing.

It checks things under the hood like EXIF data, Error Level Analysis (ELA), noise differences, and uses a CNN to flag suspicious changes. To make the processing smooth and concurrent, it implements OS-level components like an IPC manager, sync manager, and deadlock handler.

## Tech stack
- Python / Flask for the backend server
- TensorFlow, OpenCV, and Scikit-image for the deep learning / image processing stuff
- SQLAlchemy with SQLite locally or PostgreSQL in deployment
- HTML/CSS/JS for the frontend interface

## How to run it locally

1. Clone the repo
   ```bash
   git clone https://github.com/yashg2006/MorphedGaurd.git
   cd MorphedGaurd
   ```

2. Install the dependencies (using a virtual env is recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server
   ```bash
   python app.py
   ```

4. Go to `http://localhost:5000` in your browser to use the interface.

## How it works

When you upload an image (like a digital certificate), the backend spins up a few processes that:
- Check for missing, inconsistent, or stripped EXIF metadata
- Run Error Level Analysis to spot pasted or altered patches
- Analyze image noise variance
- Pass the file to a CNN to classify if it looks morphed

It combines these heuristics to calculate an overall "suspicion score". If the score crosses a certain threshold, the system flags the file as a potential fake.

## Deployment

This repo includes `render.yaml` and a `Procfile` for deployment. The Render
Blueprint provisions:

- a Flask web service started with Gunicorn
- a managed PostgreSQL database injected as `DATABASE_URL`
- a `/healthz` health check endpoint

Before deploying, set `SECRET_KEY` in the Render Dashboard. See
`DEPLOYMENT_REVIEW.md` for the deployment checklist, security notes, and model
readiness review.

## Model note

The app only uses CNN predictions when trained weights are available at
`models/cnn_weights.h5` or `MODEL_WEIGHTS_PATH`. Without trained weights, it
uses a transparent heuristic fallback and reports that status in `/healthz`.
