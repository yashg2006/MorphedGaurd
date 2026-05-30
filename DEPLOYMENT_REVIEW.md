# MorphGuard Deployment Review

## Current readiness

MorphGuard is now prepared for a Git-backed Render deployment with a managed
PostgreSQL database. The app reads `DATABASE_URL` from the environment and falls
back to local SQLite for development. Analysis status and result payloads are
stored with parameterized SQL, so polling still works across process restarts or
different web workers.

## Image detection model review

- The CNN path is not production-grade until a trained `cnn_weights.h5` file is
  supplied through `MODEL_WEIGHTS_PATH` or placed in `models/cnn_weights.h5`.
- If no trained weights are present, the app now reports a heuristic fallback
  instead of using random CNN weights. This avoids fake confidence from an
  untrained model.
- The final score is still an ensemble of ELA, noise variance, EXIF checks,
  copy-move matching, and CNN or heuristic output. Treat it as triage, not a
  forensic guarantee.
- Before production use, validate against a labeled dataset of real, edited,
  and morphed certificates. Track false positives separately for screenshots,
  generated certificates, scans, and camera photos.

## Security changes

- Uploads are image-only: JPG, PNG, BMP, and TIFF. PDF uploads were removed
  because the current pipeline does not safely parse PDFs.
- Uploaded files are validated by extension, MIME type, and Pillow image
  verification before analysis.
- Task IDs and result-image requests are constrained to expected characters.
- Browser security headers are applied on every response.
- Result rendering escapes server-controlled text before inserting HTML,
  closing an uploaded-filename/result-detail XSS path.
- Flask debug stays disabled and secrets are expected through environment
  variables.

## Database posture

- Use Render PostgreSQL in deployment through the `DATABASE_URL` injected by
  `render.yaml`.
- Do not commit database credentials. `SECRET_KEY` is marked `sync: false` and
  should be filled in the Render Dashboard.
- Current storage keeps result JSON, including embedded analysis images, in the
  database for deploy reliability. For heavier usage, move original uploads and
  generated images to private object storage and keep only signed references in
  PostgreSQL.
- Add a retention policy before handling real identity documents. Uploaded
  certificate images and EXIF metadata can be sensitive.

## Render deployment steps

1. Commit and push this repository to GitHub, GitLab, or Bitbucket.
2. Open a Render Blueprint from the repository:
   `https://dashboard.render.com/blueprint/new`
3. Select the repo containing `render.yaml`.
4. Fill `SECRET_KEY` with a strong random value.
5. Apply the Blueprint and wait for the web service and PostgreSQL database.
6. Verify `https://<your-service>.onrender.com/healthz` returns `status: ok`.

## Remaining production work

- Add authentication before accepting sensitive documents from real users.
- Add rate limiting for `/api/analyze`, `/api/batch`, and `/api/os-stats`.
- Add a background job queue for long analysis tasks if traffic grows.
- Add model evaluation tests and publish a model card with dataset, metrics,
  known failure modes, and threshold rationale.
