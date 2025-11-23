#!/bin/bash
set -e

echo "== Preparing models directory =="
mkdir -p /app/models

echo "== Running model downloader =="
python3 /app/download_models.py || {
  echo "Model downloader failed — aborting."
  exit 1
}

echo "== Models ready. Starting Streamlit =="
exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
