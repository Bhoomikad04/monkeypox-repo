# download_models.py
import os
import sys
import time
from urllib.parse import urlparse
import requests

# Use uploaded ChatGPT file paths as URLs (they will be auto-transformed)
MODELS = {
    "cnn_best.h5": "/mnt/data/cnn_best.h5",
    "resnet50_best.h5": "/mnt/data/resnet50_best.h5",
    "vgg16_best.h5": "/mnt/data/vgg16_best.h5",
}

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def download(url, out_path, max_retries=3, chunk_size=1<<20):
    for attempt in range(1, max_retries + 1):
        try:
            parsed = urlparse(url)

            # If it's a local path (no scheme), copy directly
            if parsed.scheme in ("", "file"):
                src = parsed.path if parsed.scheme == "file" else url
                if os.path.exists(src):
                    print(f"Copying local file {src} -> {out_path}")
                    with open(src, "rb") as fr, open(out_path, "wb") as fw:
                        while True:
                            chunk = fr.read(chunk_size)
                            if not chunk:
                                break
                            fw.write(chunk)
                    return True
                else:
                    raise FileNotFoundError(f"Local source not found: {src}")

            # If it's a URL, download it
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            return True

        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return False


def main():
    ok = True
    for fname, url in MODELS.items():
        out_path = os.path.join(OUT_DIR, fname)
        print(f"Processing: {fname}")

        if os.path.exists(out_path):
            print(f"{out_path} already exists — skipping.")
            continue

        if not download(url, out_path):
            ok = False

    if not ok:
        print("Model download failed.", file=sys.stderr)
        sys.exit(1)

    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
