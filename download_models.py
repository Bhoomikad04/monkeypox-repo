# download_models.py
import os
import sys
import time
import requests

MODELS = {
    "cnn_best.h5": "https://drive.google.com/uc?export=download&id=1i1VzilyLMFSspXqtbRc_ch3LPYfxUusi",
    "resnet50_best.h5": "https://drive.google.com/uc?export=download&id=1gwsZYtvtI49VEGdnHNl3ff386bX4A9rj",
    "vgg16_best.h5": "https://drive.google.com/uc?export=download&id=1vcZ_5_wJHIENYGRDWoQH75rBVxuIO4rm",
}

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)


def download(url, out_path, max_retries=3, chunk_size=1024 * 1024):
    """Download url -> out_path with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            print(f"Saved: {out_path}")
            return True

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return False


def main():
    success = True
    for fname, url in MODELS.items():
        out_path = os.path.join(OUT_DIR, fname)
        print(f"Processing model: {fname}")

        if os.path.exists(out_path):
            print(f"Already exists → {out_path}")
            continue

        ok = download(url, out_path)
        if not ok:
            success = False

    if not success:
        print("Model download failed", file=sys.stderr)
        sys.exit(1)

    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
