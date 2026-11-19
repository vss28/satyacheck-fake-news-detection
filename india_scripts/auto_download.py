"""
Automated Dataset Downloader for Indian Fake News
=================================================
Downloads:
1. IFND (Indian Fake News Dataset) directly from the official research repo on GitHub.
"""

import os
import urllib.request
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

IFND_DIR = "data/indian/raw/ifnd"
os.makedirs(IFND_DIR, exist_ok=True)

IFND_URL = "https://raw.githubusercontent.com/sonalgarg174/Dataset/master/IFND.csv"
IFND_PATH = os.path.join(IFND_DIR, "IFND.csv")

def download_ifnd():
    if os.path.exists(IFND_PATH) and os.path.getsize(IFND_PATH) > 100000:
        print(f"✅ IFND dataset already present at {IFND_PATH} ({os.path.getsize(IFND_PATH)/1e6:.2f} MB)")
        return

    print("📥 Downloading official IFND dataset (English - Indian news context)...")
    print(f"   URL: {IFND_URL}")

    def reporthook(blocknum, blocksize, totalsize):
        read = blocknum * blocksize
        if totalsize > 0:
            percent = read * 100 / totalsize
            sys.stdout.write(f"\r   Progress: {percent:.1f}% ({read/1e6:.2f} MB / {totalsize/1e6:.2f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(IFND_URL, IFND_PATH, reporthook=reporthook)
        print(f"\n✅ IFND successfully downloaded: {os.path.getsize(IFND_PATH)/1e6:.2f} MB")
    except Exception as e:
        print(f"\n❌ Failed to download IFND: {e}")

if __name__ == "__main__":
    download_ifnd()
