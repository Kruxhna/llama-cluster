#!/usr/bin/env python3
"""
Standalone model downloader script for llama-cluster.
Usage: python scripts/download_model.py [--repo REPO] [--file FILE]
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on Python path if script is invoked directly
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llama_cluster.downloader import download_gguf_model


def main():
    parser = argparse.ArgumentParser(description="Download GGUF weights for llama-cluster")
    parser.add_argument("--repo", default="bartowski/Llama-3.2-1B-Instruct-GGUF", help="Hugging Face Repository ID")
    parser.add_argument("--file", default="Llama-3.2-1B-Instruct-Q4_K_M.gguf", help="GGUF Quantization Filename")
    args = parser.parse_args()

    print(f"[*] Downloading {args.file} from Hugging Face repo {args.repo}...")
    dest = download_gguf_model(repo_id=args.repo, filename=args.file)
    print(f"[+] Download finished successfully: {dest}")


if __name__ == "__main__":
    main()
