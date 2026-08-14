"""
GGUF model downloader utility.
Downloads quantized models directly from Hugging Face Hub into the local models directory.
"""

from pathlib import Path
from typing import Optional
import sys
import requests
from llama_cluster.config import Config, get_config


def download_gguf_model(
    repo_id: str = "bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    target_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    cfg: Optional[Config] = None
) -> Path:
    """
    Downloads a GGUF file from Hugging Face to local models folder with progress logging.
    """
    config = cfg or get_config()
    out_dir = target_dir or config.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_file = out_dir / filename
    if dest_file.exists():
        print(f"[+] Model already exists at: {dest_file}")
        return dest_file

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {}
    
    token = hf_token or config.hf_token
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[*] Downloading {filename} from {repo_id}...")
    print(f"[*] URL: {url}")
    print(f"[*] Target destination: {dest_file}")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(dest_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\rProgress: [{pct:6.2f}%] ({mb_done:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()

        print("\n[+] Download complete! W model obtained, zero cap.")
        return dest_file

    except Exception as e:
        if dest_file.exists():
            dest_file.unlink()  # Remove incomplete download
        raise RuntimeError(f"Failed to download model {filename}: {e}")
