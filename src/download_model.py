import argparse
import os
 
from huggingface_hub import snapshot_download
 
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", default="models/final")
args = ap.parse_args()
 
path = snapshot_download(repo_id=args.repo, local_dir=args.out,
                         token=os.environ.get("HF_TOKEN"))
print("Đã tải về:", path)