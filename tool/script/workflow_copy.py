
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob

def copy_workflow_json_files(src_dir, dst_dir):
    """
    Recursively copy all json files from src_dir to dst_dir, preserving relative directory structure.
    """
    if not os.path.exists(src_dir):
        print(f"Source directory does not exist: {src_dir}")
        return

    if "template/nndeploy-workflow" in dst_dir:
        # delete all files in the directory but keep the directory structure
        if os.path.exists(dst_dir):
            for root, dirs, files in os.walk(dst_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".json"):
                src_file_path = os.path.join(root, file)
                # Calculate relative path
                rel_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, rel_path)
                dst_file_dir = os.path.dirname(dst_file_path)
                os.makedirs(dst_file_dir, exist_ok=True)
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied: {src_file_path} -> {dst_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recursively copy all json files from source directory to target directory (preserving relative structure)")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory (e.g. nndeploy-workflow)")
    parser.add_argument("--dst_dir", type=str, required=True, help="Target directory (e.g. resources/workflow)")
    args = parser.parse_args()
    copy_workflow_json_files(args.src_dir, args.dst_dir)

# copy to resources/workflow
## python3 workflow_copy.py --src_dir /home/always/github/public/nndeploy-workflow --dst_dir /home/always/github/public/nndeploy/resources/workflow
# copy to /resources/template/nndeploy-workflow
## python3 workflow_copy.py --src_dir /home/always/github/public/nndeploy-workflow --dst_dir /home/always/github/public/nndeploy/resources/template/nndeploy-workflow
