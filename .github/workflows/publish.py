import argparse
import os
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-C", "--manifest-path", default="Cargo.toml")
parser.add_argument("--version", required=True)
parser.add_argument("--token", required=True)
args = parser.parse_args()

out = subprocess.run(
    [
        "cargo", 
        "metadata", 
        "--format-version", 
        "1", 
        "--manifest-path", 
        args.manifest_path
    ], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE
)
out.check_returncode()

metadata = json.loads(out.stdout)
workspace = metadata["workspace_members"]

packages = [p for p in metadata['packages'] if p['id'] in workspace]
print(f"Found {len(packages)} packages:")
for p in packages:
    print("-", p["name"], p["version"])

for p in packages:
    if p['version'] == args.version.lstrip("v"):
        print("Publishing", p["name"])
        out = subprocess.run(
            [
                "cargo", 
                "publish", 
                "-p", 
                p["name"], 
                "--token", 
                args.token,
            ], 
        )
