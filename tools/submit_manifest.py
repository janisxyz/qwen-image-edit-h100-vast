#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload references and queue a production manifest.")
    parser.add_argument("--api", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    api = args.api.rstrip("/")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {args.token}"}
    asset_cache: dict[str, str] = {}

    with httpx.Client(timeout=120) as client:
        prepared_jobs = []
        for source_job in manifest["jobs"]:
            job = dict(source_job)
            uploaded = []
            for filename in job.pop("reference_files"):
                source = (manifest_path.parent / filename).resolve()
                key = str(source)
                if key not in asset_cache:
                    with source.open("rb") as handle:
                        response = client.post(
                            f"{api}/v1/assets",
                            headers=headers,
                            files={"file": (source.name, handle)},
                        )
                    response.raise_for_status()
                    asset_cache[key] = response.json()["asset"]
                uploaded.append(asset_cache[key])
            job["references"] = uploaded[:3]
            prepared_jobs.append(job)

        response = client.post(
            f"{api}/v1/batches",
            headers=headers,
            json={"jobs": prepared_jobs},
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
