#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def upload(client: httpx.Client, api: str, token: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            f"{api}/v1/assets",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (path.name, handle)},
        )
    response.raise_for_status()
    return response.json()["asset"]


def wait_for_job(client: httpx.Client, api: str, token: str, prompt_id: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        response = client.get(f"{api}/v1/jobs/{prompt_id}", headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("completed"):
            return data
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidate-sizes", default="1,2,3,4")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0)
    args = parser.parse_args()

    api = args.api.rstrip("/")
    sizes = [int(value) for value in args.candidate_sizes.split(",") if value.strip()]

    with httpx.Client(timeout=120) as client:
        references = [upload(client, api, args.token, Path(value)) for value in args.reference][:3]
        results = []

        for candidates in sizes:
            payload = {
                "prompt": args.prompt,
                "references": references,
                "candidates": candidates,
                "seed": 123456,
                "steps": args.steps,
                "cfg": args.cfg,
                "output_prefix": f"benchmark/batch_{candidates}",
            }
            headers = {"Authorization": f"Bearer {args.token}"}
            started = time.perf_counter()
            response = client.post(f"{api}/v1/jobs", headers=headers, json=payload)
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            job = wait_for_job(client, api, args.token, prompt_id)
            elapsed = time.perf_counter() - started
            record = {
                "candidates": candidates,
                "seconds_total": round(elapsed, 2),
                "seconds_per_image": round(elapsed / candidates, 2),
                "outputs": len(job.get("outputs", [])),
            }
            results.append(record)
            print(json.dumps(record))

    best = min(results, key=lambda item: item["seconds_per_image"])
    print("\nBest candidate batch:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
