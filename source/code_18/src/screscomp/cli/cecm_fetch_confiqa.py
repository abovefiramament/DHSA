from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from screscomp.data import dump_json


CONFIQA_FILES = {
    "qa": {
        "filename": "ConFiQA-QA.json",
        "url": "https://raw.githubusercontent.com/byronBBL/Context-DPO/master/ConFiQA/ConFiQA-QA.json",
    },
    "mr": {
        "filename": "ConFiQA-MR.json",
        "url": "https://raw.githubusercontent.com/byronBBL/Context-DPO/master/ConFiQA/ConFiQA-MR.json",
    },
    "mc": {
        "filename": "ConFiQA-MC.json",
        "url": "https://raw.githubusercontent.com/byronBBL/Context-DPO/master/ConFiQA/ConFiQA-MC.json",
    },
}

REQUIRED_KEYS = ("question", "cf_context", "orig_answer", "cf_answer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch official Context-DPO ConFiQA JSON files for CK/CECM runs.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tasks", type=str, default="qa", help="Comma-separated subset of qa,mr,mc or 'all'.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--retry-sleep", type=float, default=15.0)
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args()


def _parse_tasks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return ["qa", "mr", "mc"]
    tasks = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(CONFIQA_FILES))
    if unknown:
        raise ValueError(f"Unknown ConFiQA tasks: {unknown}; allowed={sorted(CONFIQA_FILES)}")
    if not tasks:
        raise ValueError("--tasks is empty")
    return tasks


def _download(url: str, path: Path, *, force: bool, retries: int, retry_sleep: float, timeout: float) -> bytes:
    if path.exists() and not force:
        return path.read_bytes()
    req = Request(url, headers={"User-Agent": "screscomp-cecm-fetch/1.0"})
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urlopen(req, timeout=timeout) as response:
                data = response.read()
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, retries):
                raise
            print(
                f"[cecm-fetch-confiqa] download failed attempt={attempt}/{retries}: {exc}; "
                f"retrying in {retry_sleep:g}s",
                flush=True,
            )
            time.sleep(retry_sleep)
    else:  # pragma: no cover
        raise RuntimeError(f"Download failed without exception: {url}") from last_exc
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def _validate_json(data: bytes, *, task: str) -> tuple[int, list[str]]:
    rows = json.loads(data.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{task} payload is not a JSON list")
    missing: set[str] = set()
    checked = rows[: min(len(rows), 20)]
    for row in checked:
        if not isinstance(row, dict):
            raise ValueError(f"{task} row is not an object")
        for key in REQUIRED_KEYS:
            if key not in row:
                missing.add(key)
    if missing:
        raise ValueError(f"{task} rows are missing required keys in first {len(checked)} rows: {sorted(missing)}")
    return len(rows), sorted(checked[0].keys()) if checked else []


def main() -> None:
    args = parse_args()
    tasks = _parse_tasks(args.tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for task in tasks:
        spec = CONFIQA_FILES[task]
        path = args.out_dir / spec["filename"]
        data = _download(
            spec["url"],
            path,
            force=args.force,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            timeout=args.timeout,
        )
        row_count, sample_keys = _validate_json(data, task=task)
        sha256 = hashlib.sha256(data).hexdigest()
        manifest_rows.append(
            {
                "task": task,
                "filename": spec["filename"],
                "url": spec["url"],
                "path": str(path),
                "bytes": len(data),
                "sha256": sha256,
                "rows": row_count,
                "required_keys": list(REQUIRED_KEYS),
                "sample_keys": sample_keys,
            }
        )
        print(f"[cecm-fetch-confiqa] task={task} rows={row_count} sha256={sha256} path={path}", flush=True)

    dump_json(
        args.out_dir / "download_manifest.json",
        {
            "source": "byronBBL/Context-DPO ConFiQA",
            "source_url": "https://github.com/byronBBL/Context-DPO/tree/master/ConFiQA",
            "tasks": manifest_rows,
        },
    )


if __name__ == "__main__":
    main()
