from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.warmup import warmup_models  # noqa: E402


def main() -> int:
    print('Model warmup starting')
    try:
        started: dict[str, float] = {}

        def progress(step: str, phase: str) -> None:
            if phase == 'start':
                started[step] = time.perf_counter()
                print(f'[{step}] start')
                return
            if phase == 'done':
                elapsed = time.perf_counter() - started[step]
                print(f'[{step}] done in {elapsed:.2f}s')
                return
            if phase == 'skipped':
                print(f'[{step}] skipped (RERANK_ENABLED=false)')

        warmup_models(progress=progress)
        print('Model warmup completed')
        return 0
    except Exception as exc:
        print(f'Model warmup failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
