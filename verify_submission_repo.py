#!/usr/bin/env python3
"""Verify core imports in the checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def check_imports(errors: list[str]) -> None:
    sys.path[:0] = [str(ROOT), str(ROOT / "minigrid_part")]
    try:
        import minigrid  # noqa: F401
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        from bcos_rl_generic.bcosDQN import BcosDQN  # noqa: F401
        from bcos_rl_generic.bcosPPO import BcosPPO  # noqa: F401
        from minigrid_part.env_wrappers import make_wrapper  # noqa: F401
    except Exception as exc:
        errors.append(f"core import error: {exc}")


def main() -> int:
    errors: list[str] = []

    print("Checking core imports...")
    check_imports(errors)

    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        print(f"Verification failed with {len(errors)} error(s).")
        return 1
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
