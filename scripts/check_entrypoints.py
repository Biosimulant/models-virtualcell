#!/usr/bin/env python3
"""Verify that model manifest entrypoints are importable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

# When running from the Biosimulant monorepo checkout (without installing bsim),
# ensure `import bsim` resolves to the installable package at `bsim/src/bsim/`.
REPO_ROOT = ROOT.parent
BSIM_SRC = REPO_ROOT / "bsim" / "src"
if BSIM_SRC.exists():
    sys.path.insert(0, str(BSIM_SRC))


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    return data


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" in entrypoint:
        module_name, attr = entrypoint.split(":", 1)
    else:
        module_name, attr = entrypoint.rsplit(".", 1)
    return module_name, attr


def main() -> int:
    errors: list[str] = []
    manifests = sorted(ROOT.rglob("model.yaml"))
    for manifest_path in manifests:
        if "/templates/" in str(manifest_path):
            continue

        try:
            manifest = _load_yaml(manifest_path)
            bsim = manifest.get("bsim") if isinstance(manifest.get("bsim"), dict) else {}
            entrypoint = bsim.get("entrypoint") or manifest.get("entrypoint")
            if not isinstance(entrypoint, str) or not entrypoint.strip():
                errors.append(f"{manifest_path}: missing bsim.entrypoint")
                continue

            module_name, attr = _split_entrypoint(entrypoint)
            model_root = manifest_path.parent
            sys.path.insert(0, str(model_root))
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, attr):
                    errors.append(f"{manifest_path}: entrypoint attribute not found: {entrypoint}")
                    continue
                target = getattr(module, attr)
                if not callable(target):
                    errors.append(f"{manifest_path}: entrypoint is not callable: {entrypoint}")
            finally:
                if sys.path and sys.path[0] == str(model_root):
                    sys.path.pop(0)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{manifest_path}: {exc}")

    if errors:
        print("Entrypoint check failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print(f"Validated entrypoints for {len(manifests)} model manifest(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
