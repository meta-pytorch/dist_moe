# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Validate the provider and runtime used by this frozen benchmark snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .metrics import artifact_digest

_ROOT = Path(__file__).parent


def verify_runtime(scope: str) -> dict:  # noqa: C901
    """Require the measured provider bytes and runtime before GPU execution.

    Args:
        scope: ``isolated`` or ``comparison`` benchmark family.

    Returns:
        Actual provider, runtime, and native extension identities.

    Raises:
        RuntimeError: A provider file, dependency, or runtime identity differs.
    """
    import dist_moe
    import torch

    folder = {"isolated": "isolated", "comparison": "mok-comparison"}[scope]
    lock = json.loads((_ROOT / folder / "runtime.json").read_text())
    providers = json.loads((_ROOT / "versions.json").read_text())
    differences = []
    if torch.version.git_version != lock["torch_revision"]:
        differences.append(f"Torch revision {torch.version.git_version}")
    if torch.version.cuda != lock["cuda"]:
        differences.append(f"CUDA version {torch.version.cuda}")
    if list(torch.cuda.nccl.version()) != lock["nccl"]:
        differences.append(f"NCCL version {torch.cuda.nccl.version()}")
    if torch.cuda.get_arch_list() != lock["torch_arch_list"]:
        differences.append(f"architecture list {torch.cuda.get_arch_list()}")
    actual = {}
    for name, expected in lock["distributions"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            differences.append(f"missing distribution {name}")
            continue
        actual[name] = version
        if version != expected:
            differences.append(f"{name}={version}, expected {expected}")
    module_root = Path(dist_moe.__file__).resolve().parent.parent
    for relative, expected in providers["dist_moe_files"].items():
        path = module_root / relative
        if not path.is_file() or artifact_digest(path) != expected:
            differences.append(f"Dist-MoE provider file {relative}")
    identity = {
        "provider_revision": providers["provider_revision"],
        "provider_path": str(module_root / "dist_moe"),
        "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda,
        "distributions": actual,
    }
    if scope == "comparison":
        from mok import _C, functional

        native_root = Path(functional.__file__).resolve().parent.parent
        for relative, expected in providers["mok"]["files"].items():
            path = native_root / relative
            if not path.is_file() or artifact_digest(path) != expected:
                differences.append(f"MoK provider file {relative}")
        identity["mok_revision"] = providers["mok"]["source_sha"]
        extension_digest = artifact_digest(Path(_C.__file__))
        if extension_digest != providers["mok"]["extension_sha256"]:
            differences.append("imported MoK extension digest")
        identity["mok_extension_sha256"] = extension_digest
    if differences:
        raise RuntimeError("frozen runtime mismatch: " + "; ".join(differences))
    return identity


def campaign_main(scope: str) -> None:
    """Run one dated benchmark family with its original measurement policy.

    Args:
        scope: Isolated Dist-MoE or shared native comparison family.
    """
    from .run import main

    sys.argv.extend(["--scope", scope])
    if scope == "comparison":
        sys.argv.extend(["--execution-mode", "cuda_graph", "--mxfp8-fast-math"])
    else:
        sys.argv.extend(["--backend", "dist_moe"])
    main()


def main() -> None:
    """Check the original installed runtime without starting a distributed job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("isolated", "comparison"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify_runtime(args.scope), indent=2))


if __name__ == "__main__":
    main()
