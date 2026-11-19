# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Reproduce the routed-only Dist-MoE benchmark campaign."""

from ..provenance import campaign_main


def main() -> None:
    """Select the isolated campaign and its original graph timing policy."""
    campaign_main("isolated")


if __name__ == "__main__":
    main()
