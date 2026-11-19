# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Reproduce the matched CUDA-graph MoK comparison campaign."""

from ..provenance import campaign_main


def main() -> None:
    """Select matched shared work, graph execution and original fast math."""
    campaign_main("comparison")


if __name__ == "__main__":
    main()
