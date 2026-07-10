# License notes

This provider uses the official Meta `facebookresearch/sam2` repository pinned at
commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` and the official
`sam2.1_hiera_small.pt` checkpoint.

The official repository `LICENSE` is Apache License 2.0. Its README explicitly states
that the SAM 2 model checkpoints, demo code, and training code are licensed under
Apache 2.0. The experiment verifies both statements and records the vendored license
hash in each run's `metrics.json`.

Apache-2.0 permits commercial use subject to its conditions, including preservation of
the license and applicable notices. The Python/CUDA runtime dependencies retain their
own licenses and must be reviewed for the final distribution. This file is an
engineering provenance record, not legal advice.

The provider does not use MatAnyone or its non-commercial S-Lab model weights.
