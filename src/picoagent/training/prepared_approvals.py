"""Prepared manifests independently reviewed after deterministic CPU rebuilding.

The builder cannot add approvals. This file is deliberately excluded from the
transform identity to avoid a hash cycle; the full training source/resume identity
still includes it. Production requires both this review and an explicit config pin.
"""

# Review: docs/validation/20261001-prepared-approval.json
# The separate independent equivalence report remains unmodified audit evidence.
# Review: docs/validation/20261002-prepared-v3-approval.json
# Review: docs/validation/20261002-prepared-v4-approval.json
APPROVED_PREPARED_MANIFESTS: frozenset[str] = frozenset({
    "24f2572dab280cfeb4d6706d7181ac9d5d5d153713d53c7aee18e83f8b9ece55",
    "9612db32062a7d90206d85f79b4de99836936f820626d81aef3da466178867c1",
    "6dcd339eb9b25ea8edb2ad625b8988cf67baa814614407d983a8c5b0734945a6",
})
