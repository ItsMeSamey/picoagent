"""Prepared manifests independently reviewed after deterministic CPU rebuilding.

The builder cannot add approvals. This file is deliberately excluded from the
transform identity to avoid a hash cycle; the full training source/resume identity
still includes it. Production requires both this review and an explicit config pin.
"""

# Review: docs/validation/20261001-prepared-approval.json
# The separate independent equivalence report remains unmodified audit evidence.
APPROVED_PREPARED_MANIFESTS: frozenset[str] = frozenset({
    "6dcd339eb9b25ea8edb2ad625b8988cf67baa814614407d983a8c5b0734945a6",
})
