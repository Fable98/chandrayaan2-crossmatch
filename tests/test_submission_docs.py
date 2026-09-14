"""
tests/test_submission_docs.py — SIH documentation consistency (Task 3).

Asserts SIH_SUBMISSION_README.md documents the current system:
descriptor-level illumination invariance, GOA preselection, traffic-light
verification, held-out RMSE, cyclic consistency, spatial uniformity, SSIM —
and fails on stale unverified claims disproven by
reports/current_region_matching_summary.json (all six regions pass Gate 1).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "SIH_SUBMISSION_README.md"

REQUIRED_TOPICS = [
    ("descriptor-level illumination invariance", ["descriptor-level illumination invariance"]),
    ("Log-Gabor Phase Congruency", ["Log-Gabor"]),
    ("GOA preselection", ["GOA preselection", "Gradient Orientation Agreement"]),
    ("traffic-light verification", ["traffic-light", "traffic light", "Traffic Light"]),
    ("held-out RMSE", ["held_out_rmse", "held-out"]),
    ("cyclic consistency", ["cyclic"]),
    ("spatial uniformity", ["uniformity", "spatial uniformity"]),
    ("SSIM", ["SSIM", "ssim_score", "structural similarity"]),
]

# Disproven by the 2026-09-14 region report (6/6 Gate-1 success): these exact
# stale wordings must not appear as current claims.
STALE_CLAIMS = [
    "only region_006 still succeeds",
    "only `region_006` succeeds",
    "region_001/002/003/005 fail Gates",
    "region_001/002/003/005` fail Gates",
    "explicitly NOT claimed sun-angle invariant",
]


def _text() -> str:
    assert README.exists(), f"missing {README}"
    return README.read_text(encoding="utf-8")


def test_submission_readme_covers_current_system():
    text = _text()
    lowered = text.lower()
    for label, variants in REQUIRED_TOPICS:
        assert any(v.lower() in lowered for v in variants), (
            f"SIH_SUBMISSION_README.md missing required topic: {label}"
        )


def test_submission_readme_has_no_stale_defeatist_claims():
    text = _text()
    for stale in STALE_CLAIMS:
        assert stale not in text, (
            f"SIH_SUBMISSION_README.md still contains stale claim {stale!r}; "
            "update it with current report evidence."
        )
