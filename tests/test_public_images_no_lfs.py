"""No LFS pointers under lunar-frontend/public/images/.

Static tiles are intentionally TRACKED as regular blobs (see .gitignore
note + .gitattributes exemption). A 131-byte `version https://git-lfs`
pointer 404s/breaks Next.js static serving on plain `git clone` + Vercel.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = REPO_ROOT / "lunar-frontend" / "public" / "images"
LFS_MARKER = b"version https://git-lfs.github.com/spec/v1"


def test_no_lfs_pointers_under_public_images():
    assert IMAGES_ROOT.is_dir(), f"missing {IMAGES_ROOT}"
    offenders = []
    for p in IMAGES_ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                head = f.read(200)
        except OSError:
            continue
        if head.startswith(LFS_MARKER):
            offenders.append(str(p.relative_to(REPO_ROOT)))
    assert not offenders, f"LFS pointers under public/images/: {offenders}"


def test_lro_nac_tiles_are_real_pngs():
    for name in ("region_001.png", "region_003.png", "region_006.png"):
        p = IMAGES_ROOT / "lro_nac" / name
        assert p.is_file(), f"missing {p}"
        with open(p, "rb") as f:
            magic = f.read(8)
        assert magic == b"\x89PNG\r\n\x1a\n", f"{name} is not a real PNG"
        assert p.stat().st_size > 10000, f"{name} suspiciously small: {p.stat().st_size}"
