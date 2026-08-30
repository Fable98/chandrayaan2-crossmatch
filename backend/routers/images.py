"""
images.py — Static image serving for /images/{sensor}/{tile_id}

This file is a PLACEHOLDER for project structure clarity.

Image serving is handled by FastAPI's StaticFiles mount in main.py:

    app.mount("/images", StaticFiles(directory=images_dir), name="images")

This means requests like GET /images/ohrc/ohrc_512.png are served directly
from the filesystem with no custom route handler, no on-the-fly processing,
and no resizing. The static mount is the fastest way to serve tiles.

If you need custom logic for image serving in the future (e.g., auth checks,
range requests, or metadata headers), convert this into a real router. Until
then, the StaticFiles mount in main.py is sufficient.
"""
