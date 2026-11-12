"""Validate and encode the original local image supplied with a question."""

import base64
from io import BytesIO
from pathlib import Path

from .decision import ImageReference


def image_data_uri(reference: ImageReference) -> str:
    if reference.uri.startswith("data:"):
        data = base64.b64decode(reference.uri.split(",", 1)[1], validate=True)
    else:
        data = Path(reference.uri).read_bytes()
    from PIL import Image

    with Image.open(BytesIO(data)) as decoded:
        if Image.MIME.get(decoded.format) != reference.mime_type:
            raise ValueError("image bytes do not match the declared MIME type")
        decoded.verify()
    return f"data:{reference.mime_type};base64," + base64.b64encode(data).decode(
        "ascii"
    )
