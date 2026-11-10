from pathlib import Path

from fastapi import HTTPException, UploadFile


def save_upload(upload: UploadFile, path: Path, limit: int) -> Path:
    """Stream an upload to path; refuse it with 413 past limit bytes."""
    size = 0
    with path.open("wb") as out:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "upload too large")
            out.write(chunk)
    return path
