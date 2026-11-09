from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from v2a_inspect.models import VideoAsset

RunStatus = Literal["idle", "running", "complete", "failed"]


@dataclass(frozen=True)
class VideoAssetSnapshot:
    asset: VideoAsset | None
    version: int
    asset_version: int
    status: RunStatus
    current_stage: str | None
    error: str | None
    updated_at: datetime


class VideoAssetStore:
    """In-memory VideoAsset state with async change notifications."""

    def __init__(self) -> None:
        self._asset: VideoAsset | None = None
        self._version = 0
        self._asset_version = 0
        self._status: RunStatus = "idle"
        self._current_stage: str | None = None
        self._error: str | None = None
        self._updated_at = datetime.now(UTC)
        self._condition = asyncio.Condition()

    async def set_asset(
        self, video_asset: VideoAsset, *, stage: str | None = None
    ) -> None:
        async with self._condition:
            self._asset = video_asset
            self._status = "running"
            self._current_stage = stage
            self._error = None
            self._asset_version += 1
            self._bump_locked()

    async def set_complete_asset(
        self, video_asset: VideoAsset, *, stage: str | None = None
    ) -> None:
        async with self._condition:
            self._asset = video_asset
            self._status = "complete"
            self._current_stage = stage
            self._error = None
            self._asset_version += 1
            self._bump_locked()

    async def publish_asset_mutation(
        self, video_asset: VideoAsset, *, stage: str | None = None
    ) -> None:
        async with self._condition:
            self._asset = video_asset
            self._status = "running"
            self._current_stage = stage
            self._error = None
            self._bump_locked()

    async def set_running(self, *, stage: str) -> None:
        async with self._condition:
            self._status = "running"
            self._current_stage = stage
            self._error = None
            self._bump_locked()

    async def set_complete(self, *, stage: str = "complete") -> None:
        async with self._condition:
            self._status = "complete"
            self._current_stage = stage
            self._error = None
            self._bump_locked()

    async def set_error(self, error: str, *, stage: str | None = None) -> None:
        async with self._condition:
            self._status = "failed"
            self._current_stage = stage or self._current_stage
            self._error = error
            self._bump_locked()

    async def clear(self, *, stage: str = "cleaned") -> None:
        async with self._condition:
            self._asset = None
            self._status = "idle"
            self._current_stage = stage
            self._error = None
            self._asset_version += 1
            self._bump_locked()

    async def touch(self, *, stage: str | None = None) -> None:
        async with self._condition:
            self._current_stage = stage or self._current_stage
            self._bump_locked()

    async def snapshot(self) -> VideoAssetSnapshot:
        async with self._condition:
            return self._snapshot_locked()

    async def wait_for_change(self, version: int) -> VideoAssetSnapshot:
        async with self._condition:
            await self._condition.wait_for(lambda: self._version != version)
            return self._snapshot_locked()

    def _bump_locked(self) -> None:
        self._version += 1
        self._updated_at = datetime.now(UTC)
        self._condition.notify_all()

    def _snapshot_locked(self) -> VideoAssetSnapshot:
        return VideoAssetSnapshot(
            asset=self._asset,
            version=self._version,
            asset_version=self._asset_version,
            status=self._status,
            current_stage=self._current_stage,
            error=self._error,
            updated_at=self._updated_at,
        )
