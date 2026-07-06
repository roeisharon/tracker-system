"""Streaming video input."""

from .source import VideoMetadata, VideoSource, VideoSourceError, is_url

__all__ = ["VideoMetadata", "VideoSource", "VideoSourceError", "is_url"]
