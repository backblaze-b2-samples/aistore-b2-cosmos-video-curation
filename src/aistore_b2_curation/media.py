# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check system media tools before transferring a video corpus."""

from __future__ import annotations

import re
import shutil
import subprocess


class MediaPrerequisiteError(RuntimeError):
    """Required media tools or codecs are unavailable."""


def _output(executable: str, *args: str, input_data: bytes | None = None) -> str:
    try:
        result = subprocess.run(
            [executable, *args],
            input=input_data,
            capture_output=True,
            text=input_data is None,
            check=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise MediaPrerequisiteError(f"{executable} is unavailable on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaPrerequisiteError(f"{executable} {' '.join(args)} timed out after 30s") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise MediaPrerequisiteError(f"{executable} failed: {detail.strip()[-1000:]}") from exc
    return result.stdout if isinstance(result.stdout, str) else ""


def _has_codec(output: str, name: str) -> bool:
    return any(re.match(rf"^\s+V\S*\s+{re.escape(name)}\s", line) for line in output.splitlines())


def check_media_tools(encoder: str) -> None:
    """Require FFmpeg 8+, software output decoding, and a usable selected encoder.

    ClipWriter probes clips in a CPU-only actor, so hardware-only decoding is
    insufficient. A one-frame encode checks runtime support, including NVENC's
    driver and GPU requirements, without writing a media file.
    """
    if encoder not in ("h264_nvenc", "libvpx-vp9"):
        raise ValueError(f"unsupported encoder: {encoder}")
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable is None:
            raise MediaPrerequisiteError(f"install FFmpeg 8+ with {name} available on PATH")
        version = _output(executable, "-version")
        match = re.match(rf"^{name} version n?(\d+)(?:[.\s-])", version)
        if match is None or int(match[1]) < 8:
            reported = version.splitlines()[0] if version else "unknown version"
            raise MediaPrerequisiteError(f"{name} 8+ is required; found {reported}")
        tools[name] = executable

    decoder = "h264" if encoder == "h264_nvenc" else "vp9"
    for name, executable in tools.items():
        if not _has_codec(_output(executable, "-hide_banner", "-decoders"), decoder):
            raise MediaPrerequisiteError(f"{name} requires the software {decoder} decoder")
    ffmpeg = tools["ffmpeg"]
    if not _has_codec(_output(ffmpeg, "-hide_banner", "-encoders"), encoder):
        raise MediaPrerequisiteError(f"ffmpeg requires the {encoder} encoder")
    # Curator's minimal FFmpeg build enables image2pipe/rawvideo/mp4, not lavfi or null muxing.
    # A 256x256 frame accommodates NVENC's minimum dimensions on newer GPUs.
    width = height = 256
    frame = bytes([16]) * (width * height) + bytes([128]) * (width * height // 2)
    _output(
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-f", "image2pipe", "-c:v", "rawvideo",
        "-pixel_format", "yuv420p", "-video_size", f"{width}x{height}", "-framerate", "1",
        "-i", "pipe:0",
        "-frames:v", "1", "-an", "-c:v", encoder, "-pix_fmt", "yuv420p",
        "-threads", "1", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1",
        input_data=frame,
    )  # fmt: skip
