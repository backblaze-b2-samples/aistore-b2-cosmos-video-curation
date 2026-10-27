# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aistore_b2_curation import media


@pytest.fixture
def media_tools(monkeypatch):
    calls = []
    versions = {"ffmpeg": "8.0.1", "ffprobe": "8.0.1"}
    decoders = {name: " VFS..D h264 H.264\n VFS..D vp9 Google VP9\n" for name in versions}
    encoders = " V....D h264_nvenc NVIDIA NVENC\n V....D libvpx-vp9 libvpx VP9\n"

    def run(cmd, **kwargs):
        calls.append(cmd)
        name = Path(cmd[0]).name
        if cmd[1:] == ["-version"]:
            out = f"{name} version {versions[name]} Copyright (c) FFmpeg developers\n"
        elif "-decoders" in cmd:
            out = decoders[name]
        elif "-encoders" in cmd:
            out = encoders
        else:
            out = ""
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(media.subprocess, "run", run)
    return calls, versions, decoders


@pytest.mark.parametrize("encoder", ["h264_nvenc", "libvpx-vp9"])
def test_preflight_checks_both_tools_and_exercises_selected_encoder(media_tools, encoder):
    calls, _, _ = media_tools
    media.check_media_tools(encoder)
    assert [Path(cmd[0]).name for cmd in calls if "-version" in cmd] == ["ffmpeg", "ffprobe"]
    encode = calls[-1]
    assert [encode[i + 1] for i, flag in enumerate(encode) if flag == "-c:v"] == [
        "rawvideo",
        encoder,
    ]
    assert encode[encode.index("-frames:v") + 1] == "1"
    assert encode[-5:] == ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"]


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_missing_tool_is_actionable(monkeypatch, media_tools, missing):
    monkeypatch.setattr(
        media.shutil, "which", lambda name: None if name == missing else f"/usr/bin/{name}"
    )
    with pytest.raises(media.MediaPrerequisiteError, match=rf"install FFmpeg 8.*{missing}"):
        media.check_media_tools("h264_nvenc")


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("version", ["7.1.2", "N-123456-gabcdef"])
def test_old_or_unidentified_version_is_rejected(media_tools, tool, version):
    calls, versions, _ = media_tools
    versions[tool] = version
    with pytest.raises(media.MediaPrerequisiteError, match=rf"{tool} 8\+ is required"):
        media.check_media_tools("h264_nvenc")
    assert not any("-c:v" in cmd for cmd in calls)


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_hardware_only_h264_decoder_is_insufficient(media_tools, tool):
    _, _, decoders = media_tools
    decoders[tool] = " V..... h264_cuvid Nvidia CUVID H264 decoder\n"
    with pytest.raises(media.MediaPrerequisiteError, match=rf"{tool}.*software h264 decoder"):
        media.check_media_tools("h264_nvenc")


def test_vp9_output_requires_software_vp9_decoding(media_tools):
    _, _, decoders = media_tools
    decoders["ffprobe"] = " VFS..D h264 H.264\n"
    with pytest.raises(media.MediaPrerequisiteError, match=r"ffprobe.*software vp9 decoder"):
        media.check_media_tools("libvpx-vp9")


def test_missing_selected_encoder_is_rejected(monkeypatch, media_tools):
    run = media.subprocess.run

    def without_nvenc(cmd, **kwargs):
        if "-encoders" in cmd:
            return SimpleNamespace(
                stdout=" V....D h264_nvenc_unavailable not the requested codec\n"
            )
        return run(cmd, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", without_nvenc)
    with pytest.raises(media.MediaPrerequisiteError, match="requires the h264_nvenc encoder"):
        media.check_media_tools("h264_nvenc")


def test_registered_nvenc_encoder_must_work_at_runtime(monkeypatch, media_tools):
    run = media.subprocess.run

    def broken_nvenc(cmd, **kwargs):
        if "-c:v" in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr="Cannot load libcuda.so.1")
        return run(cmd, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", broken_nvenc)
    with pytest.raises(media.MediaPrerequisiteError, match="Cannot load libcuda"):
        media.check_media_tools("h264_nvenc")


@pytest.mark.parametrize("error", [FileNotFoundError(), OSError("executable cannot run")])
def test_disappearing_or_broken_executable_is_actionable(monkeypatch, media_tools, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(media.subprocess, "run", fail)
    with pytest.raises(media.MediaPrerequisiteError):
        media.check_media_tools("h264_nvenc")


def test_media_probe_timeout_is_actionable(monkeypatch, media_tools):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(media.subprocess, "run", timeout)
    with pytest.raises(media.MediaPrerequisiteError, match="timed out after 30s"):
        media.check_media_tools("h264_nvenc")


@pytest.mark.parametrize("encoder", ["h264_nvenc", "libvpx-vp9"])
def test_runtime_probe_uses_components_in_curators_minimal_ffmpeg_build(
    monkeypatch, media_tools, encoder
):
    run = media.subprocess.run

    def minimal_build(cmd, **kwargs):
        if "-i" in cmd:
            formats = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "-f"]
            assert formats == ["image2pipe", "mp4"]
            assert cmd[cmd.index("-c:v") + 1] == "rawvideo"
            assert cmd[cmd.index("-i") + 1] == "pipe:0"
            width, height = map(int, cmd[cmd.index("-video_size") + 1].split("x"))
            if encoder == "h264_nvenc" and width < 145:
                raise subprocess.CalledProcessError(
                    1, cmd, stderr="Frame Dimension less than the minimum supported value"
                )
            assert width % 2 == height % 2 == 0
            assert len(kwargs["input"]) == width * height * 3 // 2
            assert kwargs["text"] is False
            assert "lavfi" not in cmd and "null" not in cmd
        return run(cmd, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", minimal_build)
    media.check_media_tools(encoder)
