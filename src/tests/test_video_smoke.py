from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tna.video import VideoRenderConfig, render_video


def test_video_render_chain(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the end-to-end video smoke test")

    out_dir = tmp_path / "video"
    config = VideoRenderConfig(
        out_dir=out_dir,
        width=192,
        height=108,
        fps=2,
        duration=1.0,
        foundation="off",
        seed=20260613,
        keep_frames=False,
        preset="preview",
        audio=False,
    )
    result = render_video(config)

    video = Path(result["video"])
    provenance = Path(result["provenance"])
    manifest = Path(result["asset_manifest"])

    assert video.is_file() and video.stat().st_size > 0
    assert provenance.is_file() and provenance.stat().st_size > 0
    assert manifest.is_file() and manifest.stat().st_size > 0
