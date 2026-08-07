from pathlib import Path

from tna.cli import build_arg_parser, run


def test_base_ladder_smoke_outputs(tmp_path: Path) -> None:
    parser = build_arg_parser()
    out_dir = tmp_path / "smoke"

    args = parser.parse_args(
        [
            "--entities",
            "fr,uy,ps",
            "--intents",
            "equal",
            "--backends",
            "pixel,palette,pca,svg",
            "--canvas",
            "32x24",
            "--out-dir",
            str(out_dir),
            "--audio-duration",
            "0.25",
            "--sample-rate",
            "8000",
            "--pca-components",
            "4",
        ]
    )

    traces = run(args)

    assert len(traces) == 4
    assert (out_dir / "run_summary.json").is_file()
    assert (out_dir / "run_summary.md").is_file()
    assert (out_dir / "figures" / "ladder_equal.png").is_file()
    assert (out_dir / "figures" / "alpha_morph_pixel.png").is_file()
    assert (out_dir / "figures" / "alpha_morph_pca.png").is_file()

    trace_files = sorted((out_dir / "traces").glob("trace_equal_*.json"))
    assert len(trace_files) == 4
