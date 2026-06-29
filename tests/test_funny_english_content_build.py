"""Tests for Funny English content build and runtime sweep scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_funny_english_content as builder
from scripts import funny_english_runtime_sweep as sweep


def test_build_funny_english_content_writes_manifest_notice_and_licenses(tmp_path: Path) -> None:
    """The build emits committed JSON metadata and generated runtime assets."""
    out = tmp_path / "funny_english"

    stats = builder.build_funny_english_content(builder.BuildOptions(out=out, force=True))

    assert stats.stages == 8
    assert stats.cards > 0
    assert (out / "manifest.json").exists()
    assert (out / "NOTICE").read_text(encoding="utf-8").startswith("Funny English NOTICE")
    assert (out / "licenses" / "MIT.txt").exists()
    assert (out / "music" / "bgm_loop.wav").exists()


def test_build_funny_english_content_includes_aesop_story_stages(tmp_path: Path) -> None:
    """Stages 6 and 7 carry the Aesop story reader cards with valid licenses."""
    out = tmp_path / "funny_english"

    builder.build_funny_english_content(builder.BuildOptions(out=out, force=True))

    hare = json.loads((out / "stages" / "stage_6.json").read_text(encoding="utf-8"))
    lion = json.loads((out / "stages" / "stage_7.json").read_text(encoding="utf-8"))
    assert hare["title"] == "The Hare and the Tortoise"
    assert lion["title"] == "The Lion and the Mouse"
    assert len(hare["cards"]) == 7
    assert len(lion["cards"]) == 8
    for card in (*hare["cards"], *lion["cards"]):
        assert card["type"] == "reader"
        assert card["card_id"].startswith("fe_aesop_")
        license_info = card["asset_license"]
        assert builder.LICENSE_REQUIRED_FIELDS <= set(license_info)
        assert license_info["license"] in builder.WHITELISTED_LICENSES


def test_build_funny_english_content_rejects_unknown_license(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown asset licenses fail the build gate."""
    out = tmp_path / "funny_english"
    cards = builder._starter_curriculum(out)
    cards[0][0]["asset_license"]["license"] = "CC-BY-SA"
    monkeypatch.setattr(builder, "_starter_curriculum", lambda _out: cards)

    with pytest.raises(builder.BuildError):
        builder.build_funny_english_content(builder.BuildOptions(out=out))


def test_funny_english_runtime_sweep_gates_zero_errors(tmp_path: Path) -> None:
    """The runtime sweep passes after a successful build and fails on missing assets."""
    repo_root = tmp_path
    out = repo_root / "assets" / "funny_english"
    builder.build_funny_english_content(builder.BuildOptions(out=out, force=True))

    stats = sweep.run_sweep(repo_root=repo_root)

    assert stats.error_count == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    first_stage = repo_root / manifest["stages"][0]["stage_path"]
    stage = json.loads(first_stage.read_text(encoding="utf-8"))
    first_image = repo_root / stage["cards"][0]["image"]
    first_image.unlink()
    with pytest.raises(ValueError):
        sweep.run_sweep(repo_root=repo_root)
