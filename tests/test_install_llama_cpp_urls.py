"""Tests for llama-cpp-python release URL construction."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "install_llama_cpp.sh"
PROVENANCE_JSON = (
    PROJECT_ROOT
    / "docs"
    / "runbooks"
    / "releases"
    / "2026-04-21-llama-cpp-python-0.3.20-b8772-release-provenance.json"
)


def _extract_script_constant(script_text: str, name: str) -> str:
    """Extract one double-quoted shell constant assignment from the install script."""
    match = re.search(rf'^{re.escape(name)}="([^"]*)"$', script_text, flags=re.MULTILINE)
    assert match is not None, f"Missing shell constant: {name}"
    return match.group(1)


def _load_script_constants() -> dict[str, str]:
    """Load release-related constants from the install script."""
    script_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    names = (
        "GITHUB_RELEASE_DOWNLOAD_BASE",
        "LEGACY_RELEASE_TAG",
        "LEGACY_WHEEL_FILENAME",
        "B8772_RELEASE_TAG",
        "B8772_WHEEL_FILENAME",
    )
    return {name: _extract_script_constant(script_text, name) for name in names}


def test_install_script_exposes_expected_release_constants() -> None:
    """Release constants should keep legacy cp310 and b8772 py3-none filenames distinct."""
    constants = _load_script_constants()

    assert constants["GITHUB_RELEASE_DOWNLOAD_BASE"] == (
        "https://github.com/OWNER/ondevice-ai-smart-speaker-mung-ee/releases/download"
    )
    assert constants["LEGACY_RELEASE_TAG"] == "v0.3.17-llama"
    assert (
        constants["LEGACY_WHEEL_FILENAME"]
        == "llama_cpp_python-0.3.17-cp310-cp310-linux_aarch64.whl"
    )
    assert constants["B8772_RELEASE_TAG"] == "v0.3.20-llama-b8772"
    assert constants["B8772_WHEEL_FILENAME"] == "llama_cpp_python-0.3.20-py3-none-linux_aarch64.whl"


@pytest.mark.parametrize(
    ("tag_constant", "wheel_constant", "expected_suffix"),
    [
        (
            "LEGACY_RELEASE_TAG",
            "LEGACY_WHEEL_FILENAME",
            "/v0.3.17-llama/llama_cpp_python-0.3.17-cp310-cp310-linux_aarch64.whl",
        ),
        (
            "B8772_RELEASE_TAG",
            "B8772_WHEEL_FILENAME",
            "/v0.3.20-llama-b8772/llama_cpp_python-0.3.20-py3-none-linux_aarch64.whl",
        ),
    ],
)
def test_install_script_constructs_release_download_urls(
    tag_constant: str,
    wheel_constant: str,
    expected_suffix: str,
) -> None:
    """Reconstructed release URLs should match both install modes without network access."""
    constants = _load_script_constants()

    release_url = (
        f"{constants['GITHUB_RELEASE_DOWNLOAD_BASE']}/"
        f"{constants[tag_constant]}/"
        f"{constants[wheel_constant]}"
    )

    assert release_url.endswith(expected_suffix)
    assert release_url == (
        f"https://github.com/OWNER/ondevice-ai-smart-speaker-mung-ee/releases/download{expected_suffix}"
    )


def test_b8772_script_wheel_filename_matches_provenance_json() -> None:
    """The b8772 script wheel filename should match the release provenance."""
    constants = _load_script_constants()
    provenance: object = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))

    assert isinstance(provenance, dict)
    wheel_filename = provenance.get("wheel_filename")
    assert isinstance(wheel_filename, str)
    assert constants["B8772_WHEEL_FILENAME"] == wheel_filename


def test_b8772_url_uses_provenance_release_tag() -> None:
    """The b8772 URL should combine the script tag with the provenance wheel filename."""
    constants = _load_script_constants()
    provenance: object = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))

    assert isinstance(provenance, dict)
    wheel_filename = provenance.get("wheel_filename")
    assert isinstance(wheel_filename, str)

    release_url = (
        f"{constants['GITHUB_RELEASE_DOWNLOAD_BASE']}/"
        f"{constants['B8772_RELEASE_TAG']}/"
        f"{wheel_filename}"
    )

    assert release_url.endswith(
        "/v0.3.20-llama-b8772/llama_cpp_python-0.3.20-py3-none-linux_aarch64.whl"
    )
