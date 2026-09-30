from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys

import pytest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BUILDER_PATH = ROOT / "scripts" / "build_portfolio_site.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("portfolio_site_builder", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = BUILDER
BUILDER_SPEC.loader.exec_module(BUILDER)
TRACKS = BUILDER.TRACKS
build_site = BUILDER.build_site
PRIVATE_OR_SECRET = re.compile(
    rb"(?:/home/|/data/|/Users/|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|"
    rb"sk-[A-Za-z0-9_-]{20,}|BEGIN [A-Z ]+PRIVATE KEY)",
    re.IGNORECASE,
)
UNSAFE_SVG = re.compile(
    r"(?:<script\b|\bon(?:load|error|click)\s*=|"
    r"(?:href|src)\s*=\s*['\"]https?://)",
    re.IGNORECASE,
)


class MarkupAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.images_without_alt: list[str] = []
        self.h1_count = 0
        self.track_cards = 0
        self.claim_boundaries = 0
        self.i18n_nodes = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        for name in ("href", "src"):
            value = attributes.get(name)
            if value:
                self.references.append(value)
        if tag == "img" and not attributes.get("alt"):
            self.images_without_alt.append(self.get_starttag_text())
        if tag == "h1":
            self.h1_count += 1
        classes = set((attributes.get("class") or "").split())
        if tag == "article" and "track-card" in classes:
            self.track_cards += 1
        if "claim-boundary" in classes:
            self.claim_boundaries += 1
        if "data-i18n" in attributes:
            self.i18n_nodes += 1


def _files(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_checked_in_site_is_deterministic(tmp_path: Path) -> None:
    build_site(tmp_path)
    assert _files(tmp_path) == _files(SITE)


def test_every_html_reference_is_local_and_resolves() -> None:
    for html_path in SITE.rglob("*.html"):
        parser = MarkupAudit()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for reference in parser.references:
            if reference.startswith("#"):
                assert reference[1:] in parser.ids, (html_path, reference)
                continue
            assert not reference.startswith(("/", "\\")), (html_path, reference)
            assert not re.match(r"^[a-z][a-z0-9+.-]*:", reference, re.IGNORECASE), (
                html_path,
                reference,
            )
            local = unquote(reference.split("#", 1)[0].split("?", 1)[0])
            parts = Path(local).parts
            assert ".." not in parts, (html_path, reference)
            assert (html_path.parent / local).is_file(), (html_path, reference)


def test_index_is_accessible_bilingual_and_boundary_first() -> None:
    parser = MarkupAudit()
    source = (SITE / "index.html").read_text(encoding="utf-8")
    parser.feed(source)
    assert parser.h1_count == 1
    assert parser.track_cards == len(TRACKS) == 5
    assert parser.claim_boundaries == len(TRACKS)
    assert parser.i18n_nodes >= 70
    assert not parser.duplicate_ids
    assert not parser.images_without_alt
    assert '<meta http-equiv="Content-Security-Policy"' in source
    assert "prefers-reduced-motion" in (SITE / "styles.css").read_text(encoding="utf-8")
    assert 'aria-live="polite"' in source
    assert "Actual GPU smoke" in source
    assert "not quality gain or benchmark value" in source
    assert "source evidence bundles" in source
    assert "5 source + 1 aggregate manifests" in source
    assert "Aggregate view · not an experiment" in source


def test_headline_metrics_are_derived_from_public_results() -> None:
    source = (SITE / "index.html").read_text(encoding="utf-8")
    visual = json.loads((TRACKS[0].source_dir / "result.json").read_text())
    gpu = json.loads((TRACKS[1].source_dir / "result.json").read_text())
    ddp = json.loads((TRACKS[3].source_dir / "result.json").read_text())
    systems = json.loads((TRACKS[4].source_dir / "result.json").read_text())

    visual_metrics = visual["metrics"]
    naive = visual_metrics["preference_accuracy"]["naive"] * 100
    robust = visual_metrics["preference_accuracy"]["robust"] * 100
    assert f"{naive:.1f}% → {robust:.1f}%" in source

    gpu_metrics = gpu["metrics"]
    trainable_m = gpu_metrics["parameters"]["trainable_parameters"] / 1_000_000
    peak_gib = gpu_metrics["gpu_memory_training_peak"]["aggregate_peak_allocated_gib"]
    runtime = gpu_metrics["runtime_seconds"]["total"]
    assert f"{trainable_m:.2f}M" in source
    assert f"{peak_gib:.2f} GiB" in source
    assert f"{runtime:.2f} s total" in source
    assert "delta 0 · n=3 synthetic" in source

    gradient_error = ddp["metrics"]["parity"]["gradient"]["max_absolute_error"]
    assert f"{gradient_error:.2e}" in source

    simulation = systems["metrics"]["simulation"]
    static = simulation["static_fcfs"]
    continuous = simulation["continuous_fcfs"]
    throughput_delta = (
        continuous["output_token_throughput_per_second"]
        / static["output_token_throughput_per_second"]
        - 1
    )
    assert f"{throughput_delta * 100:+.1f}%" in source


def test_site_release_policy_and_svg_safety() -> None:
    files = [path for path in SITE.rglob("*") if path.is_file()]
    assert files
    assert all(not path.is_symlink() for path in SITE.rglob("*"))
    assert sum(path.stat().st_size for path in files) < 100 * 1024 * 1024
    assert all(path.stat().st_size < 20 * 1024 * 1024 for path in files)
    assert {path.name for path in files if path.name.startswith(".")} == {".nojekyll"}
    assert not list(SITE.rglob("*.orig"))
    assert not list(SITE.rglob("*.rej"))

    for path in files:
        payload = path.read_bytes()
        assert not PRIVATE_OR_SECRET.search(payload), path

    for path in SITE.rglob("*.svg"):
        source = path.read_text(encoding="utf-8")
        assert source.lstrip().startswith("<svg")
        assert not UNSAFE_SVG.search(source), path


def test_packaged_evidence_matches_verified_public_sources() -> None:
    for track in TRACKS:
        packaged = SITE / "evidence" / track.slug
        source_manifest = track.source_dir / "evidence_manifest.json"
        assert (packaged / "source_evidence_manifest.json").read_bytes() == (
            source_manifest.read_bytes()
        )
        assert (packaged / "result.json").read_bytes() == (
            track.source_dir / "result.json"
        ).read_bytes()
        site_manifest = json.loads((packaged / "site_manifest.json").read_text())
        assert (
            site_manifest["source_manifest_sha256"]
            == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        )
        assert isinstance(site_manifest["transformations"], list)
        if track.slug in {"agent-eval", "inference-systems"}:
            assert site_manifest["transformations"]
        assert (packaged / "report.html").is_file()
        assert (SITE / "source" / f"{track.slug}.py").is_file()

    scorecard_source = ROOT / "public-evidence" / "meta" / "portfolio-scorecard"
    scorecard_site = SITE / "meta" / "portfolio-scorecard"
    for name in ("scorecard.json", "scorecard.svg", "evidence_manifest.json"):
        assert (scorecard_site / name).read_bytes() == (scorecard_source / name).read_bytes()


@pytest.mark.parametrize("mutation", ["bytes", "sha256"])
def test_manifest_verification_fails_closed_on_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(TRACKS[0].source_dir, bundle)
    result = bundle / "result.json"
    payload = result.read_bytes()
    if mutation == "bytes":
        result.write_bytes(payload + b"\n")
        expected = "byte-count mismatch"
    else:
        result.write_bytes(b" " + payload[1:])
        expected = "SHA-256 mismatch"
    with pytest.raises(ValueError, match=expected):
        BUILDER._verify_evidence_manifest(bundle)
