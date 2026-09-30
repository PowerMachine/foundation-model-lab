#!/usr/bin/env python3
"""Validate GitHub workflows/forms and public-repository governance metadata."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml


OFFICIAL_ACTION = re.compile(r"actions/[A-Za-z0-9_.-]+@v[1-9][0-9]*")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ISSUE_ITEM_TYPES = {"checkboxes", "dropdown", "input", "markdown", "textarea"}
REQUIRED_ROOT_FILES = {
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "SECURITY.md",
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _load_yaml(path: Path, audit: Audit) -> dict[str, Any] | None:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        audit.errors.append(f"{path}: invalid YAML: {error}")
        return None
    if not isinstance(value, dict):
        audit.errors.append(f"{path}: top-level YAML must be a mapping")
        return None
    return value


def _walk_uses(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uses" and isinstance(item, str):
                yield item
            yield from _walk_uses(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_uses(item)


def _validate_workflow(path: Path, data: dict[str, Any], audit: Audit) -> None:
    text = path.read_text(encoding="utf-8")
    audit.require("pull_request_target" not in text, f"{path}: pull_request_target is forbidden")
    audit.require("secrets." not in text, f"{path}: repository secrets are not required by CI")
    audit.require(isinstance(data.get("on"), dict), f"{path}: trigger mapping is required")
    permissions = data.get("permissions")
    audit.require(
        isinstance(permissions, dict) and permissions.get("contents") == "read",
        f"{path}: top-level contents permission must be read-only",
    )
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        audit.errors.append(f"{path}: jobs mapping is required")
        return
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            audit.errors.append(f"{path}: job {job_name!r} must be a mapping")
            continue
        audit.require("runs-on" in job, f"{path}: job {job_name!r} has no runner")
        audit.require("timeout-minutes" in job, f"{path}: job {job_name!r} has no timeout")
        job_permissions = job.get("permissions", {})
        if isinstance(job_permissions, dict):
            writes = {key for key, value in job_permissions.items() if value == "write"}
            if writes:
                audit.require(
                    path.name == "pages.yml"
                    and job_name == "deploy"
                    and writes == {"id-token", "pages"},
                    f"{path}: unexpected write permissions in job {job_name!r}: {writes}",
                )
        else:
            audit.errors.append(f"{path}: job {job_name!r} permissions must be a mapping")
    for action in _walk_uses(data):
        audit.require(
            OFFICIAL_ACTION.fullmatch(action) is not None,
            f"{path}: action must be first-party and major-version pinned: {action!r}",
        )


def _validate_issue_form(path: Path, data: dict[str, Any], audit: Audit) -> None:
    audit.require(bool(data.get("name")), f"{path}: name is required")
    audit.require(bool(data.get("description")), f"{path}: description is required")
    body = data.get("body")
    if not isinstance(body, list) or not body:
        audit.errors.append(f"{path}: body must be a non-empty list")
        return
    ids: set[str] = set()
    for index, item in enumerate(body):
        if not isinstance(item, dict):
            audit.errors.append(f"{path}: body[{index}] must be a mapping")
            continue
        item_type = item.get("type")
        audit.require(item_type in ISSUE_ITEM_TYPES, f"{path}: invalid body type {item_type!r}")
        if item_type == "markdown":
            continue
        item_id = item.get("id")
        valid_id = isinstance(item_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", item_id)
        audit.require(bool(valid_id), f"{path}: body[{index}] has an invalid id")
        if valid_id:
            audit.require(item_id not in ids, f"{path}: duplicate id {item_id!r}")
            ids.add(item_id)
        attributes = item.get("attributes")
        audit.require(
            isinstance(attributes, dict) and bool(attributes.get("label")),
            f"{path}: body[{index}] needs an attributes.label",
        )


def _validate_dependabot(path: Path, data: dict[str, Any], audit: Audit) -> None:
    audit.require(data.get("version") == "2", f"{path}: Dependabot version must be 2")
    updates = data.get("updates")
    if not isinstance(updates, list):
        audit.errors.append(f"{path}: updates must be a list")
        return
    ecosystems = {item.get("package-ecosystem") for item in updates if isinstance(item, dict)}
    audit.require(
        {"github-actions", "pip"}.issubset(ecosystems),
        f"{path}: pip and github-actions updates are required",
    )


def _validate_markdown_links(path: Path, audit: Audit) -> None:
    text = path.read_text(encoding="utf-8")
    for target in MARKDOWN_LINK.findall(text):
        target = target.strip("<>")
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        local = target.split("#", 1)[0]
        audit.require((path.parent / local).exists(), f"{path}: broken relative link {target!r}")


def validate(root: Path) -> Audit:
    audit = Audit()
    for name in REQUIRED_ROOT_FILES:
        audit.require(
            (root / name).is_file(), f"{root / name}: required repository file is missing"
        )

    workflows = sorted((root / ".github/workflows").glob("*.yml"))
    audit.require(bool(workflows), f"{root / '.github/workflows'}: no workflows found")
    for path in workflows:
        data = _load_yaml(path, audit)
        if data is not None:
            _validate_workflow(path, data, audit)

    forms = sorted((root / ".github/ISSUE_TEMPLATE").glob("*.yml"))
    audit.require(bool(forms), f"{root / '.github/ISSUE_TEMPLATE'}: no issue forms found")
    for path in forms:
        data = _load_yaml(path, audit)
        if data is not None and path.name != "config.yml":
            _validate_issue_form(path, data, audit)

    dependabot = root / ".github/dependabot.yml"
    data = _load_yaml(dependabot, audit)
    if data is not None:
        _validate_dependabot(dependabot, data, audit)

    markdown_files = [
        root / "CODE_OF_CONDUCT.md",
        root / "CONTRIBUTING.md",
        root / "SECURITY.md",
        root / ".github/pull_request_template.md",
    ]
    for path in markdown_files:
        if path.is_file():
            _validate_markdown_links(path, audit)

    makefile = (
        (root / "Makefile").read_text(encoding="utf-8") if (root / "Makefile").is_file() else ""
    )
    for target in {"check", "evidence", "format-check", "lint", "site-check", "test", "toy"}:
        audit.require(
            re.search(rf"(?m)^{re.escape(target)}\s*:", makefile) is not None,
            f"{root / 'Makefile'}: missing target {target!r}",
        )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    audit = validate(args.root.resolve())
    if audit.errors:
        for error in audit.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("repository metadata OK: workflows, forms, governance, links, and Make targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
