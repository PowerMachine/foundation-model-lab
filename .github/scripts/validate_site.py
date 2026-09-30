#!/usr/bin/env python3
"""Validate a self-contained static GitHub Pages directory."""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_SITE_BYTES = 100 * 1024 * 1024
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".svg", ".txt", ".xml"}
BINARY_SUFFIXES = {".avif", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".webp", ".woff", ".woff2"}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".env",
    ".key",
    ".orig",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
PRIVATE_PATTERN = re.compile(
    r"/(?:home|Users|data)/[^\s<>'\"]+|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|"
    r"hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}"
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        keys = {"href"} if tag in {"a", "link"} else {"src"}
        if tag in {"img", "source"}:
            keys.add("srcset")
        for key, value in attrs:
            if key in keys and value:
                values = value.split(",") if key == "srcset" else [value]
                for item in values:
                    self.links.append((key, item.strip().split()[0]))


def _resolve_local(site: Path, document: Path, raw: str) -> tuple[Path | None, str | None]:
    split = urlsplit(raw)
    if split.scheme in {"http", "https", "mailto", "tel", "data"} or split.netloc:
        return None, None
    if split.scheme:
        return None, f"unsupported URL scheme in {raw!r}"
    path_text = unquote(split.path)
    if not path_text:
        return None, None
    if path_text.startswith("/"):
        return None, f"root-relative URL is not portable to project Pages: {raw!r}"
    target = (document.parent / path_text).resolve()
    try:
        target.relative_to(site)
    except ValueError:
        return None, f"URL escapes site directory: {raw!r}"
    if target.is_dir():
        target = target / "index.html"
    return target, None


def validate(site: Path) -> list[str]:
    errors: list[str] = []
    if not site.is_dir() or site.is_symlink():
        return [f"{site}: site directory is missing or symlinked"]
    if not (site / "index.html").is_file():
        errors.append(f"{site / 'index.html'}: entry point is required")

    entries = list(site.rglob("*"))
    symlinks = sorted(path for path in entries if path.is_symlink())
    files = sorted(path for path in entries if path.is_file() and not path.is_symlink())
    if symlinks:
        errors.append(f"{site}: symbolic links are forbidden: {symlinks}")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_SITE_BYTES:
        errors.append(f"{site}: total size exceeds 100 MiB")

    for path in files:
        relative = path.relative_to(site)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            errors.append(f"{relative}: file exceeds 20 MiB")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in {".env", ".DS_Store"}:
            errors.append(f"{relative}: private/model artifact type is forbidden")
        elif (
            path.suffix.lower() not in TEXT_SUFFIXES | BINARY_SUFFIXES and path.name != ".nojekyll"
        ):
            errors.append(f"{relative}: unsupported static artifact type")
        if any(part.startswith(".") for part in relative.parts) and relative != Path(".nojekyll"):
            errors.append(f"{relative}: hidden paths are not published")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            errors.append(f"{relative}: expected UTF-8 text: {error}")
            continue
        if PRIVATE_PATTERN.search(text):
            errors.append(f"{relative}: local path, token, or private-key material detected")
        if path.suffix.lower() != ".html":
            continue
        parser = LinkParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception as error:
            errors.append(f"{relative}: invalid HTML: {error}")
            continue
        for _, raw in parser.links:
            target, link_error = _resolve_local(site, path, raw)
            if link_error:
                errors.append(f"{relative}: {link_error}")
            elif target is not None and not target.is_file():
                errors.append(f"{relative}: broken local asset/link {raw!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=Path("site"))
    args = parser.parse_args(argv)
    site = args.site.resolve()
    errors = validate(site)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    files = [path for path in site.rglob("*") if path.is_file()]
    print(f"site OK: {len(files)} files, {sum(path.stat().st_size for path in files)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
