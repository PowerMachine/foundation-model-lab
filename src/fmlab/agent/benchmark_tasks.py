from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .benchmark_types import FileAsset, HiddenCase, TaskSpec, TextPatch, sha256_json


@dataclass(frozen=True)
class _Template:
    task_id: str
    function: str
    instruction: str
    source: str
    old: str
    new: str
    public_assertions: tuple[str, ...]
    hidden: tuple[tuple[list[Any], Any], ...]
    tags: tuple[str, ...]


_TEMPLATES = (
    _Template(
        task_id="integer_addition",
        function="add",
        instruction="Fix add(a, b) so it returns the arithmetic sum for integers.",
        source="def add(a, b):\n    return a - b\n",
        old="return a - b",
        new="return a + b",
        public_assertions=("assert add(2, 3) == 5",),
        hidden=(([-4, 7], 3), ([0, 0], 0), ([-8, -2], -10), ([41, 1], 42)),
        tags=("arithmetic", "bug-fix"),
    ),
    _Template(
        task_id="bounded_clamp",
        function="clamp",
        instruction="Fix clamp(value, low, high) so values are bounded inclusively.",
        source=("def clamp(value, low, high):\n    return max(high, min(low, value))\n"),
        old="return max(high, min(low, value))",
        new="return max(low, min(high, value))",
        public_assertions=(
            "assert clamp(5, 0, 10) == 5",
            "assert clamp(-2, 0, 10) == 0",
        ),
        hidden=((([99, 0, 10]), 10), (([3, 3, 3]), 3), (([-5, -4, 8]), -4)),
        tags=("boundary", "bug-fix"),
    ),
    _Template(
        task_id="stable_slug",
        function="slugify",
        instruction=(
            "Fix slugify(text) so surrounding whitespace is removed, words are lower-cased, "
            "runs of whitespace collapse, and words are joined with hyphens."
        ),
        source=('def slugify(text):\n    return text.strip().lower().replace(" ", "_")\n'),
        old='return text.strip().lower().replace(" ", "_")',
        new='return "-".join(text.strip().lower().split())',
        public_assertions=('assert slugify(" Hello World ") == "hello-world"',),
        hidden=(
            (["A   B"], "a-b"),
            (["already-clean"], "already-clean"),
            (["  Mixed\tSpacing\nHere "], "mixed-spacing-here"),
        ),
        tags=("text", "normalization"),
    ),
    _Template(
        task_id="order_independent_median",
        function="median",
        instruction="Fix median(values) so the result does not depend on input ordering.",
        source=(
            "def median(values):\n"
            "    values = list(values)\n"
            "    middle = len(values) // 2\n"
            "    if len(values) % 2:\n"
            "        return values[middle]\n"
            "    return (values[middle - 1] + values[middle]) / 2\n"
        ),
        old="values = list(values)",
        new="values = sorted(values)",
        public_assertions=("assert median([9, 1, 5]) == 5",),
        hidden=(
            ([[4, 1, 3, 2]], 2.5),
            ([[8]], 8),
            ([[-1, 10, 0]], 0),
            ([[100, 2, 4, 6, 8]], 6),
        ),
        tags=("statistics", "ordering"),
    ),
    _Template(
        task_id="robust_boolean_parser",
        function="parse_bool",
        instruction=(
            "Fix parse_bool(value) to ignore surrounding whitespace and accept true, 1, yes, "
            "or on case-insensitively."
        ),
        source='def parse_bool(value):\n    return value.lower() == "true"\n',
        old='return value.lower() == "true"',
        new='return value.strip().lower() in {"true", "1", "yes", "on"}',
        public_assertions=(
            'assert parse_bool(" TRUE ") is True',
            'assert parse_bool("false") is False',
        ),
        hidden=(
            (["yes"], True),
            (["On"], True),
            ([" 1 "], True),
            (["0"], False),
            (["off"], False),
        ),
        tags=("parsing", "edge-cases"),
    ),
    _Template(
        task_id="stable_unique",
        function="stable_unique",
        instruction="Fix stable_unique(values) so duplicates are removed without reordering.",
        source="def stable_unique(values):\n    return list(set(values))\n",
        old="return list(set(values))",
        new="return list(dict.fromkeys(values))",
        public_assertions=("assert stable_unique([3, 1, 3, 2]) == [3, 1, 2]",),
        hidden=(
            ([["b", "a", "b", "c", "a"]], ["b", "a", "c"]),
            ([[]], []),
            ([[1, 1, 1]], [1]),
            ([[0, -1, 0, 2]], [0, -1, 2]),
        ),
        tags=("collections", "ordering"),
    ),
)


def _build_task(template: _Template, perturbation_id: str) -> TaskSpec:
    if perturbation_id == "canonical":
        source_path = "solution.py"
        test_path = "test_public.py"
        source = template.source
        assertions = template.public_assertions
    elif perturbation_id == "renamed_module":
        source_path = "candidate_impl.py"
        test_path = "verify_public.py"
        source = "# Perturbation: renamed module and an irrelevant comment.\n" + template.source
        assertions = tuple(reversed(template.public_assertions))
    else:
        raise ValueError(f"unknown perturbation: {perturbation_id}")

    module = source_path.removesuffix(".py")
    public_test = (
        f"from {module} import {template.function}\n\n"
        + "\n".join(assertions)
        + '\nprint("PUBLIC_TESTS_PASSED")\n'
    )
    return TaskSpec(
        task_id=template.task_id,
        perturbation_id=perturbation_id,
        instruction=template.instruction,
        module=module,
        function=template.function,
        public_test_path=test_path,
        assets=(
            FileAsset(source_path, source, "source"),
            FileAsset(test_path, public_test, "public_test"),
        ),
        hidden_cases=tuple(HiddenCase.build(args, expected) for args, expected in template.hidden),
        reference_patch=TextPatch(source_path, template.old, template.new),
        tags=template.tags + (perturbation_id,),
    )


def build_task_suite() -> tuple[TaskSpec, ...]:
    tasks = tuple(
        _build_task(template, perturbation)
        for template in _TEMPLATES
        for perturbation in ("canonical", "renamed_module")
    )
    if len(tasks) != 12 or len({task.uid for task in tasks}) != len(tasks):
        raise RuntimeError("the reliable agent suite must contain 12 unique tasks")
    return tasks


def task_suite_fingerprint(tasks: tuple[TaskSpec, ...] | None = None) -> str:
    suite = tasks or build_task_suite()
    return sha256_json(
        {
            "suite": "reliable-agent-eval/v1",
            "tasks": [task.fingerprint for task in sorted(suite, key=lambda item: item.uid)],
        }
    )
