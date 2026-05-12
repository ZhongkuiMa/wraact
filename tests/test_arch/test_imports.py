# === INFERRED IMPORT CONTRACTS (review before committing) ===
#
# Layers (highest rank -> lowest):
#   acthull (2) -> oney (1), root (_functions, _tangent_lines, _constants, _exceptions) (0)
#
# ALLOWED (not tested):
#   acthull -> oney, acthull -> root, oney -> root
#
# FORBIDDEN (each becomes one test method below):
#   oney -x-> acthull  [test_oney_does_not_import_acthull]
#   root -x-> acthull  [test_root_does_not_import_acthull]
#   root -x-> oney     [test_root_does_not_import_oney]
#
# [REVIEW] Approve contracts before treating ARC3 as resolved.
#   Run /python-optimize-tests apply @wraact/ to fill full boundaries.
# ================================================================
"""Import architecture tests for wraact."""

__docformat__ = "restructuredtext"

import ast
import importlib
from pathlib import Path

import pytest

import wraact

_SRC = Path(__file__).parent.parent.parent / "src" / "wraact"


def _get_imports(path: Path) -> set[str]:
    """Return top-level module names imported by path."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestImportSmoke:
    """Package imports without circular dependencies or broken __init__ chains."""

    def test_top_level_import(self):
        """Import package top-level without error."""
        assert wraact.__name__ == "wraact"

    def test_submodule_imports_cleanly(self):
        """Every submodule imports without ImportError."""
        errors: list[str] = []
        for f in _SRC.rglob("*.py"):
            rel = f.relative_to(_SRC.parent).with_suffix("")
            mod = ".".join(rel.parts)
            try:
                importlib.import_module(mod)
            except ImportError as e:
                errors.append(f"{mod}: {e}")
        assert not errors, "Import errors:\n" + "\n".join(errors)


class TestLayerBoundaries:
    """Inferred layer boundary tests — see contract header above."""

    def test_oney_does_not_import_acthull(self):
        """Oney layer must not import from acthull layer."""
        violations: list[str] = [
            str(f.relative_to(_SRC))
            for f in (_SRC / "oney").rglob("*.py")
            if "acthull" in _get_imports(f)
        ]
        assert not violations, f"oney imports acthull in: {violations}"

    @pytest.mark.parametrize(
        "module_name",
        [
            pytest.param("acthull", id="acthull"),
            pytest.param("oney", id="oney"),
        ],
    )
    def test_root_does_not_import_module(self, module_name):
        """Root wraact modules must not import from higher-layer modules."""
        violations: list[str] = [
            str(f.relative_to(_SRC)) for f in _SRC.glob("*.py") if module_name in _get_imports(f)
        ]
        assert not violations, f"root imports {module_name} in: {violations}"
