# === IMPORT CONTRACTS ===
#
# Layers (highest rank -> lowest):
#   oney (2) -> acthull (1) -> root utilities (0)
#
# ALLOWED:
#   oney -> acthull, oney -> root, acthull -> root
#
# FORBIDDEN:
#   acthull -x-> oney  [test_acthull_does_not_import_oney]
#   root -x-> acthull  [test_root_does_not_import_acthull]
#   root -x-> oney     [test_root_does_not_import_oney]
# ================================================================
"""Import architecture tests for wraact."""

__docformat__ = "restructuredtext"

import ast
import importlib
from pathlib import Path

import wraact

_SRC = Path(__file__).parent.parent.parent / "src" / "wraact"


def _get_imports(path: Path) -> set[str]:
    """Return fully qualified module names imported by path."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _imports_module(path: Path, module: str) -> bool:
    """Return whether path imports module or one of its children."""
    return any(name == module or name.startswith(f"{module}.") for name in _get_imports(path))


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
    """Enforce the package layer boundaries documented above."""

    def test_acthull_does_not_import_oney(self):
        """The base acthull layer must not depend on its oney specialization."""
        violations: list[str] = [
            str(f.relative_to(_SRC))
            for f in (_SRC / "acthull").rglob("*.py")
            if _imports_module(f, "wraact.oney")
        ]
        assert not violations, f"acthull imports oney in: {violations}"

    def test_root_utilities_do_not_import_higher_layers(self):
        """Root wraact modules must not import from higher-layer modules."""
        violations: list[str] = [
            str(f.relative_to(_SRC))
            for f in _SRC.glob("*.py")
            if f.name != "__init__.py"
            and (_imports_module(f, "wraact.acthull") or _imports_module(f, "wraact.oney"))
        ]
        assert not violations, f"root utilities import higher layers in: {violations}"
