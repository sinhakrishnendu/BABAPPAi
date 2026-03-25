# babappai/tree.py
# ============================================================
# BABAPPAi — TREE HANDLING UTILITIES
# ============================================================

from pathlib import Path
import html
import sys
import types


def _ensure_cgi_shim_for_py314() -> None:
    """Provide a minimal cgi shim for ete3 on Python versions where cgi is removed."""
    try:
        import cgi  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("cgi")
    shim.escape = html.escape
    sys.modules["cgi"] = shim


def load_tree(tree_path):
    """
    Load a phylogenetic tree from Newick format.

    Notes
    -----
    - Uses ete3 for tree parsing.
    - Explicitly checks for the 'six' dependency, which ete3
      requires at runtime but does not declare strictly.
    - Imported lazily to keep CLI startup lightweight.

    Parameters
    ----------
    tree_path : str or Path
        Path to Newick-formatted tree file.

    Returns
    -------
    ete3.Tree
        Parsed phylogenetic tree.
    """
    try:
        _ensure_cgi_shim_for_py314()
        import six  # required indirectly by ete3
        from ete3 import Tree
    except ImportError as e:
        raise ImportError(
            "Tree handling requires the 'ete3' package and its runtime "
            "dependency 'six'.\n\n"
            "On Python 3.14+, BABAPPAi installs a compatibility shim for "
            "the removed stdlib cgi module before importing ete3.\n\n"
            "Install them via:\n"
            "  pip install ete3 six\n"
        ) from e

    tree_path = Path(tree_path)
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    try:
        return Tree(str(tree_path), format=1)
    except Exception as e:
        raise ValueError(
            f"Failed to parse Newick tree: {tree_path}\n"
            "Ensure the file is valid Newick format."
        ) from e


def enumerate_branches(tree):
    """
    Enumerate non-root branches in a stable, reproducible order.

    Branch naming rules
    -------------------
    - Terminal branches use leaf names.
    - Internal branches get deterministic names based on traversal order.
    - Root is excluded by design.

    Parameters
    ----------
    tree : ete3.Tree
        Parsed phylogenetic tree.

    Returns
    -------
    list of str
        Stable list of branch identifiers.
    """
    branches = []
    internal_counter = 0

    for node in tree.traverse("preorder"):
        if node.is_root():
            continue

        if node.is_leaf():
            name = node.name
            if not name:
                raise ValueError(
                    "All leaf nodes must have names. "
                    "Unnamed taxa detected in tree."
                )
        else:
            internal_counter += 1
            name = f"internal_{internal_counter}"

        branches.append(name)

    return branches
