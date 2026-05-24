# conftest.py — loaded by pytest before any test
import sys
import os

# ── Step 1: Add project root to the FRONT of sys.path ────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
else:
    # Make sure it's at the very front, not buried
    sys.path.remove(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)

# ── Step 2: Clear stale cached `src.*` modules ───────────────────────────────
# If another package created a `src` namespace before our project root was
# added (e.g., cage-challenge-4 editable install), the cached `src` module
# won't include our `src/env` subpackage. Clearing forces a fresh import.
_stale = [k for k in sys.modules if k == "src" or k.startswith("src.")]
for _k in _stale:
    del sys.modules[_k]

# ── Step 3: Validate that critical directories exist ─────────────────────────
_missing = []
for _subpkg in ["env", "attacks", "defense", "llm_backend", "benchmark"]:
    _init = os.path.join(_PROJECT_ROOT, "src", _subpkg, "__init__.py")
    if not os.path.isfile(_init):
        _missing.append(f"  src/{_subpkg}/__init__.py")

if _missing:
    raise RuntimeError(
        "\n[conftest] MISSING FILES — run `git pull origin v2` in Colab:\n"
        + "\n".join(_missing)
        + f"\n  Project root: {_PROJECT_ROOT}"
        + f"\n  sys.path[0]:  {sys.path[0]}"
    )
