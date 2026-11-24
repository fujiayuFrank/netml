# netml before the bug fixes

`src/netml/` here is the original netml library, **before** the four bug fixes: an unchanged copy of `src/netml/` from upstream netml master (noise-lab/netml, commit `ef55aef`). It already contains the Isolation Forest fix merged upstream in noise-lab/netml#40.

It is used only by `tests/bug_fix_tests/compare_before_after.py`, which runs the same tests and measurements against this code and against the fixed code in the repository's `src/`, each in its own Python process (`PYTHONPATH` decides which netml is imported), and reports the differences.

Do not edit it: it is the reference for "before". It is not a Python package of the test suite (no `__init__.py` here), so `python -m unittest` and `tox` do not look into it, and it is excluded from flake8 (it is a reference copy, not code of this project).

To recreate it from the git history:

    git archive ef55aef src/netml | tar -x -C tests/before_bug_fix/
