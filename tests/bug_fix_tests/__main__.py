"""Run the bug-fix tests.

From the repository root:

    python -m tests.bug_fix_tests                     the tests of fixes 1-4 (about 2 minutes)
    python -m tests.bug_fix_tests --synthetic-only    only the synthetic tests (a few seconds)
    python -m tests.bug_fix_tests test_3_ipv6 -v      one module, one line per test
    python -m tests.bug_fix_tests --isolation-forest  only the Isolation Forest regression check

The Isolation Forest tests (test_5_isolation_forest) are skipped by default: that bug was fixed
upstream (noise-lab/netml#40), and they only check that the fix still holds.

The exit status is 0 if all the tests that ran passed, 1 otherwise.
"""
import argparse
import os
import sys
import unittest

from ._helpers import ISOLATION_FOREST_VARIABLE, REPO_DIR, SYNTHETIC_ONLY_VARIABLE

PACKAGE = 'tests.bug_fix_tests'
ISOLATION_FOREST_MODULE = 'test_5_isolation_forest'


def main(argv=None):
    arg_parser = argparse.ArgumentParser(
        prog=f'python -m {PACKAGE}',
        description='Run the tests of the four netml bug fixes (and of the upstream IF fix).')
    arg_parser.add_argument(
        '--synthetic-only', '--quick', action='store_true',
        help='run only the tests on synthetic pcaps, and skip every test on a real capture '
             f'(sets {SYNTHETIC_ONLY_VARIABLE})')
    arg_parser.add_argument(
        '--isolation-forest', action='store_true',
        help='run only the Isolation Forest regression check (fixed upstream in '
             f'noise-lab/netml#40; skipped by default; sets {ISOLATION_FOREST_VARIABLE})')
    arg_parser.add_argument(
        '-v', '--verbose', action='store_true', help='print one line per test')
    arg_parser.add_argument(
        'modules', nargs='*', metavar='MODULE',
        help='test modules to run, e.g. test_3_ipv6 (default: all)')
    args = arg_parser.parse_args(argv)

    if args.synthetic_only:
        os.environ[SYNTHETIC_ONLY_VARIABLE] = '1'

    # the Isolation Forest tests run only when asked for: with --isolation-forest, or by name
    modules = list(args.modules)
    if args.isolation_forest:
        modules = [ISOLATION_FOREST_MODULE]
    if ISOLATION_FOREST_MODULE in modules:
        os.environ[ISOLATION_FOREST_VARIABLE] = '1'

    loader = unittest.TestLoader()
    if modules:
        suite = loader.loadTestsFromNames([f'{PACKAGE}.{module}' for module in modules])
    else:
        suite = loader.discover(start_dir=os.path.dirname(os.path.abspath(__file__)),
                                top_level_dir=str(REPO_DIR))

    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
