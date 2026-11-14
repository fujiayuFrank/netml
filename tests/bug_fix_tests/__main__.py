"""Run the bug-fix tests.

From the repository root:

    python -m tests.bug_fix_tests                     all tests (about 2 minutes)
    python -m tests.bug_fix_tests --synthetic-only    only the synthetic tests (a few seconds)
    python -m tests.bug_fix_tests test_3_ipv6 -v      one module, one line per test

The exit status is 0 if all the tests that ran passed, 1 otherwise.
"""
import argparse
import os
import sys
import unittest

from ._helpers import REPO_DIR, SYNTHETIC_ONLY_VARIABLE

PACKAGE = 'tests.bug_fix_tests'


def main(argv=None):
    arg_parser = argparse.ArgumentParser(
        prog=f'python -m {PACKAGE}',
        description='Run the tests of the five netml bug fixes.')
    arg_parser.add_argument(
        '--synthetic-only', '--quick', action='store_true',
        help='run only the tests on synthetic pcaps, and skip every test on a real capture '
             f'(sets {SYNTHETIC_ONLY_VARIABLE})')
    arg_parser.add_argument(
        '-v', '--verbose', action='store_true', help='print one line per test')
    arg_parser.add_argument(
        'modules', nargs='*', metavar='MODULE',
        help='test modules to run, e.g. test_3_ipv6 (default: all)')
    args = arg_parser.parse_args(argv)

    if args.synthetic_only:
        os.environ[SYNTHETIC_ONLY_VARIABLE] = '1'

    loader = unittest.TestLoader()
    if args.modules:
        suite = loader.loadTestsFromNames([f'{PACKAGE}.{module}' for module in args.modules])
    else:
        suite = loader.discover(start_dir=os.path.dirname(os.path.abspath(__file__)),
                                top_level_dir=str(REPO_DIR))

    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
