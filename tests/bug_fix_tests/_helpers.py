"""Shared helpers for the bug-fix tests: synthetic packets, temporary pcaps and real captures."""
import contextlib
import os
import functools
import pathlib
import signal
import tempfile
import unittest
import warnings

from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]

# All the data the tests read: real captures from the course (hands-on notebooks, Assignment 1,
# Assignment 3) and their label / reference files. The tests read no data from anywhere else
# (see tests/data/README.md).
CAPTURES_DIR = REPO_DIR / 'tests' / 'data'

# the Assignment 3 demo capture and its labels
DEMO_PCAP = CAPTURES_DIR / 'demo.pcap'
DEMO_LABELS = CAPTURES_DIR / 'demo.csv'

# Set this variable (to any non-empty value) to run only the synthetic tests: every test on a real
# capture is then skipped, before it parses anything. `python -m tests.bug_fix_tests
# --synthetic-only` sets it.
SYNTHETIC_ONLY_VARIABLE = 'NETML_SYNTHETIC_ONLY'

MAC_SRC = '00:00:00:00:00:01'
MAC_DST = '00:00:00:00:00:02'


def ether(**fields):
    """An Ethernet header with fixed MAC addresses.

    Fixed addresses keep scapy from looking them up on the host's network when the packet is
    built, which would make the tests slow and dependent on the machine.
    """
    return Ether(src=MAC_SRC, dst=MAC_DST, **fields)


def at_time(pkt, pkt_time):
    """Set the arrival time of a packet (seconds since the epoch) and return the packet."""
    pkt.time = pkt_time
    return pkt


def truncated(pkt, pkt_time, snaplen=200):
    """Return the packet as a capture with a snaplen of `snaplen` bytes would store it.

    A capture tool with a small snaplen saves only the first `snaplen` bytes of each packet, but
    records the packet's original length in the pcap record. scapy exposes that length as
    `wirelen`, which wrpcap() writes back to the record.
    """
    wire_bytes = bytes(pkt)
    captured = Ether(wire_bytes[:snaplen])
    captured.wirelen = len(wire_bytes)
    captured.time = pkt_time
    return captured


class TimeLimitedTestCase(unittest.TestCase):
    """A TestCase whose tests fail, instead of hanging, when they run longer than `time_limit`
    seconds.

    The limit uses SIGALRM, so it applies on Linux and macOS (on Windows the tests run without
    it). The environment variable NETML_TEST_TIME_SCALE multiplies every limit, e.g. 3 on a slow
    machine.
    """

    time_limit = 60  # seconds

    def setUp(self):
        super().setUp()
        if not hasattr(signal, 'SIGALRM'):
            return
        limit = int(self.time_limit * float(os.environ.get('NETML_TEST_TIME_SCALE', '1')))

        def _too_slow(_signum, _frame):
            raise TimeoutError(f'{self.id()} took longer than its time limit of {limit} s')

        previous_handler = signal.signal(signal.SIGALRM, _too_slow)
        signal.alarm(limit)
        self.addCleanup(signal.signal, signal.SIGALRM, previous_handler)
        self.addCleanup(signal.alarm, 0)


class RealCaptureTestCase(TimeLimitedTestCase):
    """A test on real captures: parsing a capture of 100,000+ packets takes minutes.

    The whole class is skipped when $NETML_SYNTHETIC_ONLY is set. Subclasses that define
    setUpClass() must call super().setUpClass() first, so that nothing is parsed in that case.
    """

    time_limit = 15 * 60

    @classmethod
    def setUpClass(cls):
        if os.environ.get(SYNTHETIC_ONLY_VARIABLE):
            raise unittest.SkipTest(f'test on real captures ({SYNTHETIC_ONLY_VARIABLE} is set: '
                                    'synthetic tests only)')
        super().setUpClass()


class TemporaryPcapTestCase(TimeLimitedTestCase):
    """A TestCase with a temporary directory, removed after each test, for writing pcaps."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def write_pcap(self, packets, name='test.pcap'):
        """Write the packets to a pcap in the temporary directory and return its path."""
        path = os.path.join(self._tmpdir.name, name)
        wrpcap(path, packets)
        return path


def find_capture(name):
    """Path of the real capture `name` in tests/data, or None if it is missing."""
    path = CAPTURES_DIR / name
    if path.is_file():
        return path
    return None


def require_capture(test, name):
    """Return the path of the real capture `name`, or skip the test if it is not available."""
    path = find_capture(name)
    if path is None:
        test.skipTest(f'{name} not found in {CAPTURES_DIR} (see tests/data/README.md)')
    return path


@functools.lru_cache(maxsize=None)
def capture_dataframe(path):
    """pcap2pandas() of a real capture, parsed once per test run and shared by the tests.

    The tests only read the result; a test that needs to change it must work on a copy.
    """
    from netml.pparser.parser import PCAP
    pcap = PCAP(str(path))
    with quiet():
        pcap.pcap2pandas()
    return pcap.df


@functools.lru_cache(maxsize=None)
def capture_flows(path, tcp_timeout=600, udp_timeout=600):
    """_pcap2flows() of a real capture (flows split by the timeouts only), parsed once per test
    run and shared by the tests."""
    from netml.pparser import parser
    with quiet():
        return parser._pcap2flows(str(path), 2, tcp_timeout=tcp_timeout, udp_timeout=udp_timeout)


def assert_same_values(test, actual, expected, what='values'):
    """Assert that two long sequences are equal, reporting how many positions differ.

    unittest's assertEqual() on two long lists that differ builds a line-by-line diff of both for
    its message, which takes practically forever for a capture of 100,000+ packets. This reports
    the number of mismatches and the first few of them instead.
    """
    actual = list(actual)
    expected = list(expected)
    test.assertEqual(len(actual), len(expected), f'number of {what}')
    mismatches = [(i, a, e) for (i, (a, e)) in enumerate(zip(actual, expected)) if a != e]
    if mismatches:
        first = ', '.join(f'#{i}: {a!r} != {e!r}' for (i, a, e) in mismatches[:5])
        test.fail(f'{len(mismatches)} of {len(expected)} {what} differ (first: {first})')


@contextlib.contextmanager
def quiet():
    """Hide the warnings that scapy and pandas print while parsing (not the test's subject)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
