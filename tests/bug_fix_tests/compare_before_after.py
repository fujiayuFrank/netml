"""Compare netml before and after the bug fixes.

before_bug_fix/ (in the repository root) holds the original code (git commit 106d5a1: upstream
netml 0.7.1) and, in before_bug_fix/tests/bug_fix_tests/, a copy of test_my_changes.py. This script runs the same things against both versions, each in its own
Python process (PYTHONPATH=<version>/src, so the version's own netml is imported):

1. Behaviour cases: one small input per bug (mostly synthetic pcaps built here), and what each
   version outputs for it, side by side.
2. Tests: test_my_changes.py against each version, and which tests change outcome.

Use the netml development environment (conda activate netml-dev): the workers use the same
Python as this script.

Run with:
    python tests/bug_fix_tests/compare_before_after.py               (cases and tests, ~3.5 min)
    python tests/bug_fix_tests/compare_before_after.py --cases-only  (only the cases, about 30 s)
    python tests/bug_fix_tests/compare_before_after.py --report tests/bug_fix_tests/compare_report.md
                                                          (also save a markdown report)
(from the repository root; the paths work from any directory)
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
# repository root: this file is in <root>/tests/bug_fix_tests/
REPO_DIR = os.path.dirname(os.path.dirname(HERE))
# test_my_changes.py, relative to the root of each version
TEST_FILE = os.path.join('tests', 'bug_fix_tests', 'test_my_changes.py')
VERSIONS = {
    'before': os.path.join(REPO_DIR, 'before_bug_fix'),
    'after': REPO_DIR,
}

V4_A, V4_B, V4_C = '10.0.0.1', '10.0.0.2', '10.0.0.3'
V6_A, V6_B, V6_C = '2001:db8::1', '2001:db8::2', '2001:db8::3'
MAC_SRC, MAC_DST = '00:00:00:00:00:01', '00:00:00:00:00:02'
SNAPLEN = 200


# ---------------------------------------------------------------------------
# synthetic pcaps (built once, in the parent process, and shared by both versions)
# ---------------------------------------------------------------------------

def build_pcaps(out_dir):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # e.g. CryptographyDeprecationWarning from scapy's ipsec
        from scapy.all import (ARP, DNS, DNSQR, GRE, ICMP, IP, TCP, UDP, Dot1Q, Ether,
                               ICMPv6EchoRequest, IPv6, Raw, wrpcap)

    def ether():
        return Ether(src=MAC_SRC, dst=MAC_DST)

    def at(pkt, t):
        pkt.time = t
        return pkt

    def truncated(pkt, t):
        """As captured with a small snaplen: the first SNAPLEN bytes, and the wire length."""
        wire = bytes(pkt)
        captured = Ether(wire[:SNAPLEN])
        captured.wirelen = len(wire)
        captured.time = t
        return captured

    pcaps = {}

    def write(name, packets):
        path = os.path.join(out_dir, name)
        wrpcap(path, packets)
        pcaps[name] = path

    # IPv4 TCP packets of 1054 bytes, recorded with snaplen 200
    write('snaplen.pcap', [
        truncated(ether() / IP(src=V4_A, dst=V4_B) / TCP(sport=1111, dport=80) / Raw(b'x' * 1000),
                  t)
        for t in (0, 0.01, 0.02)])

    # one IPv6 TCP connection and one IPv6 UDP connection
    packets = []
    for t in (0, 1, 2):
        packets.append(at(ether() / IPv6(src=V6_A, dst=V6_B) / TCP(sport=1000, dport=80), t))
        packets.append(at(ether() / IPv6(src=V6_C, dst=V6_A) / UDP(sport=5000, dport=53), t + 0.5))
    write('ipv6.pcap', packets)

    # one packet of each of several protocols
    write('protocols.pcap', [
        at(ether() / IP(src=V4_A, dst=V4_B) / TCP(), 0),
        at(ether() / IP(src=V4_A, dst=V4_B) / UDP(), 1),
        at(ether() / IP(src=V4_A, dst=V4_B) / ICMP(), 2),
        at(ether() / ARP(psrc=V4_A, pdst=V4_B), 3),
        at(ether() / IPv6(src=V6_A, dst=V6_B) / ICMPv6EchoRequest(), 4),
        at(ether() / IP(src=V4_A, dst=V4_B) / GRE(), 5),
    ])

    # a DNS query at a time with sub-second digits
    write('dns.pcap', [
        at(ether() / IP(src=V4_A, dst=V4_B) / UDP(sport=5353, dport=53)
           / DNS(rd=1, qd=DNSQR(qname='example.com')), 1700000000.123456),
    ])

    # one TCP connection idle for 700 s (longer than the 600 s timeout) in the middle
    write('idle_tcp.pcap', [at(ether() / IP(src=V4_A, dst=V4_B) / TCP(sport=1111, dport=80), t)
                            for t in (0, 1, 2, 702, 703, 704)])

    # a UDP flow from port 6, idle for 300 s in the middle
    write('udp_port6.pcap', [at(ether() / IP(src=V4_A, dst=V4_C) / UDP(sport=6, dport=53), t)
                             for t in (0, 1, 2, 302, 303, 304)])

    # VLAN-tagged TCP flow (SYN, ACK, PSH+ACK), IP flags DF
    write('vlan.pcap', [
        at(ether() / Dot1Q(vlan=10) / IP(src=V4_A, dst=V4_B, flags='DF')
           / TCP(sport=1111, dport=80, flags=f), t)
        for (t, f) in ((0, 'S'), (0.1, 'A'), (0.2, 'PA'))])

    # packets stored out of time order (the earliest is not the first)
    write('out_of_order.pcap', [at(ether() / IP(src=V4_A, dst=V4_B) / TCP(), t)
                                for t in (10, 5, 12)])

    # only ARP: no TCP/UDP flow at all
    write('arp_only.pcap', [at(ether() / ARP(psrc=V4_A, pdst=V4_B), t) for t in range(5)])

    return pcaps


# ---------------------------------------------------------------------------
# behaviour cases (run in a worker process, against one version)
# ---------------------------------------------------------------------------

def _error(exc):
    msg = ' '.join(str(exc).split())
    if len(msg) > 160:
        msg = msg[:157] + '...'
    return f'{type(exc).__name__}: {msg}'


def _invoke_cli(*argv):
    """Run the netml CLI in-process. Returns (exit code, stdout, stderr)."""
    import argcmdr
    from netml import cli

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            # argcmdr prints "[ErrorType] message" to its outfile, bound to the real stdout
            argcmdr.main(cli.Main, argv=[str(a) for a in argv], outfile=stdout)
        except SystemExit as exc:
            code = exc.code or 0
    ansi = re.compile(r'\x1b\[[0-9;]*m')
    return code, ansi.sub('', stdout.getvalue()), ansi.sub('', stderr.getvalue())


def _cli_summary(*argv):
    """Exit code, plus the error message if the command failed."""
    (code, stdout, stderr) = _invoke_cli(*argv)
    if not code:
        return 'exit 0 (ok)'
    lines = [line.strip() for line in (stderr + '\n' + stdout).splitlines()]
    errors = [line for line in lines
              if line.startswith('[') or 'error' in line.lower() or 'warning:' in line.lower()]
    errors = [line for line in errors if 'ResourceWarning' not in line]
    message = errors[-1] if errors else (lines[-1] if lines else '')
    if len(message) > 160:
        message = message[:157] + '...'
    return f'exit {code}: {message}'


def _flow_times(flows):
    return [[round(float(pkt.time), 2) for pkt in pkts] for (_fid, pkts) in flows]


def make_cases(root, pcaps, tmp):
    """The cases: (key, bug, input, function returning the output as a short string)."""
    from netml.pparser import parser
    from netml.pparser.parser import PCAP
    from netml.utils.tool import load_data

    examples = os.path.join(root, 'examples', 'data')
    demo_pcap = os.path.join(examples, 'demo.pcap')
    normal_pcap = os.path.join(examples, 'srcIP_10.42.0.1', 'srcIP_10.42.0.1_normal.pcap')
    anomaly_pcap = os.path.join(examples, 'srcIP_10.42.0.1', 'srcIP_10.42.0.119_anomaly.pcap')

    def pandas_column(pcap_name, column):
        pcap = PCAP(pcaps[pcap_name])
        pcap.pcap2pandas()
        return pcap.df[column].tolist()

    def snaplen():
        return f'length column: {pandas_column("snaplen.pcap", "length")}'

    def ipv6_flows():
        flows = parser._pcap2flows(pcaps['ipv6.pcap'], 2)
        return f'{len(flows)} flow(s): {[fid for (fid, _pkts) in flows]}'

    def ipv6_header():
        pcap = PCAP(pcaps['ipv6.pcap'])
        pcap.pcap2flows(q_interval=0.9)
        pcap.flow2features('IAT', fft=False, header=True)
        return f'features shape {pcap.features.shape}'

    def protocols():
        return f'protocol column: {pandas_column("protocols.pcap", "protocol")}'

    def dns_columns():
        pcap = PCAP(pcaps['dns.pcap'])
        pcap.pcap2pandas()
        row = pcap.df.iloc[0]
        return (f'dns_query={row["dns_query"]!r}; time is {type(row["time"]).__name__}; '
                f'datetime={row["datetime"]}')

    def time_normed():
        times = [float(t) for t in pandas_column('out_of_order.pcap', 'time_normed')]
        return f'time_normed: {times}'

    def idle_split():
        flows = parser._pcap2flows(pcaps['idle_tcp.pcap'], 2, tcp_timeout=600)
        return f'packet times per flow: {_flow_times(flows)}'

    def interval_split():
        flows = parser._pcap2flows(pcaps['idle_tcp.pcap'], 2, tcp_timeout=10000)
        subflows = parser._flows2subflows(flows, interval=100)
        return f'packet times per subflow: {_flow_times(subflows)}'

    def udp_port6():
        flows = parser._pcap2flows(pcaps['udp_port6.pcap'], 2, tcp_timeout=600, udp_timeout=200)
        return f'packet times per flow: {_flow_times(flows)}'

    def vlan_header():
        flows = parser._pcap2flows(pcaps['vlan.pcap'], 2)
        features = parser._get_header_features(flows)
        return f'header features: {[float(v) for v in features[0]]}'

    def multi_pcap_learn():
        return _cli_summary('-v', '0', 'learn', '--pcap-normal', normal_pcap,
                            '--pcap-abnormal', anomaly_pcap, '-o', os.path.join(tmp, 'm-multi'))

    def classify_other_pcap():
        model = os.path.join(tmp, 'm-demo')
        learned = _cli_summary('-v', '0', 'learn', '-p', demo_pcap, '-o', model)
        if learned != 'exit 0 (ok)':
            return f'learn failed: {learned}'
        return _cli_summary('classify', '-m', model, '-p', normal_pcap)

    def classify_algorithm(algorithm):
        def run():
            model = os.path.join(tmp, f'm-{algorithm}')
            learned = _cli_summary('-v', '0', 'learn', '--algorithm', algorithm,
                                   '-p', demo_pcap, '-o', model)
            if learned != 'exit 0 (ok)':
                return f'learn failed: {learned}'
            (code, stdout, stderr) = _invoke_cli('classify', '--report-all', '-m', model,
                                                 '-p', demo_pcap)
            if code:
                return _cli_summary('classify', '--report-all', '-m', model, '-p', demo_pcap)
            tags = [line.split()[-1] for line in stdout.splitlines() if line.startswith('[')]
            return f'classified {len(tags)} flows: {tags.count("ANOMALY")} ANOMALY'
        return run

    def if_auc():
        # the same data for both versions (not netml features, which the other fixes change):
        # 200 normal points around 0 and 20 obvious anomalies around 6
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from netml.ndm.iforest import IF
        rng = np.random.RandomState(0)
        X = np.vstack([rng.normal(0, 1, (200, 5)), rng.normal(6, 1, (20, 5))])
        y = np.array([0] * 200 + [1] * 20)
        model = IF(random_state=42).fit(X)
        auc = roc_auc_score(y, model.decision_function(X))
        return f'AUC {auc:.3f} (1 = anomalies scored highest, 0.5 = chance, 0 = inverted)'

    def train_feature_file():
        features = os.path.join(tmp, 'features')
        extracted = _cli_summary('learn', 'extract', '-p', demo_pcap, '-f', features)
        if extracted != 'exit 0 (ok)':
            return f'extract failed: {extracted}'
        size = os.path.getsize(features)
        trained = _cli_summary('learn', 'train', '-f', features, '-o', os.path.join(tmp, 'm-f'))
        return f'{trained}; features file {size} -> {os.path.getsize(features)} bytes'

    def no_flows():
        return _cli_summary('-v', '0', 'learn', '-p', pcaps['arp_only.pcap'],
                            '-o', os.path.join(tmp, 'm-arp'))

    def param_file():
        path = os.path.join(tmp, 'params.yaml')
        with open(path, 'w') as fh:
            fh.write('nu: 0.2\n')
        model = os.path.join(tmp, 'm-param')
        result = _cli_summary('-v', '0', 'learn', '--algorithm', 'ocsvm', '--param', path,
                              '-p', demo_pcap, '-o', model)
        if result != 'exit 0 (ok)':
            return result
        return f'{result}; model nu = {load_data(model)[0].nu}'

    def default_contamination():
        model = os.path.join(tmp, 'm-default')
        result = _cli_summary('-v', '0', 'learn', '--algorithm', 'ocsvm', '-p', demo_pcap,
                              '-o', model)
        if result != 'exit 0 (ok)':
            return result
        return f'OCSVM nu = {load_data(model)[0].nu}'

    def contamination_option():
        return _cli_summary('-v', '0', 'learn', '--contamination', '0.05', '-p', demo_pcap,
                            '-o', os.path.join(tmp, 'm-cont'))

    def capture_example():
        from scapy.all import IP, TCP, Ether, rdpcap
        path = os.path.join(root, 'examples', 'capture', 'capture_pkts.py')
        spec = importlib.util.spec_from_file_location('capture_pkts', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        packets = []
        for (t, dst) in ((0, V4_B), (1, V4_C), (4000, V4_B)):
            pkt = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src=V4_A, dst=dst) / TCP(sport=1111)
            pkt.time = t
            packets.append(pkt)
        module.sniff = lambda iface, count: packets
        out_file = os.path.join(tmp, 'capture.pcap')
        with contextlib.redirect_stdout(io.StringIO()):
            module.capture(iface='test', out_file=out_file)
        return f'ok, {len(rdpcap(out_file))} packets written'

    def committed_example_output():
        path = os.path.join(root, 'examples', 'out', 'data', 'OCSVM-results.dat')
        if not os.path.exists(path):
            return 'file removed (ndm_demo.py writes it)'
        load_data(path)
        return 'loads'

    return [
        ('snaplen', 'packet length with a small snaplen',
         '3 IPv4 TCP packets of 1054 bytes, recorded with snaplen 200', snaplen),
        ('ipv6_flows', 'IPv6 flows', '1 IPv6 TCP + 1 IPv6 UDP connection', ipv6_flows),
        ('ipv6_header', 'IPv6 header features', 'same pcap, flow2features(header=True)',
         ipv6_header),
        ('protocols', 'pcap2pandas protocol labels', 'TCP, UDP, ICMP, ARP, ICMPv6, GRE packets',
         protocols),
        ('dns_columns', 'pcap2pandas columns', 'DNS query for example.com at t=1700000000.123456',
         dns_columns),
        ('time_normed', 'pcap2pandas time_normed', 'packets at 10, 5, 12 s (earliest is not first)', time_normed),
        ('idle_split', 'idle-timeout split drops the last subflow',
         'TCP packets at 0,1,2,702,703,704 s; timeout 600 s', idle_split),
        ('interval_split', 'interval split drops the last subflow',
         'same packets, split into 100 s intervals', interval_split),
        ('udp_port6', '"6 in fid": UDP flow from port 6 treated as TCP',
         'UDP sport 6 at 0,1,2,302,303,304 s; tcp_timeout 600, udp_timeout 200', udp_port6),
        ('vlan_header', 'header features assume Ethernet/IPv4/TCP',
         'VLAN-tagged TCP flow: SYN, ACK, PSH+ACK', vlan_header),
        ('multi_pcap_learn', 'learn with several pcaps',
         'netml learn --pcap-normal normal.pcap --pcap-abnormal anomaly.pcap', multi_pcap_learn),
        ('classify_other_pcap', 'classify a pcap other than the training one',
         'learn on demo.pcap, classify srcIP_10.42.0.1_normal.pcap', classify_other_pcap),
        ('classify_if', 'classify with IF', 'learn + classify demo.pcap, --algorithm if',
         classify_algorithm('if')),
        ('classify_kde', 'classify with KDE', 'same, --algorithm kde', classify_algorithm('kde')),
        ('classify_gmm', 'classify with GMM', 'same, --algorithm gmm', classify_algorithm('gmm')),
        ('classify_pca', 'classify with PCA', 'same, --algorithm pca', classify_algorithm('pca')),
        ('if_auc', 'IF anomaly scores inverted', '200 normal points + 20 obvious anomalies (same data for both)', if_auc),
        ('train_feature_file', 'learn train -f FILE empties FILE',
         'learn extract -f F, then learn train -f F', train_feature_file),
        ('no_flows', 'pcap without flows', 'netml learn -p arp_only.pcap', no_flows),
        ('param_file', '--param FILE', '--param params.yaml containing "nu: 0.2"', param_file),
        ('default_contamination', 'OCSVM default nu (fraction of flows flagged)',
         'netml learn --algorithm ocsvm -p demo.pcap', default_contamination),
        ('contamination_option', 'learn --contamination', 'netml learn --contamination 0.05',
         contamination_option),
        ('capture_example', 'examples/capture/capture_pkts.py flow bookkeeping',
         'capture() on packets A@0 s, B@1 s, A@4000 s', capture_example),
        ('committed_example_output', 'shipped examples/out/data/OCSVM-results.dat',
         'load_data(...)', committed_example_output),
    ]


def run_cases(root, out_json, pcaps_json):
    pcaps = json.loads(pcaps_json)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for (key, bug, given, func) in make_cases(root, pcaps, tmp):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with contextlib.redirect_stdout(io.StringIO()), \
                            contextlib.redirect_stderr(io.StringIO()):
                        output = func()
            except Exception as exc:
                output = _error(exc)
            results.append({'key': key, 'bug': bug, 'input': given, 'output': output})
    with open(out_json, 'w') as fh:
        json.dump(results, fh)


# ---------------------------------------------------------------------------
# tests (run in a worker process, against one version)
# ---------------------------------------------------------------------------

class RecordingResult(unittest.TestResult):
    """Records one outcome per test: pass, fail, error or skip (with a short reason)."""

    def __init__(self):
        super().__init__()
        self.outcomes = {}
        self.class_errors = {}

    @staticmethod
    def _name(test):
        return f'{type(test).__name__}.{test._testMethodName}'

    @staticmethod
    def _reason(err):
        lines = traceback.format_exception_only(err[0], err[1])
        reason = ' '.join(' '.join(lines).split())
        return reason if len(reason) <= 200 else reason[:197] + '...'

    def _set(self, test, outcome, reason=''):
        name = self._name(test)
        # a failed subtest makes the test fail; keep the first reason
        if self.outcomes.get(name, ('pass',))[0] in ('pass',):
            self.outcomes[name] = (outcome, reason)

    def startTest(self, test):
        super().startTest(test)
        self.outcomes.setdefault(self._name(test), ('pass', ''))

    def addError(self, test, err):
        if not isinstance(test, unittest.TestCase):
            # setUpClass / module fixture failed: description is "setUpClass (module.Class)"
            class_name = str(test).split('(')[-1].rstrip(')').split('.')[-1]
            self.class_errors[class_name] = 'setUpClass: ' + self._reason(err)
            return
        self._set(test, 'error', self._reason(err))

    def addFailure(self, test, err):
        self._set(test, 'fail', self._reason(err))

    def addSkip(self, test, reason):
        self.outcomes[self._name(test)] = ('skip', reason)

    def addSubTest(self, test, subtest, err):
        if err is None:
            return
        outcome = 'fail' if issubclass(err[0], test.failureException) else 'error'
        self._set(test, outcome, f'{subtest._subDescription()}: {self._reason(err)}')


def _all_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _all_tests(item)
        else:
            yield item


def run_tests(root, out_json):
    path = os.path.join(root, TEST_FILE)
    spec = importlib.util.spec_from_file_location('test_my_changes', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['test_my_changes'] = module
    spec.loader.exec_module(module)

    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    tests = list(_all_tests(suite))
    result = RecordingResult()
    result.buffer = True  # hide the tests' own output
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        suite.run(result)

    outcomes = {}
    for test in tests:
        name = RecordingResult._name(test)
        class_name = type(test).__name__
        if class_name in result.class_errors:
            outcomes[name] = ('error', result.class_errors[class_name])
        else:
            outcomes[name] = result.outcomes.get(name, ('error', 'did not run'))
        doc = (test._testMethodDoc or '').strip().splitlines()
        outcomes[name] = list(outcomes[name]) + [doc[0] if doc else '']
    with open(out_json, 'w') as fh:
        json.dump(outcomes, fh)


# ---------------------------------------------------------------------------
# parent: run the workers, print the comparison
# ---------------------------------------------------------------------------

def _start_worker(version, mode, out_json, *extra):
    root = VERSIONS[version]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(root, 'src')  # this version's netml comes first
    env['PYTHONWARNINGS'] = 'ignore'
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), f'--worker-{mode}', root, out_json, *extra],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait(proc, what):
    stdout, stderr = proc.communicate()
    if proc.returncode < 0:
        import signal
        name = signal.Signals(-proc.returncode).name
        sys.exit(f'{what} crashed ({name}): a compiled package (numpy, pandas, ...) failed inside '
                 f'{sys.executable}.\nThis is a problem of that Python environment, not of netml: '
                 f'run the script with the netml development environment, e.g.\n'
                 f'    conda activate netml-dev && python tests/bug_fix_tests/compare_before_after.py\n'
                 f'{stderr[-3000:]}')
    if proc.returncode:
        sys.exit(f'{what} failed (exit {proc.returncode}):\n{stderr[-3000:]}')


REQUIRED_MODULES = ['argcmdr', 'netaddr', 'numpy', 'pandas', 'pyod', 'scapy', 'sklearn', 'yaml']


def _check_environment():
    """Stop early, with a clear message, if this Python lacks netml's dependencies."""
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(f'{sys.executable} lacks {", ".join(missing)}, needed by netml and its CLI.\n'
                 f'Run the script with the netml development environment, e.g.\n'
                 f'    conda activate netml-dev && python tests/bug_fix_tests/compare_before_after.py')


def _check_netml_location(version):
    """Make sure each worker really imports its own version of netml."""
    root = VERSIONS[version]
    env = dict(os.environ, PYTHONPATH=os.path.join(root, 'src'))
    location = subprocess.run([sys.executable, '-c', 'import netml; print(netml.__file__)'],
                              cwd=root, env=env, capture_output=True, text=True).stdout.strip()
    expected = os.path.join(root, 'src', 'netml', '__init__.py')
    if os.path.realpath(location) != os.path.realpath(expected):
        sys.exit(f'{version}: imports netml from {location!r}, expected {expected!r}')


def _md_cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def compare(cases_only, report):
    if not os.path.isdir(os.path.join(VERSIONS['before'], 'src', 'netml')):
        sys.exit(f'{VERSIONS["before"]}/src/netml is missing: export the original code first\n'
                 f'    git archive 106d5a1 | tar -x -C before_bug_fix   (in {REPO_DIR})')
    for version in VERSIONS:
        if not os.path.exists(os.path.join(VERSIONS[version], TEST_FILE)):
            sys.exit(f'{os.path.join(VERSIONS[version], TEST_FILE)} is missing (the "before" copy: '
                     f'cp {TEST_FILE} before_bug_fix/{TEST_FILE})')
    _check_environment()
    for version in VERSIONS:
        _check_netml_location(version)

    lines = []

    def out(text=''):
        print(text)
        lines.append(text)

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        pcaps = build_pcaps(tmp)
        procs = {}
        for version in VERSIONS:
            procs[('cases', version)] = _start_worker(
                version, 'cases', os.path.join(tmp, f'cases-{version}.json'), json.dumps(pcaps))
            if not cases_only:
                procs[('tests', version)] = _start_worker(
                    version, 'tests', os.path.join(tmp, f'tests-{version}.json'))
        print(f'running {len(procs)} workers (before_bug_fix/ and current code)...\n')
        for ((mode, version), proc) in procs.items():
            _wait(proc, f'{mode} against {version}')

        def load(name):
            with open(os.path.join(tmp, name)) as fh:
                return json.load(fh)

        cases = {version: load(f'cases-{version}.json') for version in VERSIONS}
        tests = None if cases_only else {version: load(f'tests-{version}.json')
                                         for version in VERSIONS}

    out('# netml before vs after the bug fixes')
    out()
    out(f'before: {VERSIONS["before"]} (original code, commit 106d5a1)  ')
    out(f'after:  {VERSIONS["after"]} (current code)')
    out()
    out('## 1. Behaviour on inputs that trigger each bug')
    out()
    changed = 0
    for (before, after) in zip(cases['before'], cases['after']):
        same = before['output'] == after['output']
        changed += not same
        out(f'### {before["bug"]}{"  (no change)" if same else ""}')
        out(f'- input:  {before["input"]}')
        out(f'- before: {before["output"]}')
        out(f'- after:  {after["output"]}')
        out()
    out(f'{changed} of {len(cases["before"])} cases give a different output.')

    if tests is not None:
        out()
        out(f'## 2. {TEST_FILE} against each version')
        out()
        for version in VERSIONS:
            counts = {}
            for (outcome, _reason, _doc) in tests[version].values():
                counts[outcome] = counts.get(outcome, 0) + 1
            summary = ', '.join(f'{counts.get(k, 0)} {k}' for k in ('pass', 'fail', 'error',
                                                                     'skip'))
            out(f'- {version}: {len(tests[version])} tests: {summary}')
        out()

        fixed = [name for name in tests['after']
                 if tests['after'][name][0] == 'pass'
                 and tests['before'].get(name, ['missing'])[0] in ('fail', 'error')]
        broken = [name for name in tests['after']
                  if tests['after'][name][0] in ('fail', 'error')
                  and tests['before'].get(name, ['missing'])[0] == 'pass']
        other = [name for name in tests['after']
                 if name not in fixed and name not in broken
                 and tests['after'][name][0] != tests['before'].get(name, ['missing'])[0]]
        unchanged_pass = [name for name in tests['after']
                          if tests['after'][name][0] == 'pass'
                          and tests['before'].get(name, ['missing'])[0] == 'pass']

        out(f'### Fail before, pass after ({len(fixed)} tests)')
        out()
        out('| test | before | why it failed before |')
        out('|---|---|---|')
        for name in fixed:
            (outcome, reason, _doc) = tests['before'][name]
            out(f'| {name} | {outcome} | {_md_cell(reason)} |')
        out()
        out(f'### Pass before, fail after ({len(broken)} tests)')
        out()
        for name in broken:
            out(f'- {name}: {tests["after"][name][1]}')
        if not broken:
            out('none')
        out()
        out(f'### Pass in both versions ({len(unchanged_pass)} tests)')
        out()
        out('Behaviour that was already right and is kept: '
            + (', '.join(unchanged_pass) if unchanged_pass else 'none'))
        if other:
            out()
            out(f'### Other changes ({len(other)} tests)')
            out()
            for name in other:
                before_outcome = tests['before'].get(name, ['missing'])[0]
                out(f'- {name}: {before_outcome} -> {tests["after"][name][0]} '
                    f'({tests["after"][name][1]})')

    print(f'\n(took {time.time() - started:.0f} s)')
    if report:
        with open(report, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f'report written to {os.path.abspath(report)}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-cases':
        run_cases(*sys.argv[2:5])
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-tests':
        with contextlib.redirect_stdout(io.StringIO()):
            run_tests(*sys.argv[2:4])
        return

    arg_parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    arg_parser.add_argument('--cases-only', action='store_true',
                            help='only the behaviour cases, not the test suite')
    arg_parser.add_argument('--report', metavar='FILE', help='also save the output as markdown')
    args = arg_parser.parse_args()
    compare(args.cases_only, args.report)


if __name__ == '__main__':
    main()
