"""Compare netml before and after the five bug fixes.

Runs the same tests and measurements against two versions of netml, each in its own Python
process (PYTHONPATH decides which netml is imported):

    before   tests/before_bug_fix/src   the original code (upstream netml 0.7.1, commit 21ba115)
    after    src                        the fixed code of this repository

and prints, for each fix:

    1. the bug-fix tests (tests/bug_fix_tests/test_N_*.py): how many pass on each version;
    2. cases: a small input that triggers the bug (a synthetic pcap, or a real capture from
       tests/data), the output of each version, and why the original output is wrong. The cases on
       real captures are the before/after measurements (flows, packets and bytes lost, flows
       merged, packets mislabeled, ...).

Usage, from the repository root, in the netml development environment (conda activate netml-dev;
the workers use the same Python as this script):

    python -m tests.bug_fix_tests.compare_before_after                   everything (~5 minutes)
    python -m tests.bug_fix_tests.compare_before_after --synthetic-only  synthetic only (seconds)
    python -m tests.bug_fix_tests.compare_before_after --report FILE     also a Markdown report
    python -m tests.bug_fix_tests.compare_before_after --fix 1 3         only some fixes

The exit status is 1 if a case behaves unexpectedly (an output that should change did not, or one
that should not change did) or if a bug-fix test fails on the fixed code, and 0 otherwise.
"""
import argparse
import contextlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:  # (when run as a script rather than with -m)
    sys.path.insert(0, str(REPO_DIR))

from tests.bug_fix_tests import _helpers  # noqa: E402  (needs REPO_DIR on sys.path)

VERSIONS = {
    'before': REPO_DIR / 'tests' / 'before_bug_fix' / 'src',
    'after': REPO_DIR / 'src',
}
VERSION_LABELS = {
    'before': 'tests/before_bug_fix/src (upstream netml 0.7.1, commit 21ba115)',
    'after': 'src (this repository, with the five fixes)',
}

ENVIRONMENT_HINT = 'conda activate netml-dev && python -m tests.bug_fix_tests.compare_before_after'

# packages netml and the tests need (import names)
REQUIRED_MODULES = ['netaddr', 'numpy', 'pandas', 'pyod', 'scapy', 'sklearn']

CASE_TIME_LIMIT = 15 * 60  # seconds per case: a case that runs longer fails instead of hanging
PROGRESS_INTERVAL = 30     # seconds between "still running" lines

FIXES = {
    1: ('Packet size', 'test_1_packet_size',
        'Packet sizes were measured with len(pkt), the number of bytes saved in the pcap. A '
        'capture recorded with a small snaplen saves only the first bytes of each packet, so '
        'every larger packet was measured as the snaplen (e.g. 200 bytes).'),
    2: ('Flow splitting', 'test_2_flow_splitting',
        'Splitting a flow (by idle timeout in _pcap2flows, by interval in _flows2subflows) '
        'dropped its final subflow, and TCP/UDP was chosen with "6 in fid", which also matches '
        'port 6.'),
    3: ('IPv6', 'test_3_ipv6',
        'The network layer was looked up with scapy\'s IP class, which matches IPv4 only: IPv6 '
        'connections were merged under one placeholder flow id, IPv6 addresses were left empty, '
        'and header features crashed.'),
    4: ('pcap2pandas', 'test_4_pcap2pandas',
        'pcap2pandas() labeled only TCP/UDP/ICMP, stored DNS values as tuples (and crashed on '
        'invalid UTF-8), kept time as scapy EDecimal, cut datetime to whole seconds in local '
        'time, and measured time_normed from the first row instead of the earliest packet.'),
    5: ('Isolation Forest', 'test_5_isolation_forest',
        'netml\'s IF inverted pyod\'s anomaly scores, which pyod had already inverted, so normal '
        'flows got the highest scores and the AUC fell below 0.5.'),
}

# The cases, in the order they are printed. "change": whether the fix should change the output
# (False for controls, which must give the same output on both versions).
CASES = [
    # --- Fix 1 ---
    dict(key='lengths', fix=1, kind='synthetic', change=True,
         input='three 1054-byte TCP packets, saved at 200 bytes each (snaplen 200)',
         why='each packet is 1054 bytes on the wire; 200 is only how much of it was saved, so '
             'every packet size was under-reported by 854 bytes.'),
    dict(key='size_features', fix=1, kind='synthetic', change=True,
         input='the same flow: SIZE features and STATS total bytes',
         why='the features describe the 200-byte fragments in the file, not the traffic: the '
             'flow carried 3,162 bytes, not 600.'),
    dict(key='netflix_lengths', fix=1, kind='real', change=True,
         input='netflix.pcap (Assignment 1), recorded with a 200-byte snaplen, vs Wireshark',
         why='Wireshark reports the length on the wire, which the pcap records for every '
             'packet; the original sizes capped every large packet at 200 bytes and counted only '
             'a small fraction of the bytes really sent.'),
    dict(key='http_lengths', fix=1, kind='real', change=False,
         input='control: http.pcap (hands-on), recorded without a snaplen limit, vs Wireshark',
         why='(control: no packet was cut short, so the saved length is the length on the wire '
             'and both versions must agree with Wireshark)'),
    # --- Fix 2 ---
    dict(key='timeout_split', fix=2, kind='synthetic', change=True,
         input='one TCP flow with packets at 0, 1, 2, 702, 703, 704 s; timeout 600 s',
         why='the flow is idle for 700 s and is correctly split in two, but the second subflow '
             '(702-704 s) was then silently dropped.'),
    dict(key='interval_split', fix=2, kind='synthetic', change=True,
         input='one TCP flow with packets at 0-2, 12-14 and 24-26 s; interval 5 s',
         why='the flow is split in three, but the third subflow (24-26 s) was dropped.'),
    dict(key='udp_port_6', fix=2, kind='synthetic', change=True,
         input='a UDP flow from port 6, packets at 0, 1, 101, 102 s; TCP timeout 600 s, UDP 50 s',
         why='"6 in fid" matched the port, so this UDP flow got the TCP timeout (600 s) and the '
             '100-second gap did not split it.'),
    dict(key='log4j_timeout', fix=2, kind='real', change=True,
         input='log4j.pcap (hands-on): flows split by the 600 s timeout, vs an independent '
               'implementation of the same split',
         why='the original kept fewer flows and packets than the split itself produces: the '
             'final subflow of every split flow was lost.'),
    dict(key='demo_interval', fix=2, kind='real', change=True,
         input='demo.pcap (Assignment 3): pcap2flows(q_interval=0.9)',
         why='none of its subflows is too short to keep, so every packet should be kept; the '
             'missing packets are the dropped final subflows.'),
    dict(key='normal_interval', fix=2, kind='real', change=True,
         input='srcIP_10.42.0.1_normal.pcap (netml example): pcap2flows(q_interval=0.9)',
         why='the dropped final subflows: fewer subflows, packets and bytes reach the features '
             'and the model.'),
    # --- Fix 3 ---
    dict(key='ipv6_fid', fix=3, kind='synthetic', change=True,
         input='_get_fid() of one IPv6 TCP packet, 2001:db8::1:1000 -> 2001:db8::2:80',
         why='an IPv6 TCP packet has a real five-tuple; the placeholder means "not TCP/UDP".'),
    dict(key='ipv6_merge', fix=3, kind='synthetic', change=True,
         input='one IPv6 TCP connection and one unrelated IPv6 UDP connection',
         why='two unrelated connections were merged into one fake flow, mixing their packets '
             '(and their features).'),
    dict(key='ipv6_header', fix=3, kind='synthetic', change=True,
         input='header features (TCP flags, TTL) of the IPv6 TCP flow',
         why='IPv6 has no "proto" or "ttl" field (it has "nh" and "hlim"), so the original '
             'crashed on any IPv6 packet.'),
    dict(key='netflix_ipv6_flows', fix=3, kind='real', change=True,
         input='netflix.pcap (Assignment 1): IPv6 flows found by _pcap2flows()',
         why='all the IPv6 connections of the capture were merged into one placeholder flow.'),
    dict(key='netflix_ipv6_addresses', fix=3, kind='real', change=True,
         input='netflix.pcap: pcap2pandas() rows of the IPv6 packets with an address',
         why='IPv6 packets were left without source and destination addresses.'),
    dict(key='netflix_header', fix=3, kind='real', change=True,
         input='netflix.pcap: pcap2flows() then flow2features(header=True)',
         why='the IPv6 packets crashed the header features, so no features at all could be '
             'extracted from this capture.'),
    # --- Fix 4 ---
    dict(key='protocol_labels', fix=4, kind='synthetic', change=True,
         input='one ARP, ICMPv6, EAPOL, IGMP (IP protocol 2) and TCP packet',
         why='only TCP/UDP/ICMP were labeled: real, common protocols were reported as None.'),
    dict(key='dns_values', fix=4, kind='synthetic', change=True,
         input='a DNS response: example.com A 93.184.216.34',
         why='DNS values were 1-element tuples, which do not compare or group as strings '
             '(e.g. df.dns_query == "example.com." never matches).'),
    dict(key='dns_invalid', fix=4, kind='synthetic', change=True,
         input='a DNS query whose name is not valid UTF-8 (malformed traffic)',
         why='one malformed packet aborted the parse of the whole capture.'),
    dict(key='time_columns', fix=4, kind='synthetic', change=True,
         input='a packet received at 2022-11-03 16:01:02.794004 UTC',
         why='time was a column of scapy objects, and datetime lost the fraction of a second and '
             'was shifted to the local time zone of the machine running netml.'),
    dict(key='time_normed', fix=4, kind='synthetic', change=True,
         input='packets stored out of time order, at 62.794, 63.000, 62.5, 64.25 s',
         why='time_normed was measured from the first row, not the earliest packet, giving a '
             'negative time and a wrong origin.'),
    dict(key='protocol_counts', fix=4, kind='real', change=True,
         input='packets labeled None, per course capture',
         why='ARP, EAPOL, ICMPv6, IGMP, SCTP, ... packets were labeled None; the remaining None '
             'are protocols without any name.'),
    dict(key='dns_tuples', fix=4, kind='real', change=True,
         input='http.pcap (hands-on): DNS queries stored as tuples',
         why='every DNS query was a tuple instead of a string.'),
    dict(key='datetime_fraction', fix=4, kind='real', change=True,
         input='http.pcap: datetimes that lost the fraction of a second',
         why='every packet\'s datetime was cut to whole seconds (in local time), so packets '
             'within the same second could not be told apart or ordered by datetime.'),
    dict(key='time_normed_real', fix=4, kind='real', change=False,
         input='control: course captures whose first packet is not the earliest',
         why='(control: in every course capture the first packet is also the earliest, so '
             'time_normed is the same on both versions; the bug needs an out-of-order capture, '
             'shown by the synthetic case)'),
    # --- Fix 5 ---
    dict(key='demo_auc', fix=5, kind='real', change=True,
         input='IF on the labeled flows of demo.pcap (Assignment 3), on identical features',
         why='an AUC below 0.5 is worse than chance: the scores ranked normal flows as the most '
             'anomalous. The two AUCs add up to 1: the scores were turned the wrong way round.'),
]

V6_A, V6_B, V6_C = '2001:db8::1', '2001:db8::2', '2001:db8::3'
PLACEHOLDER_FID = ('', '', -1, -1, -1)


# ---------------------------------------------------------------------------------------------
# Inputs built once, in the parent process (scapy only, no netml)
# ---------------------------------------------------------------------------------------------

def build_synthetic_pcaps(out_dir):
    """Write the synthetic pcaps of the cases; return {name: path}."""
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    from scapy.layers.eap import EAPOL
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import ICMPv6EchoRequest, IPv6
    from scapy.layers.l2 import ARP, Ether
    from scapy.packet import Raw
    from scapy.utils import wrpcap

    ether = _helpers.ether
    at = _helpers.at_time

    def tcp(t, sport=1111):
        return at(ether() / IP(src='10.0.0.1', dst='10.0.0.2') / TCP(sport=sport, dport=80), t)

    def reread(pkt, t):
        return at(Ether(bytes(pkt)), t)

    pcaps = {
        'snaplen': [_helpers.truncated(ether() / IP(src='10.0.0.1', dst='10.0.0.2')
                                       / TCP(sport=1111, dport=80) / Raw(b'x' * 1000), t)
                    for t in (0, 0.01, 0.02)],
        'idle_tcp': [tcp(t) for t in (0, 1, 2, 702, 703, 704)],
        'interval_tcp': [tcp(t) for t in (0, 1, 2, 12, 13, 14, 24, 25, 26)],
        'udp_port6': [at(ether() / IP(src='10.0.0.1', dst='10.0.0.3') / UDP(sport=6, dport=53), t)
                      for t in (0, 1, 101, 102)],
        # an IPv6 TCP connection (SYN, ACK, PSH+ACK) interleaved with an unrelated IPv6 UDP one
        'ipv6': [p for (t, flags) in ((0, 'S'), (1, 'A'), (2, 'PA')) for p in (
            at(ether() / IPv6(src=V6_A, dst=V6_B, hlim=57)
               / TCP(sport=1000, dport=80, flags=flags), t),
            at(ether() / IPv6(src=V6_C, dst=V6_A, hlim=57) / UDP(sport=5000, dport=53), t + 0.5))],
        'protocols': [reread(ether() / ARP(), 1),
                      reread(ether() / IPv6() / ICMPv6EchoRequest(), 2),
                      reread(ether() / EAPOL(), 3),
                      reread(ether() / IP(proto=2) / Raw(b'\x11' * 8), 4),
                      reread(ether() / IP() / TCP(), 5)],
        'dns': [reread(ether() / IP(src='10.0.0.1', dst='10.0.0.3') / UDP(sport=53, dport=5353)
                       / DNS(qr=1, qd=DNSQR(qname='example.com'),
                             an=DNSRR(rrname='example.com', type='A', rdata='93.184.216.34')),
                       1)],
        'dns_invalid': [reread(ether() / IP(src='10.0.0.1', dst='10.0.0.3')
                               / UDP(sport=5353, dport=53)
                               / DNS(rd=1, qd=DNSQR(qname=b'\xff\xfe.example.com')), 1)],
        'times': [tcp(t) for t in (1667491262.794004, 1667491263.000001, 1667491262.5,
                                   1667491264.25)],
    }
    paths = {}
    for (name, packets) in pcaps.items():
        paths[name] = str(pathlib.Path(out_dir) / f'{name}.pcap')
        wrpcap(paths[name], packets)
    return paths


# ---------------------------------------------------------------------------------------------
# Worker: runs in its own process, against one version of netml
# ---------------------------------------------------------------------------------------------

def _capture(name):
    """Path of a capture in tests/data, or raise _Missing."""
    path = _helpers.find_capture(name)
    if path is None:
        raise _Missing(f'{name} is not in tests/data')
    return path


class _Missing(Exception):
    pass


def _times(flows):
    return [[round(float(pkt.time), 3) for pkt in pkts] for (_fid, pkts) in flows]


def _case_functions(ctx):
    """The functions computing each case's output (a short string), with this process's netml."""
    import numpy as np
    import pandas as pd
    from scapy.layers.inet import TCP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import Ether

    from netml.pparser import parser
    from netml.pparser.parser import PCAP

    def pandas_of(name):
        pcap = PCAP(ctx['pcaps'][name])
        pcap.pcap2pandas()
        return pcap.df

    def wire_lengths_vs_wireshark(pcap_name, csv_name):
        df = _helpers.capture_dataframe(_capture(pcap_name))
        reference = pd.read_csv(_capture(csv_name))['Length'].to_numpy()
        lengths = df['length'].astype(int).to_numpy()
        differ = int((lengths != reference).sum())
        return (f'{differ:,} of {len(reference):,} packet lengths differ from Wireshark; largest '
                f'{lengths.max():,} bytes; {lengths.sum():,} bytes in all (Wireshark: '
                f'{reference.sum():,})')

    def interval_split(pcap_path):
        raw = parser._pcap2flows(str(pcap_path), 2)
        pcap = PCAP(str(pcap_path))
        pcap.pcap2flows(q_interval=0.9)
        kept = sum(len(pkts) for (_fid, pkts) in pcap.flows)
        total = sum(len(pkts) for (_fid, pkts) in raw)
        wire = sum(_helpers_wirelen(p) for (_fid, pkts) in pcap.flows for p in pkts)
        return (f'{len(pcap.flows):,} subflows; {kept:,} of {total:,} packets kept '
                f'({wire:,} bytes)')

    def lengths():
        return f'length column: {pandas_of("snaplen")["length"].tolist()}'

    def size_features():
        flows = parser._pcap2flows(ctx['pcaps']['snaplen'], 2)
        (sizes, _fids) = parser._get_SIZE(flows)
        (stats, _fids) = parser._get_STATS(flows)
        return f'SIZE features {sizes[0]}; STATS total bytes {stats[0][11]:,.0f}'

    def netflix_lengths():
        return wire_lengths_vs_wireshark('netflix.pcap', 'netflix.csv.gz')

    def http_lengths():
        return wire_lengths_vs_wireshark('http.pcap', 'http.csv.gz')

    def timeout_split():
        flows = parser._pcap2flows(ctx['pcaps']['idle_tcp'], 2, tcp_timeout=600)
        return f'packet times per flow: {_times(flows)}'

    def interval_split_synthetic():
        flows = parser._pcap2flows(ctx['pcaps']['interval_tcp'], 2)
        return f'packet times per subflow: {_times(parser._flows2subflows(flows, 5))}'

    def udp_port_6():
        flows = parser._pcap2flows(ctx['pcaps']['udp_port6'], 2, tcp_timeout=600, udp_timeout=50)
        return f'packet times per flow: {_times(flows)}'

    def log4j_timeout():
        from tests.bug_fix_tests.test_2_flow_splitting import _reference_timeout_split
        path = _capture('log4j.pcap')
        flows = _helpers.capture_flows(path)
        (ref_flows, ref_packets) = _reference_timeout_split(path)
        packets = sum(len(pkts) for (_fid, pkts) in flows)
        return (f'{len(flows):,} flows, {packets:,} packets (the split itself: {ref_flows:,} '
                f'flows, {ref_packets:,} packets)')

    def demo_interval():
        return interval_split(_capture('demo.pcap'))

    def normal_interval():
        return interval_split(_capture('srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap'))

    def ipv6_fid():
        pkt = Ether() / IPv6(src=V6_A, dst=V6_B) / TCP(sport=1000, dport=80)
        return f'flow id {parser._get_fid(pkt)}'

    def ipv6_merge():
        flows = parser._pcap2flows(ctx['pcaps']['ipv6'], 2)
        return '; '.join(f'flow {fid}: {len(pkts)} packets' for (fid, pkts) in flows)

    def ipv6_header():
        flows = parser._pcap2flows(ctx['pcaps']['ipv6'], 2)
        return f'TCP flow: {[float(v) for v in parser._get_header_features(flows)[0]]}'

    def netflix_ipv6_flows():
        flows = _helpers.capture_flows(_capture('netflix.pcap'))
        ipv6 = [fid for (fid, _pkts) in flows if ':' in fid[0]]
        merged = [pkts for (fid, pkts) in flows if fid == PLACEHOLDER_FID]
        merged_text = (f'1 placeholder flow {PLACEHOLDER_FID} holding {len(merged[0])} packets'
                       if merged else 'no placeholder flow')
        return f'{len(ipv6)} IPv6 flows; {merged_text}'

    def netflix_ipv6_addresses():
        from tests.bug_fix_tests.test_3_ipv6 import _ipv6_reference
        path = _capture('netflix.pcap')
        df = _helpers.capture_dataframe(path)
        (_fids, n_ipv6) = _ipv6_reference(path)
        with_address = int(df['ip_src'].fillna('').astype(str).str.contains(':').sum())
        return f'{with_address} of the {n_ipv6} IPv6 packets have their addresses'

    def netflix_header():
        pcap = PCAP(str(_capture('netflix.pcap')))
        pcap.pcap2flows(q_interval=0.9)
        pcap.flow2features('IAT', fft=False, header=True)
        return f'features extracted: {pcap.features.shape[0]} flows x {pcap.features.shape[1]}'

    def protocol_labels():
        return f'labels: {pandas_of("protocols")["protocol"].tolist()}'

    def dns_values():
        row = pandas_of('dns').iloc[0]
        return f'dns_query = {row["dns_query"]!r}; dns_resp = {row["dns_resp"]!r}'

    def dns_invalid():
        return f'dns_query = {pandas_of("dns_invalid")["dns_query"].iloc[0]!r}'

    def time_columns():
        df = pandas_of('times')
        return (f'time is {type(df["time"].iloc[0]).__name__}; '
                f'datetime = {df["datetime"].iloc[0]} ({df["datetime"].dtype}) '
                f'[this machine\'s time zone: {time.tzname[0]}]')

    def time_normed():
        values = [round(float(v), 6) for v in pandas_of('times')['time_normed']]
        return f'time_normed: {values}'

    def protocol_counts():
        names = ['google_home.pcap', 'tplink_switch.pcap', 'nestcam_live.pcap', 'netflix.pcap',
                 'log4j.pcap', 'http.pcap']
        counts = []
        for name in names:
            protocols = _helpers.capture_dataframe(_capture(name))['protocol']
            counts.append(f'{name.split(".")[0]} {int(protocols.isna().sum()):,}')
        return 'labeled None: ' + ', '.join(counts)

    def dns_tuples():
        queries = _helpers.capture_dataframe(_capture('http.pcap'))['dns_query'].dropna()
        tuples = int(sum(isinstance(q, tuple) for q in queries))
        return f'{tuples} of {len(queries)} DNS queries are tuples'

    def datetime_fraction():
        df = _helpers.capture_dataframe(_capture('http.pcap'))
        datetimes = pd.to_datetime(df['datetime'])
        has_fraction = (df['time'].astype(float) % 1) > 1e-6
        lost = int(((datetimes.dt.microsecond == 0) & has_fraction).sum())
        return (f'{lost:,} of {int(has_fraction.sum()):,} datetimes lost their fraction of a '
                f'second; time zone: {datetimes.dt.tz or "none (local time)"}')

    def time_normed_real():
        names = ['netflix.pcap', 'http.pcap', 'log4j.pcap', 'nestcam_live.pcap',
                 'google_home.pcap', 'tplink_switch.pcap']
        negative = []
        for name in names:
            values = pd.to_numeric(_helpers.capture_dataframe(_capture(name))['time_normed'])
            negative.append(f'{name.split(".")[0]} {int((values < 0).sum())}')
        return 'negative time_normed values: ' + ', '.join(negative)

    def demo_auc():
        from sklearn.metrics import roc_auc_score
        from netml.ndm.iforest import IF
        data = np.load(ctx['demo_features'])
        (X, y) = (data['X'], data['y'])
        aucs = [roc_auc_score(y, IF(random_state=seed, verbose=0).fit(X).decision_function(X))
                for seed in range(5)]
        return (f'AUC {np.mean(aucs):.3f} (5 random seeds: {min(aucs):.3f} to {max(aucs):.3f}; '
                '0.5 is chance)')

    return {
        'lengths': lengths, 'size_features': size_features, 'netflix_lengths': netflix_lengths,
        'http_lengths': http_lengths, 'timeout_split': timeout_split,
        'interval_split': interval_split_synthetic, 'udp_port_6': udp_port_6,
        'log4j_timeout': log4j_timeout, 'demo_interval': demo_interval,
        'normal_interval': normal_interval, 'ipv6_fid': ipv6_fid, 'ipv6_merge': ipv6_merge,
        'ipv6_header': ipv6_header, 'netflix_ipv6_flows': netflix_ipv6_flows,
        'netflix_ipv6_addresses': netflix_ipv6_addresses, 'netflix_header': netflix_header,
        'protocol_labels': protocol_labels, 'dns_values': dns_values, 'dns_invalid': dns_invalid,
        'time_columns': time_columns, 'time_normed': time_normed,
        'protocol_counts': protocol_counts, 'dns_tuples': dns_tuples,
        'datetime_fraction': datetime_fraction, 'time_normed_real': time_normed_real,
        'demo_auc': demo_auc,
    }


def _helpers_wirelen(pkt):
    """Length of a packet on the wire, measured the same way for both versions."""
    return pkt.wirelen if getattr(pkt, 'wirelen', None) is not None else len(pkt)


def _short_error(exc):
    text = ' '.join(f'{type(exc).__name__}: {exc}'.split())
    return text if len(text) <= 160 else text[:157] + '...'


@contextlib.contextmanager
def _time_limit(seconds):
    """Raise TimeoutError if the block runs longer than `seconds` (Linux and macOS)."""
    if not hasattr(signal, 'SIGALRM'):
        yield
        return

    def _too_slow(_signum, _frame):
        raise TimeoutError(f'took longer than {seconds} s')

    previous = signal.signal(signal.SIGALRM, _too_slow)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_cases(ctx, keys):
    functions = _case_functions(ctx)
    outputs = {}
    for key in keys:
        try:
            with warnings.catch_warnings(), _time_limit(CASE_TIME_LIMIT):
                warnings.simplefilter('ignore')
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs[key] = functions[key]()
        except _Missing as exc:
            outputs[key] = f'skipped: {exc}'
        except Exception as exc:  # the original code's crashes are results, not failures
            outputs[key] = _short_error(exc)
    return outputs


class RecordingResult(unittest.TestResult):
    """Records one outcome per test: pass, fail, error or skip, with the reason."""

    def __init__(self):
        super().__init__()
        self.outcomes = {}
        self.class_problems = {}

    @staticmethod
    def name(test):
        module = type(test).__module__.split('.')[-1]
        return f'{module}.{type(test).__name__}.{test._testMethodName}'

    @staticmethod
    def reason(err):
        text = ' '.join(''.join(traceback.format_exception_only(err[0], err[1])).split())
        return text if len(text) <= 200 else text[:197] + '...'

    def _set(self, test, outcome, reason=''):
        if self.outcomes.get(self.name(test), ('pass',))[0] == 'pass':
            self.outcomes[self.name(test)] = (outcome, reason)

    def startTest(self, test):
        super().startTest(test)
        self.outcomes.setdefault(self.name(test), ('pass', ''))

    def addError(self, test, err):
        # a failed setUpClass is reported as "setUpClass (module.Class)", not as a TestCase
        if not isinstance(test, unittest.TestCase):
            self.class_problems[str(test).split('(')[-1].rstrip(')')] = (
                'error', 'setUpClass: ' + self.reason(err))
            return
        self._set(test, 'error', self.reason(err))

    def addFailure(self, test, err):
        self._set(test, 'fail', self.reason(err))

    def addSkip(self, test, reason):
        if not isinstance(test, unittest.TestCase):
            self.class_problems[str(test).split('(')[-1].rstrip(')')] = ('skip', reason)
            return
        self.outcomes[self.name(test)] = ('skip', reason)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            outcome = 'fail' if issubclass(err[0], test.failureException) else 'error'
            self._set(test, outcome, f'{subtest._subDescription()}: {self.reason(err)}')


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def run_tests(modules):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames([f'tests.bug_fix_tests.{m}' for m in modules])
    tests = list(_iter_tests(suite))
    result = RecordingResult()
    result.buffer = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        suite.run(result)
    outcomes = {}
    for test in tests:
        name = RecordingResult.name(test)
        class_id = f'{type(test).__module__}.{type(test).__name__}'
        outcome = result.class_problems.get(class_id) or result.outcomes.get(name)
        outcomes[name] = list(outcome or ('error', 'did not run'))
    return outcomes


def worker_main(version, out_json, ctx_json):
    import netml
    ctx = json.loads(ctx_json)
    expected = VERSIONS[version] / 'netml' / '__init__.py'
    if pathlib.Path(netml.__file__).resolve() != expected.resolve():
        raise SystemExit(f'{version}: imported netml from {netml.__file__}, expected {expected}')
    started = time.time()
    modules = [FIXES[fix][1] for fix in ctx['fixes']]
    tests = run_tests(modules)
    cases = run_cases(ctx, ctx['case_keys'])
    with open(out_json, 'w') as fh:
        json.dump({'tests': tests, 'cases': cases, 'seconds': time.time() - started}, fh)


def features_main(out_npz):
    """Extract the labeled IAT features of demo.pcap with the fixed code (for the IF case)."""
    import numpy as np
    from netml.pparser.parser import PCAP
    pcap = PCAP(str(_helpers.DEMO_PCAP))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        pcap.pcap2flows(q_interval=0.9)
        pcap.label_flows(label_file=str(_helpers.DEMO_LABELS))
        pcap.flow2features('IAT', fft=False, header=False)
    np.savez(out_npz, X=pcap.features, y=pcap.labels)


# ---------------------------------------------------------------------------------------------
# Parent: start the workers, then print the comparison
# ---------------------------------------------------------------------------------------------

def _python(version, args, synthetic_only):
    env = dict(os.environ, PYTHONPATH=str(VERSIONS[version]), PYTHONWARNINGS='ignore')
    env.pop(_helpers.SYNTHETIC_ONLY_VARIABLE, None)
    if synthetic_only:
        env[_helpers.SYNTHETIC_ONLY_VARIABLE] = '1'
    return subprocess.Popen([sys.executable, '-m', 'tests.bug_fix_tests.compare_before_after']
                            + args, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _wait_all(procs):
    """Wait for the worker processes, printing a progress line every PROGRESS_INTERVAL s."""
    started = time.time()
    last = started
    while any(p.poll() is None for p in procs.values()):
        time.sleep(1)
        if time.time() - last >= PROGRESS_INTERVAL:
            last = time.time()
            running = [v for (v, p) in procs.items() if p.poll() is None]
            print(f'  ... still running ({", ".join(running)}): {int(last - started)} s',
                  flush=True)
    for (version, proc) in procs.items():
        (_out, err) = proc.communicate()
        if proc.returncode < 0:
            # killed by a signal, e.g. SIGSEGV: a compiled package (numpy, pandas, ...) crashed
            signal_name = signal.Signals(-proc.returncode).name
            raise SystemExit(
                f'the {version} worker crashed ({signal_name}) inside a compiled package (numpy, '
                f'pandas, scapy, ...) of {sys.executable}.\nThis is a problem of that Python '
                'environment, not of netml: run the comparison in the netml development '
                f'environment, e.g.\n    {ENVIRONMENT_HINT}\n{err[-3000:]}')
        if proc.returncode:
            raise SystemExit(f'the {version} worker failed (exit {proc.returncode}):\n'
                             f'{err[-3000:]}')


def _wrap(text, indent, width=99):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def compare(fixes, synthetic_only, report_path):
    lines = []      # the Markdown report
    problems = []   # unexpected results

    def say(text=''):
        print(text)

    cases = [c for c in CASES
             if c['fix'] in fixes and not (synthetic_only and c['kind'] == 'real')]
    print('netml before vs after the five bug fixes')
    for version in VERSIONS:
        print(f'  {version + ":":8s}{VERSION_LABELS[version]}')
    started = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        ctx = {'pcaps': build_synthetic_pcaps(tmp), 'fixes': fixes,
               'case_keys': [c['key'] for c in cases],
               'demo_features': str(pathlib.Path(tmp) / 'demo_features.npz')}
        if any(c['key'] == 'demo_auc' for c in cases):
            proc = _python('after', ['--worker-features', ctx['demo_features']], False)
            _wait_all({'features': proc})
        estimate = 'a few seconds' if synthetic_only else 'about 5 minutes'
        print(f'\nRunning the tests and cases on both versions ({estimate})...', flush=True)
        procs = {}
        for version in VERSIONS:
            out = str(pathlib.Path(tmp) / f'{version}.json')
            procs[version] = (out, _python(version, ['--worker', version, out, json.dumps(ctx)],
                                           synthetic_only))
        _wait_all({v: p for (v, (_o, p)) in procs.items()})
        results = {}
        for (version, (out, _proc)) in procs.items():
            with open(out) as fh:
                results[version] = json.load(fh)

    lines.append('# netml before vs after the five bug fixes\n')
    for version in VERSIONS:
        lines.append(f'- **{version}**: `{VERSION_LABELS[version]}`')
    lines.append('')

    summary = []
    for fix in fixes:
        (title, module, bug) = FIXES[fix]
        say(f'\n{"=" * 99}\nFix {fix}/5: {title}\n{"=" * 99}')
        say(_wrap(f'The bug: {bug}', ''))
        lines.append(f'## Fix {fix}/5: {title}\n')
        lines.append(f'**The bug:** {bug}\n')

        # tests
        counts = {}
        for version in VERSIONS:
            tests = {n: o for (n, o) in results[version]['tests'].items()
                     if n.startswith(module + '.')}
            tally = {}
            for (outcome, _reason) in tests.values():
                tally[outcome] = tally.get(outcome, 0) + 1
            counts[version] = (tests, tally)
        (before_tests, before_tally) = counts['before']
        (after_tests, after_tally) = counts['after']

        def tally_text(tally):
            return ', '.join(f'{tally[k]} {k}' for k in ('pass', 'fail', 'error', 'skip')
                             if tally.get(k))

        fixed = [n for n in after_tests if after_tests[n][0] == 'pass'
                 and before_tests.get(n, ['?'])[0] in ('fail', 'error')]
        guards = [n for n in after_tests if after_tests[n][0] == 'pass'
                  and before_tests.get(n, ['?'])[0] == 'pass']
        broken = [n for n in after_tests if after_tests[n][0] in ('fail', 'error')]
        say(f'\nTests (tests/bug_fix_tests/{module}.py):')
        say(f'  before: {tally_text(before_tally)}')
        say(f'  after:  {tally_text(after_tally)}')
        say(f'  {len(fixed)} fail before and pass after; {len(guards)} pass on both '
            '(regression guards and checks of the test data)')
        for name in broken:
            problems.append(f'fix {fix}: test {name} fails on the fixed code: '
                            f'{after_tests[name][1]}')
        lines.append(f'**Tests** (`tests/bug_fix_tests/{module}.py`): before '
                     f'{tally_text(before_tally)}; after {tally_text(after_tally)}.\n')
        lines.append('| test | before | after |')
        lines.append('|---|---|---|')
        for name in after_tests:
            (b_outcome, b_reason) = before_tests.get(name, ['?', ''])
            before_text = b_outcome + (f': {b_reason}' if b_outcome in ('fail', 'error') else '')
            lines.append(f'| `{name.split(".", 1)[1]}` | {_md(before_text)} | '
                         f'{after_tests[name][0]} |')
        lines.append('')

        # cases
        fix_cases = [c for c in cases if c['fix'] == fix]
        changed = 0
        for case in fix_cases:
            before = results['before']['cases'][case['key']]
            after = results['after']['cases'][case['key']]
            differs = before != after
            skipped = before.startswith('skipped') or after.startswith('skipped')
            changed += differs and not skipped
            say(f'\n[{case["kind"]}] {case["input"]}')
            say(_wrap(f'before: {before}', '  ', 999 if len(before) < 200 else 99))
            say(_wrap(f'after:  {after}', '  ', 999 if len(after) < 200 else 99))
            lines.append(f'**[{case["kind"]}] {case["input"]}**\n')
            lines.append(f'- before: `{before}`')
            lines.append(f'- after: `{after}`')
            if skipped:
                note = 'skipped (capture missing)'
            elif differs and case['change']:
                note = f'Why the original output is wrong: {case["why"]}'
            elif not differs and not case['change']:
                note = f'No change, as expected {case["why"]}'
            elif differs:
                note = f'UNEXPECTED: this control case should not change {case["why"]}'
                problems.append(f'fix {fix}: control case changed: {case["input"]}')
            else:
                note = 'UNEXPECTED: the output did not change'
                problems.append(f'fix {fix}: case did not change: {case["input"]}')
            say(_wrap(note, '  '))
            lines.append(f'- {note}\n')
        summary.append((fix, title, len(fixed), len(after_tests), changed, len(fix_cases)))

    say(f'\n{"=" * 99}\nSummary\n{"=" * 99}')
    say(f'{"fix":<22}{"tests fail before, pass after":>32}{"cases changed":>18}')
    lines.append('## Summary\n')
    lines.append('| fix | tests that fail before and pass after | cases whose output changed |')
    lines.append('|---|---|---|')
    for (fix, title, n_fixed, n_tests, n_changed, n_cases) in summary:
        (name, tests_text, cases_text) = (f'{fix}. {title}', f'{n_fixed} of {n_tests}',
                                          f'{n_changed} of {n_cases}')
        say(f'{name:<22}{tests_text:>32}{cases_text:>18}')
        lines.append(f'| {fix}. {title} | {n_fixed} of {n_tests} | {n_changed} of {n_cases} |')
    lines.append('')
    say('(the other tests pass on both versions: regression guards, checks of the test data; '
        'the unchanged cases are controls)')
    if problems:
        say('\nUnexpected results:')
        for problem in problems:
            say(f'  - {problem}')
        lines.append('**Unexpected results:**\n')
        lines.extend(f'- {p}' for p in problems)
    else:
        say('\nNo unexpected results.')
    say(f'\n(took {time.time() - started:.0f} s)')

    if report_path:
        pathlib.Path(report_path).write_text('\n'.join(lines) + '\n')
        print(f'report written to {pathlib.Path(report_path).resolve()}')
    return 1 if problems else 0


def _check_environment():
    """Stop early, with a clear message, if this Python lacks netml's dependencies (the workers
    use the same Python)."""
    import importlib.util
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f'{sys.executable} lacks {", ".join(missing)}, needed by netml and the '
                         'tests.\nRun the comparison in the netml development environment, e.g.\n'
                         f'    {ENVIRONMENT_HINT}')


def _md(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--worker':
        worker_main(*argv[1:4])
        return 0
    if argv and argv[0] == '--worker-features':
        features_main(argv[1])
        return 0

    arg_parser = argparse.ArgumentParser(
        prog='python -m tests.bug_fix_tests.compare_before_after',
        description='Compare netml before and after the five bug fixes.')
    arg_parser.add_argument('--synthetic-only', '--quick', action='store_true',
                            help='only the synthetic tests and cases (a few seconds)')
    arg_parser.add_argument('--fix', type=int, nargs='+', choices=sorted(FIXES),
                            default=sorted(FIXES), metavar='N',
                            help='only these fixes (1-5; default: all)')
    arg_parser.add_argument('--report', metavar='FILE', help='also write a Markdown report')
    args = arg_parser.parse_args(argv)
    _check_environment()
    if not (VERSIONS['before'] / 'netml').is_dir():
        raise SystemExit(f'{VERSIONS["before"]} is missing (see tests/before_bug_fix/README.md)')
    return compare(sorted(set(args.fix)), args.synthetic_only, args.report)


if __name__ == '__main__':
    sys.exit(main())
