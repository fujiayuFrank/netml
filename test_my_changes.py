"""Tests for my netml changes.

Limitations covered:

1. Packet length. ``len(pkt)`` is the number of *captured* bytes. When a pcap is
   recorded with a small snaplen (e.g. netflix.pcap caps every record at 200
   bytes), this underestimates the real packet size. The size on the wire is
   ``pkt.wirelen``.

2. IPv6. ``pkt[IP]`` only matches IPv4, so IPv6 packets get an empty flow id,
   no addresses in pcap2pandas, and crash ``_get_header_features``.

3. Protocol labels in pcap2pandas. Only TCP/UDP/ICMP were labeled; everything else
   (ARP, ICMPv6, IGMP, ...) was None. Now: a list of explicitly checked protocols, then
   a fallback that names the protocol from the packet's own numbers (IP protocol number /
   EtherType, via scapy's tables), then None.

4. Shared feature-extraction settings (cli.py "TODO: ... simply concatenating features").
   Each pcap used to pick its own split interval and feature length, so `netml learn` with
   several pcaps crashed, and `netml classify` crashed on any pcap other than the training one.
   Now the settings are computed once over all pcaps, saved with the features and the model,
   and reused by classify.

5. `learn train -f FILE` emptied FILE before reading it (argparse.FileType('w+b')), and a failed
   `learn` run emptied existing -f / -o files. Now files are opened only when their data is ready.

6. Flow splitting. _pcap2flows (idle timeout) and _flows2subflows (interval) dropped the last
   subflow of every flow they split, and decided TCP vs UDP with "6 in fid", which also matches
   a port number of 6.

7. Header features and link layers. _get_header_features read pkt.payload.proto/.ttl and
   pkt.payload.payload.flags, i.e. assumed Ethernet / IPv4 / TCP: IPv6 and raw-IP captures
   crashed, and VLAN-tagged frames silently gave the IP flags instead of the TCP flags.

Run with:
    python test_my_changes.py            (all tests)
    python test_my_changes.py SyntheticTests  (only the fast synthetic tests)
"""
import contextlib
import io
import ipaddress
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
from scapy.all import (AH, ARP, EAPOL, ESP, GRE, ICMP, IP, LLC, SCTP, STP, TCP, UDP, VRRP,
                       CookedLinux, Dot1Q, Dot3, Ether, ICMPv6EchoRequest, ICMPv6MLReport, IPv6,
                       IPv6ExtHdrHopByHop, PPPoED, Raw, rdpcap, wrpcap)

from netml.pparser import parser
from netml.pparser.parser import PCAP
from netml.utils.tool import FeatureData, dump_data, load_data


DATA_DIR = '/home/frank/ml-systems/docs/notebooks/data'
NETFLIX_PCAP = os.path.join(DATA_DIR, 'netflix.pcap')
NETFLIX_CSV = os.path.join(DATA_DIR, 'netflix.csv.gz')
HTTP_PCAP = os.path.join(DATA_DIR, 'http.pcap')
HTTP_CSV = os.path.join(DATA_DIR, 'http.csv.gz')

# small pcaps shipped with netml (also used by tests/)
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'data')
DEMO_PCAP = os.path.join(EXAMPLES_DIR, 'demo.pcap')
NORMAL_PCAP = os.path.join(EXAMPLES_DIR, 'srcIP_10.42.0.1', 'srcIP_10.42.0.1_normal.pcap')
ANOMALY_PCAP = os.path.join(EXAMPLES_DIR, 'srcIP_10.42.0.1', 'srcIP_10.42.0.119_anomaly.pcap')

SNAPLEN = 200

# Synthetic flows: (fid, ip layer factory, transport factory, payload size,
#                   tcp flags per packet or None, ttl / hop limit, start time)
V4_A, V4_B, V4_C = '10.0.0.1', '10.0.0.2', '10.0.0.3'
V6_A, V6_B, V6_C = '2001:db8::1', '2001:db8::2', '2001:db8::3'

FID_V4_TCP = (V4_A, V4_B, 1111, 80, 6)
FID_V6_TCP = (V6_A, V6_B, 2222, 443, 6)
FID_V6_UDP = (V6_A, V6_C, 5353, 53, 17)
FID_V4_UDP = (V4_A, V4_C, 3333, 53, 17)

# wire length = Ether(14) + IP header (20 for v4, 40 for v6) + L4 (20 TCP, 8 UDP) + payload
WIRE_V4_TCP = 14 + 20 + 20 + 1000   # 1054, truncated to 200 on disk
WIRE_V6_TCP = 14 + 40 + 20 + 100    # 174, not truncated
WIRE_V6_UDP = 14 + 40 + 8 + 50      # 112, not truncated
WIRE_V4_UDP = 14 + 20 + 8 + 500     # 542, truncated to 200 on disk

GAP = 0.01  # seconds between packets of the same flow

MAC_SRC, MAC_DST = '00:00:00:00:00:01', '00:00:00:00:00:02'


def ETHER(**fields):
    """Ethernet header with explicit MACs, so scapy doesn't try to resolve them on the host network."""
    return Ether(src=MAC_SRC, dst=MAC_DST, **fields)


def _truncate(pkt, t):
    """Mimic a capture with a small snaplen: keep the first SNAPLEN bytes, remember the wire length."""
    wire = bytes(pkt)
    captured = Ether(wire[:SNAPLEN])
    captured.wirelen = len(wire)
    captured.time = t
    return captured


def _build_synthetic_packets():
    pkts = []

    tcp_v4_flags = ['S', 'A', 'PA']
    for i, flg in enumerate(tcp_v4_flags):
        p = (ETHER() / IP(src=V4_A, dst=V4_B, ttl=64)
             / TCP(sport=1111, dport=80, flags=flg) / Raw(b'a' * 1000))
        pkts.append(_truncate(p, 100.0 + i * GAP))

    tcp_v6_flags = ['S', 'A', 'FA']
    for i, flg in enumerate(tcp_v6_flags):
        p = (ETHER() / IPv6(src=V6_A, dst=V6_B, hlim=50)
             / TCP(sport=2222, dport=443, flags=flg) / Raw(b'b' * 100))
        pkts.append(_truncate(p, 100.001 + i * GAP))

    for i in range(2):
        p = (ETHER() / IPv6(src=V6_A, dst=V6_C, hlim=40)
             / UDP(sport=5353, dport=53) / Raw(b'c' * 50))
        pkts.append(_truncate(p, 100.002 + i * GAP))

    for i in range(2):
        p = (ETHER() / IP(src=V4_A, dst=V4_C, ttl=30)
             / UDP(sport=3333, dport=53) / Raw(b'd' * 500))
        pkts.append(_truncate(p, 100.003 + i * GAP))

    # a non-TCP/UDP packet: not a flow, but still a row in pcap2pandas
    icmp = ETHER() / IP(src=V4_A, dst=V4_B) / ICMP()
    pkts.append(_truncate(icmp, 100.004))

    pkts.sort(key=lambda p: float(p.time))
    return pkts


def _flows_by_fid(flows):
    return {fid: pkts for fid, pkts in flows}


def _features_by_fid(features, fids):
    return dict(zip(fids, features))


class SyntheticTests(unittest.TestCase):
    """Small hand-built pcap with exactly known sizes, addresses and flags."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.pcap_file = os.path.join(cls.tmpdir.name, 'synthetic.pcap')
        cls.packets = _build_synthetic_packets()
        wrpcap(cls.pcap_file, cls.packets)

        cls.flows = parser._pcap2flows(cls.pcap_file, flow_pkts_thres=2)
        cls.flow_map = _flows_by_fid(cls.flows)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    # --- sanity check on the fixture itself ---

    def test_fixture_is_truncated(self):
        """The pcap on disk really is truncated (otherwise the size tests prove nothing)."""
        with parser.PcapReader(self.pcap_file) as pcap_reader:
            captured = [len(p) for p in pcap_reader]
        self.assertEqual(max(captured), SNAPLEN)

    # --- IPv6: flow ids ---

    def test_get_fid_ipv4_unchanged(self):
        pkt = ETHER() / IP(src=V4_A, dst=V4_B) / TCP(sport=1111, dport=80)
        self.assertEqual(parser._get_fid(pkt), FID_V4_TCP)

    def test_get_fid_ipv6_tcp(self):
        pkt = ETHER() / IPv6(src=V6_A, dst=V6_B) / TCP(sport=2222, dport=443)
        self.assertEqual(parser._get_fid(pkt), FID_V6_TCP)

    def test_get_fid_ipv6_udp(self):
        pkt = ETHER() / IPv6(src=V6_A, dst=V6_C) / UDP(sport=5353, dport=53)
        self.assertEqual(parser._get_fid(pkt), FID_V6_UDP)

    def test_pcap2flows_keeps_ipv6_flows_separate(self):
        expected = {FID_V4_TCP, FID_V6_TCP, FID_V6_UDP, FID_V4_UDP}
        self.assertEqual(set(self.flow_map), expected)

    def test_pcap2flows_packet_counts(self):
        counts = {fid: len(pkts) for fid, pkts in self.flows}
        self.assertEqual(counts, {FID_V4_TCP: 3, FID_V6_TCP: 3, FID_V6_UDP: 2, FID_V4_UDP: 2})

    # --- length: size-based features use the wire length ---

    def test_get_SIZE_uses_wire_length(self):
        features, fids = parser._get_SIZE(self.flows)
        sizes = _features_by_fid(features, fids)
        self.assertEqual(sizes[FID_V4_TCP], [WIRE_V4_TCP] * 3)
        self.assertEqual(sizes[FID_V6_TCP], [WIRE_V6_TCP] * 3)
        self.assertEqual(sizes[FID_V6_UDP], [WIRE_V6_UDP] * 2)
        self.assertEqual(sizes[FID_V4_UDP], [WIRE_V4_UDP] * 2)

    def test_get_IAT_SIZE_uses_wire_length(self):
        features, fids = parser._get_IAT_SIZE(self.flows)
        iat_size = _features_by_fid(features, fids)[FID_V4_TCP]
        # layout: iat0, size0, iat1, size1, size2
        sizes = iat_size[1::2] + [iat_size[-1]]
        self.assertEqual(sizes, [WIRE_V4_TCP] * 3)
        np.testing.assert_allclose(iat_size[0:-1:2], [GAP, GAP], atol=1e-5)

    def test_get_STATS_uses_wire_length(self):
        features, fids = parser._get_STATS(self.flows)
        stats = _features_by_fid(features, fids)[FID_V4_TCP]
        (duration, pkts_rate, bytes_rate, mean, std,
         q1, q2, q3, min_size, max_size, num_pkts, num_bytes) = stats
        self.assertEqual(num_pkts, 3)
        self.assertEqual(num_bytes, 3 * WIRE_V4_TCP)
        self.assertEqual(mean, WIRE_V4_TCP)
        self.assertEqual(min_size, WIRE_V4_TCP)
        self.assertEqual(max_size, WIRE_V4_TCP)
        self.assertAlmostEqual(bytes_rate, 3 * WIRE_V4_TCP / duration, places=3)

    def test_get_SAMP_SIZE_uses_wire_length(self):
        # one big window covers the whole flow, so the single value is the flow's total bytes
        features, fids = parser._get_SAMP_SIZE(self.flows, sampling_rate=10)
        samp = _features_by_fid(features, fids)
        self.assertEqual(samp[FID_V4_TCP], [3 * WIRE_V4_TCP])
        self.assertEqual(samp[FID_V4_UDP], [2 * WIRE_V4_UDP])

    def test_get_SAMP_NUM_unchanged(self):
        features, fids = parser._get_SAMP_NUM(self.flows, sampling_rate=10)
        samp = _features_by_fid(features, fids)
        self.assertEqual(samp[FID_V4_TCP], [3])

    # --- IPv6: header features ---

    def test_header_features_ipv4_tcp(self):
        headers = parser._get_header_features([(FID_V4_TCP, self.flow_map[FID_V4_TCP])])[0]
        # flag order: F S R P A U E C; packets are S, A, PA
        self.assertEqual(headers[:8], [0, 1, 0, 1, 2, 0, 0, 0])
        self.assertEqual(headers[8:], [64, 64, 64])

    def test_header_features_ipv6_tcp(self):
        headers = parser._get_header_features([(FID_V6_TCP, self.flow_map[FID_V6_TCP])])[0]
        # packets are S, A, FA; IPv6 hop limit plays the role of TTL
        self.assertEqual(headers[:8], [1, 1, 0, 0, 2, 0, 0, 0])
        self.assertEqual(headers[8:], [50, 50, 50])

    def test_header_features_ipv6_udp(self):
        headers = parser._get_header_features([(FID_V6_UDP, self.flow_map[FID_V6_UDP])])[0]
        self.assertEqual(headers[:8], [0] * 8)
        self.assertEqual(headers[8:], [40, 40])

    # --- pcap2pandas ---

    def test_pcap2pandas(self):
        pcap = PCAP(self.pcap_file)
        pcap.pcap2pandas()
        df = pcap.df.sort_values('time').reset_index(drop=True)

        expected_lengths = [p.wirelen for p in self.packets]
        self.assertEqual(df['length'].tolist(), expected_lengths)

        is_v6 = df['ip_src'].str.contains(':', na=False)
        v6 = df[is_v6]
        self.assertEqual(len(v6), 5)  # 3 IPv6 TCP + 2 IPv6 UDP
        self.assertEqual(set(v6['ip_src']), {V6_A})
        self.assertEqual(set(v6['ip_dst']), {V6_B, V6_C})
        self.assertEqual(set(v6['protocol']), {'TCP', 'UDP'})

        # 128-bit ints must be exact (no float rounding)
        self.assertEqual(set(v6['ip_src_int']), {int(ipaddress.ip_address(V6_A))})

        # IPv4 rows are unchanged
        is_v4 = df['ip_src'] == V4_A
        self.assertEqual(int(is_v4.sum()), 6)  # 3 TCP + 2 UDP + 1 ICMP
        self.assertEqual(set(df.loc[is_v4, 'ip_src_int']), {int(ipaddress.ip_address(V4_A))})

        is_icmp = df['protocol'] == 'ICMP'
        self.assertEqual(int(is_icmp.sum()), 1)

    # --- end to end through the PCAP class ---

    def test_pipeline_with_header(self):
        pcap = PCAP(self.pcap_file, flow_ptks_thres=2)
        pcap.pcap2flows(q_interval=0.9)
        pcap.flow2features('STATS', fft=False, header=True)
        self.assertEqual(pcap.features.shape[0], 4)
        self.assertTrue(np.all(np.isfinite(pcap.features)))


# (expected protocol label, packet). Grouped by which part of the labeling logic decides it.
PROTOCOL_CASES = [
    # explicit if/elif branches
    ('TCP', ETHER() / IP() / TCP()),
    ('UDP', ETHER() / IPv6() / UDP()),
    ('ICMP', ETHER() / IP() / ICMP()),
    ('ICMPv6', ETHER() / IPv6() / ICMPv6EchoRequest()),
    ('ICMPv6', ETHER() / IPv6() / IPv6ExtHdrHopByHop() / ICMPv6MLReport()),
    ('SCTP', ETHER() / IP() / SCTP()),
    ('GRE', ETHER() / IP() / GRE() / Raw(b'x' * 8)),
    ('ESP', ETHER() / IP() / ESP(spi=1, seq=1, data=b'x' * 16)),
    ('AH', ETHER() / IP() / AH(nh=59)),
    ('VRRP', ETHER() / IP(dst='224.0.0.18') / VRRP()),
    ('ARP', ETHER() / ARP()),
    ('ARP', ETHER() / Dot1Q() / ARP()),
    ('EAPOL', ETHER() / EAPOL()),
    ('STP', Dot3(src=MAC_SRC, dst=MAC_DST) / LLC() / STP()),
    ('PPPoE', ETHER() / PPPoED()),
    # order of the branches: same labels as before for these
    ('TCP', ETHER() / IP() / GRE() / IP() / TCP()),        # tunneled TCP keeps the inner label
    ('ICMP', ETHER() / IP() / ICMP(type=3) / IP() / TCP()),  # ICMP error quoting a TCP header
    # fallback: name from the packet's own numbers (scapy has no dissector for these)
    ('IGMP', ETHER() / IP(proto=2) / Raw(b'\x11' * 8)),
    ('OSPF', ETHER() / IP(proto=89) / Raw(b'x' * 8)),
    ('OSPF', ETHER() / IPv6() / IPv6ExtHdrHopByHop(nh=89) / Raw(b'x' * 8)),  # skips extension header
    ('LLDP', ETHER(type=0x88cc) / Raw(b'x' * 8)),
    ('LLDP', ETHER() / Dot1Q(type=0x88cc) / Raw(b'x' * 8)),  # EtherType inside the VLAN tag
    # no name anywhere: None
    (None, ETHER() / IP(proto=253) / Raw(b'x' * 8)),  # 253 = reserved for experimentation
    (None, ETHER(type=0x88b5) / Raw(b'x' * 8)),       # 0x88b5 = local experimental EtherType
]


def _reparse(pkt):
    """Turn a built packet into bytes and dissect it again, as when reading it from a pcap."""
    return Ether(bytes(pkt))


class ProtocolTests(unittest.TestCase):
    """Protocol labels in pcap2pandas.

    The fallback cases rely on scapy's protocol tables, which it loads from /etc/protocols
    and /etc/ethertypes.
    """

    def test_get_protocol(self):
        for expected, pkt in PROTOCOL_CASES:
            with self.subTest(expected=expected, packet=pkt.summary()):
                self.assertEqual(parser._get_protocol(_reparse(pkt)), expected)

    def test_pcap2pandas_protocol_column(self):
        packets = [_truncate(pkt, 100.0 + i) for i, (_, pkt) in enumerate(PROTOCOL_CASES)]
        expected = [label for label, _ in PROTOCOL_CASES]

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_file = os.path.join(tmpdir, 'protocols.pcap')
            wrpcap(pcap_file, packets)
            pcap = PCAP(pcap_file)
            pcap.pcap2pandas()

        df = pcap.df.sort_values('time').reset_index(drop=True)
        protocols = df['protocol'].where(df['protocol'].notna(), None).tolist()
        self.assertEqual(protocols, expected)


def _protocol_counts(df):
    return df['protocol'].value_counts(dropna=False).to_dict()


@unittest.skipUnless(os.path.exists(NETFLIX_PCAP), 'netflix.pcap not available')
class NetflixTests(unittest.TestCase):
    """netflix.pcap: every record is truncated to 200 bytes, and it has some IPv6 traffic.

    netflix.csv.gz is a Wireshark export of the same capture; its 'Length' column is the
    wire length, so it is an independent ground truth.
    """

    @classmethod
    def setUpClass(cls):
        cls.pcap = PCAP(NETFLIX_PCAP)
        cls.pcap.pcap2pandas()
        cls.df = cls.pcap.df.sort_values('time', kind='stable').reset_index(drop=True)
        cls.flows = parser._pcap2flows(NETFLIX_PCAP, flow_pkts_thres=2)

    def test_lengths_match_wireshark(self):
        csv = pd.read_csv(NETFLIX_CSV)
        self.assertEqual(len(self.df), len(csv))
        np.testing.assert_array_equal(self.df['length'].to_numpy(), csv['Length'].to_numpy())

    def test_lengths_exceed_snaplen(self):
        self.assertGreater(self.df['length'].max(), SNAPLEN)

    def test_ipv6_addresses_present(self):
        is_v6 = self.df['ip_src'].str.contains(':', na=False)
        # 105 = number of IPv6 packets in netflix.pcap (61 UDP + 44 other, counted with scapy)
        self.assertEqual(int(is_v6.sum()), 105)

    def test_ipv6_flows_extracted(self):
        fids = pd.DataFrame([fid for fid, _ in self.flows],
                            columns=['src', 'dst', 'sport', 'dport', 'proto'])
        is_v6 = fids['src'].str.contains(':')
        self.assertGreater(int(is_v6.sum()), 0)
        # no flow should have the "unknown" placeholder id
        is_placeholder = fids['proto'] == -1
        self.assertEqual(int(is_placeholder.sum()), 0)

    def test_protocol_counts(self):
        # before: 80 packets (ICMPv6, ARP, IGMP, EAPOL) had protocol None
        expected = {'TCP': 141143, 'UDP': 232, 'ICMPv6': 44, 'ARP': 26, 'ICMP': 16,
                    'IGMP': 8, 'EAPOL': 2}
        self.assertEqual(_protocol_counts(self.df), expected)

    def test_flow_sizes_exceed_snaplen(self):
        features, _ = parser._get_SIZE(self.flows)
        max_size = max(max(sizes) for sizes in features)
        self.assertGreater(max_size, SNAPLEN)

    def test_pipeline_with_header(self):
        pcap = PCAP(NETFLIX_PCAP, flow_ptks_thres=2)
        pcap.pcap2flows(q_interval=0.9)
        pcap.flow2features('STATS', fft=False, header=True)
        max_pkt_size_col = 8 + 9  # 8 header flags, then STATS index 9 is max size
        self.assertGreater(pcap.features[:, max_pkt_size_col].max(), SNAPLEN)


@unittest.skipUnless(os.path.exists(HTTP_PCAP), 'http.pcap not available')
class HttpRegressionTests(unittest.TestCase):
    """http.pcap is not truncated: the fix must not change anything here."""

    def test_lengths_match_wireshark(self):
        pcap = PCAP(HTTP_PCAP)
        pcap.pcap2pandas()
        df = pcap.df.sort_values('time', kind='stable').reset_index(drop=True)
        csv = pd.read_csv(HTTP_CSV)
        np.testing.assert_array_equal(df['length'].to_numpy(), csv['Length'].to_numpy())

    def test_wire_length_equals_captured_length(self):
        flows = parser._pcap2flows(HTTP_PCAP, flow_pkts_thres=2)
        features, _ = parser._get_SIZE(flows)
        captured = [[len(p) for p in pkts] for _, pkts in flows]
        self.assertEqual(features, captured)


class RealDataProtocolTests(unittest.TestCase):
    """Protocol counts on the other real pcaps (counted from their scapy layer stacks)."""

    def _check(self, filename, expected):
        pcap_file = os.path.join(DATA_DIR, filename)
        if not os.path.exists(pcap_file):
            self.skipTest(f'{filename} not available')
        pcap = PCAP(pcap_file)
        pcap.pcap2pandas()
        self.assertEqual(_protocol_counts(pcap.df), expected)

    def test_log4j(self):
        # before: 3 SCTP packets were None
        self._check('log4j.pcap', {'TCP': 62155, 'ICMP': 15682, 'UDP': 3003, 'SCTP': 3})

    def test_nestcam(self):
        # before: 76 ARP + 4 EAPOL were None
        self._check('nestcam_live.pcap', {'TCP': 29389, 'ARP': 76, 'EAPOL': 4})

    def test_uchicago_cs(self):
        # before: 3 ARP + 9 IGMP were None
        self._check('uchicago-cs-20221103.pcap', {'TCP': 26774, 'UDP': 294, 'IGMP': 9, 'ARP': 3})


def _run_cli(*argv):
    """Run the netml CLI in-process. Returns (stdout, stderr); fails the test on any error exit."""
    import argcmdr
    from netml import cli

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            argcmdr.main(cli.Main, argv=[str(a) for a in argv])
        except SystemExit as exc:
            if exc.code:
                raise AssertionError(f'netml {argv} exited with {exc.code}: {stderr.getvalue()}')
    return stdout.getvalue(), stderr.getvalue()


def _run_cli_status(*argv):
    """Run the netml CLI in-process, expecting it may fail. Returns (exit code, stderr)."""
    import argcmdr
    from netml import cli

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            argcmdr.main(cli.Main, argv=[str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code or 0
    return code, stderr.getvalue()


def _read_bytes(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _iat_features(pcap_file, **pcap2flows_kwargs):
    pcap = PCAP(pcap_file)
    pcap.pcap2flows(**pcap2flows_kwargs)
    pcap.flow2features('IAT')
    return pcap


class ExtractionSettingsLibraryTests(unittest.TestCase):
    """parser.py: two-step flow extraction, fixed feature length, shared settings."""

    def test_pcap2flows_with_interval(self):
        """Before: pcap2flows(interval=...) left q_interval unset, so flow2features crashed."""
        pcap = _iat_features(DEMO_PCAP, interval=2.0)
        self.assertEqual(pcap.interval, 2.0)
        self.assertEqual(pcap.features.shape[1], pcap.pkts_dim - 1)

    def test_two_steps_equal_pcap2flows(self):
        one_step = _iat_features(DEMO_PCAP, q_interval=0.9)

        two_steps = PCAP(DEMO_PCAP)
        two_steps.extract_flows()
        interval = parser.get_shared_interval([two_steps], q_interval=0.9)
        two_steps.split_flows(interval)
        pkts_dim = parser.get_shared_pkts_dim([two_steps], q_interval=0.9)
        two_steps.flow2features('IAT', pkts_dim=pkts_dim)

        self.assertEqual(two_steps.interval, one_step.interval)
        self.assertEqual(two_steps.pkts_dim, one_step.pkts_dim)
        self.assertEqual(two_steps.fids, one_step.fids)
        np.testing.assert_array_equal(two_steps.features, one_step.features)

    def test_pkts_dim_fixes_feature_length(self):
        pcap = PCAP(DEMO_PCAP)
        pcap.pcap2flows(q_interval=0.9)
        expected_widths = {'IAT': 19, 'SIZE': 20, 'IAT_SIZE': 39, 'SAMP_NUM': 19, 'STATS': 12}
        for feat_type, width in expected_widths.items():
            with self.subTest(feat_type=feat_type):
                pcap.flow2features(feat_type, pkts_dim=20, sampling_rate=0.5)
                self.assertEqual(pcap.features.shape[1], width)

    def test_shared_interval_is_quantile_of_all_flows(self):
        pcaps = []
        for pcap_file in (NORMAL_PCAP, ANOMALY_PCAP):
            pcap = PCAP(pcap_file)
            pcap.extract_flows()
            pcaps.append(pcap)

        durations = []
        for pcap in pcaps:
            durations.extend(parser._get_flow_durations(pcap.raw_flows))
        expected = np.quantile(durations, 0.9)

        self.assertAlmostEqual(parser.get_shared_interval(pcaps, q_interval=0.9), expected)


class ExtractionSettingsCLITests(unittest.TestCase):
    """cli.py: learn with several pcaps, classify with saved settings, old file formats."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = cls.tmpdir.name

        # several pcaps: before the fix, this crashed in np.concatenate
        cls.multi_features_path = os.path.join(cls.tmp, 'multi-features')
        _run_cli('learn', 'extract', '-f', cls.multi_features_path,
                 '--pcap-normal', NORMAL_PCAP, '--pcap-abnormal', ANOMALY_PCAP)

        cls.demo_model_path = os.path.join(cls.tmp, 'demo-model')
        _run_cli('learn', '-p', DEMO_PCAP, '-o', cls.demo_model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_feature_file_still_unpacks_as_pair(self):
        data = load_data(self.multi_features_path)
        (features, labels) = data
        self.assertIsInstance(data, FeatureData)
        self.assertEqual(len(features), len(labels))

    def test_multi_pcap_settings(self):
        data = load_data(self.multi_features_path)
        (features, _labels) = data
        params = data.params

        self.assertEqual(params['feat_type'], 'IAT')
        self.assertEqual(features.shape[1], params['pkts_dim'] - 1)

        # the interval is computed from both pcaps (normal alone: ~1.01 s, anomaly alone: ~18 s)
        pcaps = []
        for pcap_file in (NORMAL_PCAP, ANOMALY_PCAP):
            pcap = PCAP(pcap_file)
            pcap.extract_flows()
            pcaps.append(pcap)
        self.assertAlmostEqual(params['interval'], parser.get_shared_interval(pcaps, 0.9))

    def test_single_pcap_unchanged(self):
        """With one pcap, the CLI gives the same features as before (that pcap's own settings)."""
        feature_path = os.path.join(self.tmp, 'demo-features')
        _run_cli('learn', 'extract', '-p', DEMO_PCAP, '-f', feature_path)
        (features, _labels) = load_data(feature_path)

        expected = _iat_features(DEMO_PCAP, q_interval=0.9)
        np.testing.assert_array_equal(features, expected.features)

    def test_model_saves_settings(self):
        (model, _history) = load_data(self.demo_model_path)
        expected = _iat_features(DEMO_PCAP, q_interval=0.9)
        self.assertEqual(model.extraction_params['interval'], expected.interval)
        self.assertEqual(model.extraction_params['pkts_dim'], expected.pkts_dim)

    def test_classify_other_pcap(self):
        """Before: 'X has 12 features, but OCSVM is expecting 51 features as input'."""
        _stdout, stderr = _run_cli('classify', '-m', self.demo_model_path, '-p', NORMAL_PCAP)
        self.assertNotIn('Error', stderr)

    def test_classify_with_old_model_file(self):
        """A model saved by an older netml has no settings: feature length taken from the model."""
        (model, history) = load_data(self.demo_model_path)
        del model.extraction_params
        old_model_path = os.path.join(self.tmp, 'old-model')
        dump_data((model, history), out_file=old_model_path)

        _stdout, stderr = _run_cli('classify', '-m', old_model_path, '-p', NORMAL_PCAP)
        self.assertIn('no saved feature-extraction settings', stderr)
        self.assertNotIn('Error', stderr)


class CLIFeatureFileTests(unittest.TestCase):
    """-f/--feature and -o/--output are opened only when their data is ready."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self.tmpdir.name
        self.features_path = os.path.join(self.tmp, 'features')
        self.model_path = os.path.join(self.tmp, 'model')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_in_isolation(self):
        """Before: the features file was emptied on open, then 'EOFError: Ran out of input'."""
        _run_cli('learn', 'extract', '-p', DEMO_PCAP, '-f', self.features_path)
        saved = _read_bytes(self.features_path)

        _run_cli('learn', 'train', '-f', self.features_path, '-o', self.model_path)

        self.assertEqual(_read_bytes(self.features_path), saved)  # input left as it was
        (model, _history) = load_data(self.model_path)
        self.assertEqual(model.extraction_params, load_data(self.features_path).params)

    def test_extract_and_train_with_feature_file(self):
        _run_cli('learn', '-p', DEMO_PCAP, '-f', self.features_path, '-o', self.model_path)
        (features, _labels) = load_data(self.features_path)
        (model, _history) = load_data(self.model_path)
        self.assertEqual(features.shape[1], model.extraction_params['pkts_dim'] - 1)

    def test_train_missing_feature_file(self):
        code, stderr = _run_cli_status('learn', 'train', '-f', self.features_path,
                                       '-o', self.model_path)
        self.assertEqual(code, 2)
        self.assertIn('not found', stderr)
        self.assertFalse(os.path.exists(self.model_path))

    def test_failed_extract_keeps_existing_files(self):
        _run_cli('learn', '-p', DEMO_PCAP, '-f', self.features_path, '-o', self.model_path)
        features_before = _read_bytes(self.features_path)
        model_before = _read_bytes(self.model_path)

        # a CSV is not a pcap: extraction fails
        code, _stderr = _run_cli_status('learn', '-p', os.path.join(EXAMPLES_DIR, 'demo.csv'),
                                        '-f', self.features_path, '-o', self.model_path)
        self.assertNotEqual(code, 0)
        self.assertEqual(_read_bytes(self.features_path), features_before)
        self.assertEqual(_read_bytes(self.model_path), model_before)

    def test_bad_arguments_keep_existing_files(self):
        _run_cli('learn', 'extract', '-p', DEMO_PCAP, '-f', self.features_path)
        before = _read_bytes(self.features_path)

        code, _stderr = _run_cli_status('learn', 'extract', '-p', DEMO_PCAP,
                                        '-f', self.features_path, '--no-such-option')
        self.assertEqual(code, 2)
        self.assertEqual(_read_bytes(self.features_path), before)

    def test_train_from_old_feature_file(self):
        """A feature file saved by an older netml is a plain (features, labels) tuple."""
        _run_cli('learn', 'extract', '-p', DEMO_PCAP, '-f', self.features_path)
        (features, labels) = load_data(self.features_path)
        with open(self.features_path, 'wb') as fh:
            pickle.dump((features, labels), fh)

        _run_cli('learn', 'train', '-f', self.features_path, '-o', self.model_path)
        (model, _history) = load_data(self.model_path)
        self.assertIsNone(model.extraction_params)


def _tcp_v4(t, sport=1111, dport=80):
    p = ETHER() / IP(src=V4_A, dst=V4_B) / TCP(sport=sport, dport=dport)
    return _truncate(p, t)


def _packet_times(flows):
    return [[float(p.time) for p in pkts] for _, pkts in flows]


class SubflowSplitTests(unittest.TestCase):
    """_get_fid / _pcap2flows / _flows2subflows: no flow is merged or lost by splitting."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _flows_from(self, packets, **kwargs):
        pcap_file = os.path.join(self.tmpdir.name, 'flows.pcap')
        wrpcap(pcap_file, packets)
        return parser._pcap2flows(pcap_file, 2, **kwargs)

    def test_unrelated_ipv6_connections_not_merged(self):
        """Before the IPv6 fix, both connections got the placeholder ('', '', -1, -1, -1)."""
        packets = []
        for t in (0, 1):
            tcp = ETHER() / IPv6(src=V6_A, dst=V6_B) / TCP(sport=1000, dport=80)
            udp = ETHER() / IPv6(src=V6_C, dst=V6_A) / UDP(sport=5000, dport=53)
            packets.append(_truncate(tcp, t))
            packets.append(_truncate(udp, t + 0.5))

        fids = [fid for fid, _ in self._flows_from(packets)]
        self.assertEqual(fids, [(V6_A, V6_B, 1000, 80, 6), (V6_C, V6_A, 5000, 53, 17)])

    # --- _pcap2flows: idle timeout ---

    def test_timeout_split_keeps_last_subflow(self):
        packets = [_tcp_v4(t) for t in (0, 1, 2, 702, 703, 704)]  # idle 700 s > tcp_timeout
        flows = self._flows_from(packets, tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2], [702, 703, 704]])

    def test_timeout_split_into_three(self):
        packets = [_tcp_v4(t) for t in (0, 1, 700, 701, 1400, 1401)]
        flows = self._flows_from(packets, tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1], [700, 701], [1400, 1401]])

    def test_timeout_single_packet_tail_dropped(self):
        """A last subflow with fewer than flow_pkts_thres packets is dropped, like any other."""
        packets = [_tcp_v4(t) for t in (0, 1, 2, 702)]
        flows = self._flows_from(packets, tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2]])

    def test_timeout_not_split(self):
        packets = [_tcp_v4(t) for t in (0, 1, 2, 3)]
        flows = self._flows_from(packets, tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2, 3]])

    def test_udp_port_6_uses_udp_timeout(self):
        """Before: '6 in fid' matched source port 6, so this UDP flow got the TCP timeout."""
        packets = []
        for t in (0, 1, 101, 102):  # idle 100 s: > tcp_timeout, < udp_timeout
            p = ETHER() / IP(src=V4_A, dst=V4_C) / UDP(sport=6, dport=53)
            packets.append(_truncate(p, t))
        flows = self._flows_from(packets, tcp_timeout=50, udp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 101, 102]])

    def test_tcp_port_17_uses_tcp_timeout(self):
        packets = [_tcp_v4(t, sport=17) for t in (0, 1, 101, 102)]
        flows = self._flows_from(packets, tcp_timeout=50, udp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1], [101, 102]])

    # --- _flows2subflows: interval ---

    def _subflows(self, times, interval):
        packets = [_tcp_v4(t) for t in times]
        flows = [(parser._get_fid(packets[0]), packets)]
        return parser._flows2subflows(flows, interval, flow_pkts_thres=2)

    def test_interval_split_keeps_last_subflow(self):
        subflows = self._subflows((0, 1, 2, 12, 13, 14, 24, 25, 26), interval=5)
        self.assertEqual(_packet_times(subflows), [[0, 1, 2], [12, 13, 14], [24, 25, 26]])

    def test_interval_single_packet_tail_dropped(self):
        subflows = self._subflows((0, 1, 2, 12), interval=5)
        self.assertEqual(_packet_times(subflows), [[0, 1, 2]])

    def test_interval_not_split(self):
        subflows = self._subflows((0, 1, 2, 3), interval=5)
        self.assertEqual(_packet_times(subflows), [[0, 1, 2, 3]])

    def test_no_packets_lost(self):
        """Every packet ends up in exactly one subflow (when no subflow is too short)."""
        times = (0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41)
        subflows = self._subflows(times, interval=5)
        kept = sorted(t for pkts in _packet_times(subflows) for t in pkts)
        self.assertEqual(kept, list(times))


# pcap link types
LINKTYPE_RAW = 101           # packets start with the IP header
LINKTYPE_LINUX_SLL = 113     # Linux "cooked" capture (e.g. tcpdump -i any)

# (name, link type or None for Ethernet, packet factory taking (ttl, tcp flags), ttl)
HEADER_LAYOUTS = [
    ('Ether/IPv4', None, lambda ttl, flg: ETHER() / IP(src=V4_A, dst=V4_B, ttl=ttl)
     / TCP(sport=1111, dport=80, flags=flg)),
    ('Ether/IPv6', None, lambda ttl, flg: ETHER() / IPv6(src=V6_A, dst=V6_B, hlim=ttl)
     / TCP(sport=1111, dport=80, flags=flg)),
    ('Ether/IPv6/hop-by-hop', None, lambda ttl, flg: ETHER() / IPv6(src=V6_A, dst=V6_B, hlim=ttl)
     / IPv6ExtHdrHopByHop() / TCP(sport=1111, dport=80, flags=flg)),
    ('Ether/VLAN/IPv4', None, lambda ttl, flg: ETHER() / Dot1Q(vlan=10)
     / IP(src=V4_A, dst=V4_B, ttl=ttl, flags='DF') / TCP(sport=1111, dport=80, flags=flg)),
    ('raw IPv4', LINKTYPE_RAW, lambda ttl, flg: IP(src=V4_A, dst=V4_B, ttl=ttl)
     / TCP(sport=1111, dport=80, flags=flg)),
    ('raw IPv6', LINKTYPE_RAW, lambda ttl, flg: IPv6(src=V6_A, dst=V6_B, hlim=ttl)
     / TCP(sport=1111, dport=80, flags=flg)),
    ('Linux cooked/IPv4', LINKTYPE_LINUX_SLL, lambda ttl, flg: CookedLinux(proto=0x0800)
     / IP(src=V4_A, dst=V4_B, ttl=ttl) / TCP(sport=1111, dport=80, flags=flg)),
]

LAYOUT_TCP_FLAGS = ['S', 'SA', 'A', 'PA', 'FA']
# F S R P A U E C counts for the flags above
LAYOUT_FLAG_COUNTS = [1, 2, 0, 1, 4, 0, 0, 0]
LAYOUT_TTL = 57


class HeaderLayoutTests(unittest.TestCase):
    """Header features (TCP flags + TTL / hop limit) on different network and link layers.

    Before: IPv6 and raw-IP layouts raised AttributeError, and VLAN-tagged frames counted the IP
    flags (DF) instead of the TCP flags.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_flow(self, linktype, make_packet):
        packets = []
        for i, flg in enumerate(LAYOUT_TCP_FLAGS):
            pkt = make_packet(LAYOUT_TTL, flg)
            pkt.time = 100.0 + i * GAP
            packets.append(pkt)

        pcap_file = os.path.join(self.tmpdir.name, 'layout.pcap')
        wrpcap(pcap_file, packets, linktype=linktype)
        return pcap_file

    def test_header_features(self):
        for name, linktype, make_packet in HEADER_LAYOUTS:
            with self.subTest(layout=name):
                pcap_file = self._write_flow(linktype, make_packet)
                packets = rdpcap(pcap_file)
                headers = parser._get_header_features([(None, packets)])[0]
                self.assertEqual(headers[:8], LAYOUT_FLAG_COUNTS)
                self.assertEqual(headers[8:], [LAYOUT_TTL] * len(LAYOUT_TCP_FLAGS))

    def test_pipeline_with_header(self):
        """pcap -> flows -> STATS + header features, end to end."""
        for name, linktype, make_packet in HEADER_LAYOUTS:
            with self.subTest(layout=name):
                pcap = PCAP(self._write_flow(linktype, make_packet), flow_ptks_thres=2)
                pcap.pcap2flows(q_interval=0.9)
                self.assertEqual(len(pcap.flows), 1)
                self.assertEqual(pcap.flows[0][0][2:], (1111, 80, 6))

                pcap.flow2features('STATS', header=True)
                self.assertEqual(list(pcap.features[0, :8]), LAYOUT_FLAG_COUNTS)

    def test_pcap2pandas(self):
        for name, linktype, make_packet in HEADER_LAYOUTS:
            with self.subTest(layout=name):
                pcap = PCAP(self._write_flow(linktype, make_packet))
                pcap.pcap2pandas()
                self.assertEqual(set(pcap.df['protocol']), {'TCP'})
                self.assertEqual(set(pcap.df['port_dst']), {80})
                self.assertTrue(pcap.df['ip_src'].notna().all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
