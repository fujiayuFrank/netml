"""Fix 3: IPv6 packets get real flow ids, addresses and header features.

The bugs
    netml looked up the network layer with scapy's ``IP`` class, which matches IPv4 only.
    - _get_fid() gave every IPv6 TCP/UDP packet the placeholder id ('', '', -1, -1, -1), so all
      IPv6 connections of a pcap were merged into one fake flow;
    - pcap2pandas() left the addresses of IPv6 packets empty;
    - _get_header_features() read the IPv4-only fields "proto" and "ttl", and crashed on IPv6
      packets (AttributeError).

Affected
    pcap2flows() and everything built on its flows, flow2features(header=True), pcap2pandas().
"""
import functools
import ipaddress
import unittest

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6, IPv6ExtHdrHopByHop
from scapy.utils import PcapReader

from netml.pparser import parser
from netml.pparser.parser import PCAP

from ._helpers import (RealCaptureTestCase, TemporaryPcapTestCase, TimeLimitedTestCase, at_time,
                       capture_dataframe, capture_flows, ether, quiet, require_capture)

V6_A, V6_B, V6_C = '2001:db8::1', '2001:db8::2', '2001:db8::3'
PLACEHOLDER_FID = ('', '', -1, -1, -1)
HOP_LIMIT = 57


def _v6_tcp(pkt_time, flags='A', src=V6_A, dst=V6_B, sport=1000):
    return at_time(ether() / IPv6(src=src, dst=dst, hlim=HOP_LIMIT)
                   / TCP(sport=sport, dport=80, flags=flags), pkt_time)


def _v6_udp(pkt_time, src=V6_C, dst=V6_A):
    return at_time(ether() / IPv6(src=src, dst=dst, hlim=HOP_LIMIT) / UDP(sport=5000, dport=53),
                   pkt_time)


class FlowIdTests(TimeLimitedTestCase):
    """_get_fid(): the five-tuple of a packet."""

    def test_ipv6_tcp(self):
        """Expected: (source, destination, source port, destination port, 6).
        Before the fix: the placeholder ('', '', -1, -1, -1)."""
        self.assertEqual(parser._get_fid(_v6_tcp(0)), (V6_A, V6_B, 1000, 80, 6))

    def test_ipv6_udp(self):
        """Expected: (source, destination, source port, destination port, 17).
        Before the fix: the placeholder."""
        self.assertEqual(parser._get_fid(_v6_udp(0)), (V6_C, V6_A, 5000, 53, 17))

    def test_ipv4_unchanged(self):
        """Regression guard: IPv4 flow ids are unchanged.
        Expected: ('10.0.0.1', '10.0.0.2', 1111, 80, 6), before and after the fix."""
        pkt = ether() / IP(src='10.0.0.1', dst='10.0.0.2') / TCP(sport=1111, dport=80)
        self.assertEqual(parser._get_fid(pkt), ('10.0.0.1', '10.0.0.2', 1111, 80, 6))


class IPv6PcapTests(TemporaryPcapTestCase):
    """A synthetic pcap with one IPv6 TCP connection (SYN, ACK, PSH+ACK) and one unrelated IPv6
    UDP connection, interleaved."""

    def setUp(self):
        super().setUp()
        packets = []
        for (t, flags) in ((0, 'S'), (1, 'A'), (2, 'PA')):
            packets.append(_v6_tcp(t, flags=flags))
            packets.append(_v6_udp(t + 0.5))
        self.pcap_file = self.write_pcap(packets)

    def test_connections_not_merged(self):
        """Expected: two flows, one per connection, with their real five-tuples.
        Before the fix: one fake flow ('', '', -1, -1, -1) holding all six packets."""
        with quiet():
            flows = parser._pcap2flows(self.pcap_file, 2)
        self.assertEqual([fid for (fid, _pkts) in flows],
                         [(V6_A, V6_B, 1000, 80, 6), (V6_C, V6_A, 5000, 53, 17)])
        self.assertEqual([len(pkts) for (_fid, pkts) in flows], [3, 3])

    def test_pcap2pandas_addresses(self):
        """pcap2pandas() fills in the IPv6 addresses and their integer forms.
        Expected: ip_src / ip_dst as strings, ip_src_int / ip_dst_int as integers.
        Before the fix: all four columns were empty (None)."""
        pcap = PCAP(self.pcap_file)
        with quiet():
            pcap.pcap2pandas()
        first = pcap.df.iloc[0]
        self.assertEqual((first['ip_src'], first['ip_dst']), (V6_A, V6_B))
        self.assertEqual(first['ip_src_int'], int(ipaddress.ip_address(V6_A)))
        self.assertEqual(first['ip_dst_int'], int(ipaddress.ip_address(V6_B)))

    def test_header_features(self):
        """_get_header_features() counts the TCP flags and uses the IPv6 hop limit as the TTL
        (the hop limit is the IPv6 equivalent of the IPv4 TTL).

        Expected, TCP flow: flag counts [F, S, R, P, A, U, E, C] = [0, 1, 0, 1, 2, 0, 0, 0],
        then the hop limit of each packet (57, 57, 57). UDP flow: no flags, then the hop limits.
        Before the fix: AttributeError ("proto").
        """
        with quiet():
            flows = parser._pcap2flows(self.pcap_file, 2)
        features = parser._get_header_features(flows)
        self.assertEqual(features[0], [0, 1, 0, 1, 2, 0, 0, 0] + [HOP_LIMIT] * 3)
        self.assertEqual(features[1], [0] * 8 + [HOP_LIMIT] * 3)

    def test_pipeline_with_header_features(self):
        """The full pipeline, pcap2flows() then flow2features(header=True), runs on IPv6.
        Expected: one row of features per flow. Before the fix: AttributeError ("proto")."""
        pcap = PCAP(self.pcap_file)
        with quiet():
            pcap.pcap2flows(q_interval=0.9)
            pcap.flow2features('IAT', fft=False, header=True)
        self.assertEqual(len(pcap.features), 2)


class ExtensionHeaderTests(TemporaryPcapTestCase):
    """IPv6 with an extension header between the IPv6 header and TCP."""

    def test_header_features_with_extension_header(self):
        """IPv6 extension headers (here hop-by-hop) sit between IPv6 and TCP; the TCP flags are
        still counted.

        Expected: flag counts [0, 1, 0, 0, 1, 0, 0, 0] (SYN, ACK), then the hop limits.
        Before the fix: AttributeError ("proto").
        """
        packets = [at_time(ether() / IPv6(src=V6_A, dst=V6_B, hlim=HOP_LIMIT)
                           / IPv6ExtHdrHopByHop() / TCP(sport=1000, dport=80, flags=flags), t)
                   for (t, flags) in ((0, 'S'), (1, 'A'))]
        with quiet():
            flows = parser._pcap2flows(self.write_pcap(packets), 2)
        features = parser._get_header_features(flows)
        self.assertEqual(features[0], [0, 1, 0, 0, 1, 0, 0, 0] + [HOP_LIMIT] * 2)


@functools.lru_cache(maxsize=None)
def _ipv6_reference(pcap_file):
    """Independent count of IPv6 traffic in a real capture: the IPv6 TCP/UDP five-tuples with at
    least 2 packets, and the number of IPv6 packets."""
    counts = {}
    n_ipv6_packets = 0
    with PcapReader(str(pcap_file)) as reader:
        for pkt in reader:
            if IPv6 not in pkt:
                continue
            n_ipv6_packets += 1
            for transport, proto in ((TCP, 6), (UDP, 17)):
                if transport in pkt:
                    key = (pkt[IPv6].src, pkt[IPv6].dst, pkt[transport].sport,
                           pkt[transport].dport, proto)
                    counts[key] = counts.get(key, 0) + 1
                    break
    return {key for (key, count) in counts.items() if count >= 2}, n_ipv6_packets


class RealCaptureTests(RealCaptureTestCase):
    """netflix.pcap (Assignment 1), the course capture with IPv6 traffic."""

    def setUp(self):
        self.pcap_file = require_capture(self, 'netflix.pcap')

    def test_ipv6_flows_not_merged(self):
        """Expected: the IPv6 flows found by _pcap2flows() are exactly the IPv6 TCP/UDP
        five-tuples with at least 2 packets (5 flows), and no flow has the placeholder id.
        Before the fix: the 5 flows were merged into one placeholder flow of 61 packets."""
        flows = capture_flows(self.pcap_file)
        with quiet():
            (expected_fids, _n_ipv6) = _ipv6_reference(self.pcap_file)
        fids = [fid for (fid, _pkts) in flows]
        self.assertNotIn(PLACEHOLDER_FID, fids)
        self.assertEqual({fid for fid in fids if ':' in fid[0]}, expected_fids)
        self.assertEqual(len(expected_fids), 5)

    def test_pcap2pandas_ipv6_addresses(self):
        """Expected: every IPv6 packet has its addresses in pcap2pandas().
        Before the fix: none had (the 105 IPv6 packets had empty addresses)."""
        df = capture_dataframe(self.pcap_file)
        with quiet():
            (_fids, n_ipv6) = _ipv6_reference(self.pcap_file)
        has_ipv6_address = df['ip_src'].fillna('').str.contains(':')
        self.assertEqual(int(has_ipv6_address.sum()), n_ipv6)
        self.assertEqual(n_ipv6, 105)

    def test_header_features_run(self):
        """Expected: flow2features(header=True) gives one row per flow.
        Before the fix: AttributeError ("proto") on the IPv6 packets."""
        pcap = PCAP(str(self.pcap_file))
        with quiet():
            pcap.pcap2flows(q_interval=0.9)
            pcap.flow2features('IAT', fft=False, header=True)
        self.assertEqual(len(pcap.features), len(pcap.flows))


if __name__ == '__main__':
    unittest.main()
