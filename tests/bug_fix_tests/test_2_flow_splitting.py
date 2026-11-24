"""Fix 2: splitting a flow keeps its final subflow, and TCP/UDP is told apart by protocol number.

The bugs
    _pcap2flows() splits a flow wherever it is idle for longer than the TCP/UDP timeout, and
    _flows2subflows() splits it again by an interval. Both appended a subflow to their result only
    when the *next* split started, so the packets after the last split of every split flow were
    silently dropped. Both also chose the TCP or UDP rule with ``6 in fid``, which is true when
    *any* item of the flow id is 6, e.g. a UDP flow from port 6, so such a flow got the TCP
    timeout.

Affected
    the flows returned by pcap2flows(), and so every feature and every model trained on them.
"""
import unittest

from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

from netml.pparser import parser
from netml.pparser.parser import PCAP

from ._helpers import (DEMO_PCAP, RealCaptureTestCase, TemporaryPcapTestCase, at_time,
                       capture_flows, ether, quiet, require_capture)


def _tcp(pkt_time, sport=1111):
    return at_time(ether() / IP(src='10.0.0.1', dst='10.0.0.2') / TCP(sport=sport, dport=80),
                   pkt_time)


def _udp(pkt_time, sport=5353):
    return at_time(ether() / IP(src='10.0.0.1', dst='10.0.0.3') / UDP(sport=sport, dport=53),
                   pkt_time)


def _packet_times(flows):
    """The arrival times of the packets of each flow, e.g. [[0, 1], [700, 701]]."""
    return [[float(pkt.time) for pkt in pkts] for (_fid, pkts) in flows]


class TimeoutSplitTests(TemporaryPcapTestCase):
    """_pcap2flows(): a flow is split where it is idle for longer than the timeout (600 s)."""

    def _split(self, packets, **timeouts):
        with quiet():
            return parser._pcap2flows(self.write_pcap(packets), 2, **timeouts)

    def test_final_subflow_kept(self):
        """A TCP flow idle for 700 s between its third and fourth packets is split in two.

        Expected: [[0, 1, 2], [702, 703, 704]]. Before the fix: [[0, 1, 2]] (the packets after
        the split were dropped).
        """
        flows = self._split([_tcp(t) for t in (0, 1, 2, 702, 703, 704)], tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2], [702, 703, 704]])

    def test_split_in_three(self):
        """A flow with two idle gaps is split in three.

        Expected: [[0, 1], [700, 701], [1400, 1401]]. Before the fix: the last subflow was
        dropped.
        """
        flows = self._split([_tcp(t) for t in (0, 1, 700, 701, 1400, 1401)], tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1], [700, 701], [1400, 1401]])

    def test_short_final_subflow_dropped(self):
        """A final subflow with fewer than 2 packets is dropped, like any other subflow that is
        too short to give features.

        Expected: [[0, 1, 2]], before and after the fix.
        """
        flows = self._split([_tcp(t) for t in (0, 1, 2, 702)], tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2]])

    def test_unsplit_flow_unchanged(self):
        """Regression guard: a flow without long idle gaps is returned whole.

        Expected: [[0, 1, 2, 3]], before and after the fix.
        """
        flows = self._split([_tcp(t) for t in (0, 1, 2, 3)], tcp_timeout=600)
        self.assertEqual(_packet_times(flows), [[0, 1, 2, 3]])

    def test_udp_port_6_uses_udp_timeout(self):
        """A UDP flow from port 6 is split by the UDP timeout (50 s here), not the TCP one.

        Expected: [[0, 1], [101, 102]]. Before the fix: [[0, 1, 101, 102]], because
        "6 in fid" matched the port and the TCP timeout (600 s) was used.
        """
        packets = [_udp(t, sport=6) for t in (0, 1, 101, 102)]
        flows = self._split(packets, tcp_timeout=600, udp_timeout=50)
        self.assertEqual(_packet_times(flows), [[0, 1], [101, 102]])


class IntervalSplitTests(TemporaryPcapTestCase):
    """_flows2subflows(): a flow is split where it is idle for longer than the interval."""

    def _subflows(self, packets, interval):
        with quiet():
            flows = parser._pcap2flows(self.write_pcap(packets), 2)
        return parser._flows2subflows(flows, interval)

    def test_final_subflow_kept(self):
        """A flow with idle gaps of 10 s, split by a 5-second interval, gives three subflows.

        Expected: [[0, 1, 2], [12, 13, 14], [24, 25, 26]]. Before the fix: the last subflow was
        dropped.
        """
        packets = [_tcp(t) for t in (0, 1, 2, 12, 13, 14, 24, 25, 26)]
        subflows = self._subflows(packets, interval=5)
        self.assertEqual(_packet_times(subflows), [[0, 1, 2], [12, 13, 14], [24, 25, 26]])

    def test_no_packet_lost(self):
        """Every packet of the flow is in exactly one subflow (no subflow has fewer than 2
        packets here).

        Expected: all 11 packets, in order. Before the fix: the last 2 were dropped.
        """
        times = [0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41]
        subflows = self._subflows([_tcp(t) for t in times], interval=5)
        kept = [t for subflow_times in _packet_times(subflows) for t in subflow_times]
        self.assertEqual(kept, times)

    def test_unsplit_flow_unchanged(self):
        """Regression guard: a flow without gaps longer than the interval stays whole.

        Expected: [[0, 1, 2, 3]], before and after the fix.
        """
        subflows = self._subflows([_tcp(t) for t in (0, 1, 2, 3)], interval=5)
        self.assertEqual(_packet_times(subflows), [[0, 1, 2, 3]])


def _reference_timeout_split(pcap_file, timeout=600):
    """An independent implementation of the timeout split, for real captures.

    Groups the IPv4 TCP/UDP packets by five-tuple, sorts each group by time, cuts it wherever two
    consecutive packets are more than `timeout` seconds apart, and keeps the pieces with at least
    2 packets. Returns (number of flows, number of packets in them).
    """
    groups = {}
    with PcapReader(str(pcap_file)) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            for transport, proto in ((TCP, 6), (UDP, 17)):
                if transport in pkt:
                    key = (pkt[IP].src, pkt[IP].dst, pkt[transport].sport,
                           pkt[transport].dport, proto)
                    groups.setdefault(key, []).append(float(pkt.time))
                    break

    n_flows = n_packets = 0
    for times in groups.values():
        times.sort()
        piece = [times[0]]
        pieces = []
        for (previous, current) in zip(times, times[1:]):
            if current - previous > timeout:
                pieces.append(piece)
                piece = []
            piece.append(current)
        pieces.append(piece)
        kept = [p for p in pieces if len(p) >= 2]
        n_flows += len(kept)
        n_packets += sum(len(p) for p in kept)
    return n_flows, n_packets


class RealCaptureTests(RealCaptureTestCase):
    """Real captures from the course."""

    def test_log4j_timeout_split_matches_reference(self):
        """log4j.pcap (hands-on notebooks) has flows idle for more than 600 s.

        Expected: _pcap2flows() returns the same number of flows and packets as an independent
        implementation of the timeout split (4,790 flows, 24,947 packets).
        Before the fix: 4,766 flows and 24,864 packets (24 final subflows dropped).
        """
        pcap_file = require_capture(self, 'log4j.pcap')
        flows = capture_flows(pcap_file, tcp_timeout=600, udp_timeout=600)
        with quiet():
            expected = _reference_timeout_split(pcap_file, timeout=600)
        self.assertEqual((len(flows), sum(len(pkts) for _fid, pkts in flows)), expected)

    def test_demo_interval_split_loses_no_packet(self):
        """demo.pcap (Assignment 3) split with pcap2flows(q_interval=0.9):
        none of its subflows has fewer than 2 packets, so every packet of its flows must be kept.

        Expected: 5,984 packets, as many as in its flows before the interval split.
        Before the fix: 5,388 packets (14 final subflows dropped).
        """
        with quiet():
            flows = parser._pcap2flows(str(DEMO_PCAP), 2)
            pcap = PCAP(str(DEMO_PCAP))
            pcap.pcap2flows(q_interval=0.9)
        packets_in_flows = sum(len(pkts) for _fid, pkts in flows)
        packets_kept = sum(len(pkts) for _fid, pkts in pcap.flows)
        self.assertEqual(packets_kept, packets_in_flows)


if __name__ == '__main__':
    unittest.main()
