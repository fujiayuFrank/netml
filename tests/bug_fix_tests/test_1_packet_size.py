"""Fix 1: packet sizes use the packet's length on the wire, not the number of captured bytes.

The bug
    netml measured packets with ``len(pkt)``, the number of bytes *saved in the pcap*. A capture
    recorded with a small snaplen saves only the first bytes of each packet (e.g. 200), so every
    larger packet was measured as 200 bytes. The pcap record also stores the packet's original
    length, which scapy exposes as ``pkt.wirelen``; this is what Wireshark reports as the packet
    length.

Affected
    the packet-size features of flow2features() (SIZE, IAT_SIZE, STATS, SAMP_SIZE) and the
    "length" column of pcap2pandas().
"""
import unittest

import pandas as pd
from scapy.layers.inet import IP, TCP
from scapy.packet import Raw

from netml.pparser import parser
from netml.pparser.parser import PCAP

from ._helpers import (RealCaptureTestCase, TemporaryPcapTestCase, assert_same_values, at_time,
                       capture_dataframe, ether, quiet, require_capture, truncated)

SNAPLEN = 200

# a 1054-byte packet: Ethernet (14) + IPv4 (20) + TCP (20) + 1000 bytes of payload
WIRE_LENGTH = 14 + 20 + 20 + 1000
PKT_TIMES = (0.0, 0.01, 0.02)


def _large_packet():
    return (ether() / IP(src='10.0.0.1', dst='10.0.0.2') / TCP(sport=1111, dport=80)
            / Raw(b'x' * 1000))


class TruncatedCaptureTests(TemporaryPcapTestCase):
    """A synthetic capture with snaplen 200: three 1054-byte packets of one TCP flow, each saved
    as its first 200 bytes."""

    def setUp(self):
        super().setUp()
        packets = [truncated(_large_packet(), t, snaplen=SNAPLEN) for t in PKT_TIMES]
        self.pcap_file = self.write_pcap(packets)
        with quiet():
            self.flows = parser._pcap2flows(self.pcap_file, 2)

    def test_test_data_is_truncated(self):
        """Check of the test data itself: each pcap record saves 200 bytes and records 1054 as
        the original length."""
        (_fid, pkts) = self.flows[0]
        self.assertEqual([len(pkt) for pkt in pkts], [SNAPLEN] * 3)
        self.assertEqual([pkt.wirelen for pkt in pkts], [WIRE_LENGTH] * 3)

    def test_size_features(self):
        """SIZE features are the packets' lengths on the wire.

        Expected: [1054, 1054, 1054]. Before the fix: [200, 200, 200].
        """
        (features, _fids) = parser._get_SIZE(self.flows)
        self.assertEqual(features, [[WIRE_LENGTH] * 3])

    def test_iat_size_features(self):
        """IAT_SIZE features interleave inter-arrival times with lengths on the wire.

        Expected: [0.01, 1054, 0.01, 1054, 1054]. Before the fix: 200 in place of 1054.
        """
        (features, _fids) = parser._get_IAT_SIZE(self.flows)
        sizes = features[0][1::2] + [features[0][-1]]
        self.assertEqual(sizes, [WIRE_LENGTH] * 3)

    def test_stats_features(self):
        """STATS byte statistics are computed from lengths on the wire.

        Expected: mean, min and max 1054, and 3 * 1054 = 3162 bytes in total.
        Before the fix: 200 and 600.
        """
        (features, _fids) = parser._get_STATS(self.flows)
        stats = features[0]
        # [duration, pkts_rate, bytes_rate, mean, std, q1, q2, q3, min, max, num_pkts, num_bytes]
        self.assertEqual(stats[3], WIRE_LENGTH)
        self.assertEqual(stats[8], WIRE_LENGTH)
        self.assertEqual(stats[9], WIRE_LENGTH)
        self.assertEqual(stats[11], 3 * WIRE_LENGTH)

    def test_samp_size_features(self):
        """SAMP_SIZE features sum the lengths on the wire of the packets in each time window.

        All three packets fall in one 1-second window. Expected: [3162]. Before the fix: [600].
        """
        (features, _fids) = parser._get_SAMP_SIZE(self.flows, sampling_rate=1)
        self.assertEqual(features, [[3 * WIRE_LENGTH]])

    def test_pcap2pandas_length(self):
        """The "length" column of pcap2pandas() holds the lengths on the wire.

        Expected: [1054, 1054, 1054]. Before the fix: [200, 200, 200].
        """
        pcap = PCAP(self.pcap_file)
        with quiet():
            pcap.pcap2pandas()
        self.assertEqual(pcap.df['length'].tolist(), [WIRE_LENGTH] * 3)


class FullCaptureTests(TemporaryPcapTestCase):
    """Regression guard: when packets are saved in full, the length on the wire equals the
    captured length, so the fix changes nothing."""

    def test_untruncated_lengths_unchanged(self):
        """Expected: 1054 bytes for each packet, before and after the fix."""
        packets = [at_time(_large_packet(), t) for t in PKT_TIMES]
        pcap = PCAP(self.write_pcap(packets))
        with quiet():
            pcap.pcap2pandas()
        self.assertEqual(pcap.df['length'].tolist(), [WIRE_LENGTH] * 3)

    def test_packets_built_in_memory(self):
        """Packets built in memory (never read from a pcap) have no wire length; their size is
        their own length. Expected: 1054 for each packet, before and after the fix."""
        pkts = [at_time(_large_packet(), t) for t in PKT_TIMES]
        (features, _fids) = parser._get_SIZE([(('10.0.0.1', '10.0.0.2', 1111, 80, 6), pkts)])
        self.assertEqual(features, [[WIRE_LENGTH] * 3])


def _wireshark_lengths(csv_file):
    """The "Length" column of a Wireshark packet-list export (one row per packet)."""
    return pd.read_csv(csv_file)['Length'].tolist()


class RealCaptureTests(RealCaptureTestCase):
    """Real captures from the course, compared with Wireshark's packet lengths."""

    def _pcap2pandas_lengths(self, pcap_file):
        return capture_dataframe(pcap_file)['length'].tolist()

    def test_netflix_lengths_match_wireshark(self):
        """netflix.pcap (Assignment 1) was recorded with a 200-byte snaplen: 87,199 of its
        141,471 packets were cut short.

        Expected: every packet's length equals Wireshark's "Length" column (netflix.csv.gz).
        Before the fix: only 38.4% of the packets matched, and no length exceeded 200.
        """
        pcap_file = require_capture(self, 'netflix.pcap')
        csv_file = require_capture(self, 'netflix.csv.gz')
        assert_same_values(self, self._pcap2pandas_lengths(pcap_file),
                           _wireshark_lengths(csv_file), 'packet lengths')

    def test_http_lengths_unchanged(self):
        """Control: http.pcap (hands-on notebooks) was recorded without a snaplen limit, so no
        packet was cut short.

        Expected: every packet's length equals Wireshark's "Length" column (http.csv.gz), before
        and after the fix.
        """
        pcap_file = require_capture(self, 'http.pcap')
        csv_file = require_capture(self, 'http.csv.gz')
        assert_same_values(self, self._pcap2pandas_lengths(pcap_file),
                           _wireshark_lengths(csv_file), 'packet lengths')


if __name__ == '__main__':
    unittest.main()
