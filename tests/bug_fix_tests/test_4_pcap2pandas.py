"""Fix 4: pcap2pandas() labels every protocol it can name, and its DNS and time columns hold
plain, exact values.

The bugs and gaps
    a. "protocol": only TCP, UDP and ICMP were labeled; everything else (ARP, ICMPv6, IGMP,
       EAPOL, SCTP, ...) was None.
    b. "dns_query" / "dns_resp": 1-element tuples such as ('example.com.',) instead of strings;
       a DNS name that is not valid UTF-8 crashed the whole parse (UnicodeDecodeError).
    c. "time": scapy EDecimal objects in an object column, instead of floats.
    d. "datetime": cut to whole seconds, and in the local time zone of the machine running netml.
    e. "time_normed": measured from the *first row*, which is not the earliest packet when
       packets are stored out of time order (negative values), and computed with one lookup per
       row.
"""
import os
import time
import unittest

import numpy as np
import pandas as pd
from scapy.data import ETHER_TYPES, IP_PROTOS
from scapy.layers.dns import DNS, DNSQR, DNSRR
from scapy.layers.eap import EAPOL
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import ICMPv6EchoRequest, ICMPv6MLReport, IPv6, IPv6ExtHdrHopByHop
from scapy.layers.ipsec import AH, ESP
from scapy.layers.l2 import ARP, GRE, LLC, STP, Dot1Q, Dot3, Ether
from scapy.layers.ppp import PPPoED
from scapy.layers.sctp import SCTP
from scapy.layers.vrrp import VRRP
from scapy.packet import Raw

from netml.pparser.parser import PCAP

from ._helpers import (MAC_DST, MAC_SRC, RealCaptureTestCase, TemporaryPcapTestCase, at_time,
                       capture_dataframe, ether, quiet, require_capture)

# (expected label, packet), grouped by the part of the labeling that decides the label.
EXPLICIT_PROTOCOLS = [
    # protocols checked explicitly
    ('TCP', ether() / IP() / TCP()),
    ('UDP', ether() / IPv6() / UDP()),
    ('ICMP', ether() / IP() / ICMP()),
    ('ICMPv6', ether() / IPv6() / ICMPv6EchoRequest()),
    ('ICMPv6', ether() / IPv6() / IPv6ExtHdrHopByHop() / ICMPv6MLReport()),
    ('SCTP', ether() / IP() / SCTP()),
    ('GRE', ether() / IP() / GRE() / Raw(b'x' * 8)),
    ('ESP', ether() / IP() / ESP(spi=1, seq=1, data=b'x' * 16)),
    ('AH', ether() / IP() / AH(nh=59)),
    ('VRRP', ether() / IP(dst='224.0.0.18') / VRRP()),
    ('ARP', ether() / ARP()),
    ('ARP', ether() / Dot1Q() / ARP()),
    ('EAPOL', ether() / EAPOL()),
    ('STP', Dot3(src=MAC_SRC, dst=MAC_DST) / LLC() / STP()),
    ('PPPoE', ether() / PPPoED()),
    # the order of the checks keeps the labels netml gave before for these
    ('TCP', ether() / IP() / GRE() / IP() / TCP()),          # tunneled TCP: the inner protocol
    ('ICMP', ether() / IP() / ICMP(type=3) / IP() / TCP()),  # ICMP error quoting a TCP header
    # no name anywhere: still None
    (None, ether() / IP(proto=253) / Raw(b'x' * 8)),  # 253: reserved for experimentation
    (None, ether(type=0x88b5) / Raw(b'x' * 8)),       # 0x88b5: local experimental EtherType
]

# (table, number, packet): protocols scapy cannot dissect, named from the number in the packet's
# own headers. scapy loads the name tables from /etc/protocols and /etc/ethertypes, so the
# expected name is read from the same table (and the case is skipped if the table lacks it).
NUMBERED_PROTOCOLS = [
    (IP_PROTOS, 2, ether() / IP(proto=2) / Raw(b'\x11' * 8)),                       # IGMP
    (IP_PROTOS, 89, ether() / IP(proto=89) / Raw(b'x' * 8)),                        # OSPF
    (IP_PROTOS, 89, ether() / IPv6() / IPv6ExtHdrHopByHop(nh=89) / Raw(b'x' * 8)),  # OSPF
    (ETHER_TYPES, 0x88cc, ether(type=0x88cc) / Raw(b'x' * 8)),                      # LLDP
    (ETHER_TYPES, 0x88cc, ether() / Dot1Q(type=0x88cc) / Raw(b'x' * 8)),            # LLDP
]


def _as_read_from_pcap(pkt, pkt_time):
    """Turn a built packet into bytes and dissect it again, as when it is read from a pcap."""
    return at_time(Ether(bytes(pkt)), pkt_time)


class PandasTestCase(TemporaryPcapTestCase):

    def parse(self, packets):
        """pcap2pandas() of a pcap holding the packets."""
        pcap = PCAP(self.write_pcap(packets))
        with quiet():
            pcap.pcap2pandas()
        return pcap.df


class ProtocolLabelTests(PandasTestCase):
    """a. The "protocol" column."""

    def test_explicit_protocols(self):
        """Each packet is labeled with the protocol it carries.

        Expected: the labels listed in EXPLICIT_PROTOCOLS (e.g. 'ARP', 'ICMPv6', 'EAPOL').
        Before the fix: only 'TCP', 'UDP' and 'ICMP'; every other packet was None.
        """
        packets = [_as_read_from_pcap(pkt, 100 + i)
                   for (i, (_label, pkt)) in enumerate(EXPLICIT_PROTOCOLS)]
        labels = self.parse(packets)['protocol'].tolist()
        for ((expected, pkt), label) in zip(EXPLICIT_PROTOCOLS, labels):
            with self.subTest(packet=pkt.summary()):
                self.assertEqual(label, expected)

    def test_protocols_named_from_their_number(self):
        """Protocols scapy cannot dissect are named from the IP protocol number (IPv4 "proto",
        IPv6 "nh" after any extension headers) or the EtherType (also inside a VLAN tag).

        Expected: e.g. 'IGMP' for IP protocol 2, 'LLDP' for EtherType 0x88cc.
        Before the fix: None.
        """
        for (i, (table, number, pkt)) in enumerate(NUMBERED_PROTOCOLS):
            with self.subTest(packet=pkt.summary()):
                if number not in table:
                    self.skipTest(f'protocol number {number} has no name on this system')
                labels = self.parse([_as_read_from_pcap(pkt, 100 + i)])['protocol'].tolist()
                self.assertEqual(labels, [table[number].upper()])


def _dns(pkt_time, dns):
    return _as_read_from_pcap(ether() / IP(src='10.0.0.1', dst='10.0.0.3')
                              / UDP(sport=5353, dport=53) / dns, pkt_time)


class DnsColumnTests(PandasTestCase):
    """b. The "dns_query" and "dns_resp" columns."""

    def setUp(self):
        super().setUp()
        packets = [
            _dns(1, DNS(rd=1, qd=DNSQR(qname='example.com'))),
            _dns(2, DNS(qr=1, qd=DNSQR(qname='example.com'),
                        an=DNSRR(rrname='example.com', type='A', rdata='93.184.216.34'))),
            _dns(3, DNS(qr=1, qd=DNSQR(qname='www.example.com'),
                        an=DNSRR(rrname='www.example.com', type='CNAME', rdata='example.com'))),
            _dns(4, DNS(qr=1, qd=DNSQR(qname='example.com', qtype='TXT'),
                        an=DNSRR(rrname='example.com', type='TXT', rdata=['hello', 'world']))),
            at_time(ether() / IP(src='10.0.0.1', dst='10.0.0.2') / TCP(sport=1111, dport=80), 5),
        ]
        self.df = self.parse(packets)

    def test_query_is_string(self):
        """Expected: 'example.com.' (the query name). Before the fix: ('example.com.',)."""
        self.assertEqual(self.df['dns_query'].tolist()[:4],
                         ['example.com.', 'example.com.', 'www.example.com.', 'example.com.'])

    def test_response_is_string(self):
        """Expected, by record type: A '93.184.216.34', CNAME 'example.com.', TXT
        'hello world' (its strings joined), and None for a query without an answer.
        Before the fix: 1-element tuples."""
        self.assertEqual(self.df['dns_resp'].tolist()[:4],
                         [None, '93.184.216.34', 'example.com.', 'hello world'])

    def test_non_dns_packet(self):
        """Regression guard: a packet without DNS has is_dns False and no DNS values.
        Expected: False, None, None, before and after the fix."""
        row = self.df.iloc[4]
        self.assertEqual((bool(row['is_dns']), row['dns_query'], row['dns_resp']),
                         (False, None, None))


class InvalidDnsNameTests(PandasTestCase):
    """b. A DNS name that is not valid UTF-8 (malformed or malicious traffic)."""

    def test_invalid_utf8_does_not_crash(self):
        """Expected: the parse succeeds, and the invalid bytes are replaced in the string
        ('��.example.com.'). Before the fix: UnicodeDecodeError for the whole pcap."""
        df = self.parse([_dns(1, DNS(rd=1, qd=DNSQR(qname=b'\xff\xfe.example.com')))])
        self.assertEqual(df['dns_query'].tolist(), ['��.example.com.'])


# arrival times with microsecond digits; the earliest packet (2.5) is NOT the first one
PKT_TIMES = [1667491262.794004, 1667491263.000001, 1667491262.5, 1667491264.25]


class TimeColumnTests(PandasTestCase):
    """c, d, e. The "time", "datetime" and "time_normed" columns."""

    def setUp(self):
        super().setUp()
        self.packets = [at_time(ether() / IP(src='10.0.0.1', dst='10.0.0.2')
                                / TCP(sport=1111, dport=80), t) for t in PKT_TIMES]
        self.df = self.parse(self.packets)

    def test_time_is_float(self):
        """c. Expected: a float64 column equal to the arrival times (to the microsecond).
        Before the fix: an object column of scapy EDecimal values."""
        self.assertEqual(self.df['time'].dtype, np.float64)
        np.testing.assert_allclose(self.df['time'], PKT_TIMES, rtol=0, atol=1e-6)

    def test_datetime_full_precision_utc(self):
        """d. Expected: datetime64[ns, UTC]; the first packet is 2022-11-03 16:01:02.794004 UTC.
        Before the fix: 16:01:02 (fraction of a second lost), in the machine's local time."""
        self.assertEqual(str(self.df['datetime'].dtype), 'datetime64[ns, UTC]')
        self.assertEqual(self.df['datetime'].iloc[0],
                         pd.Timestamp('2022-11-03 16:01:02.794004', tz='UTC'))

    def test_datetime_independent_of_local_time_zone(self):
        """d. Expected: the same datetimes whatever the machine's time zone (here Tokyo and
        Chicago). Before the fix: they differed by the time-zone offset (14 hours here)."""
        old_tz = os.environ.get('TZ')
        try:
            os.environ['TZ'] = 'Asia/Tokyo'
            time.tzset()
            tokyo = self.parse(self.packets)['datetime']
            os.environ['TZ'] = 'America/Chicago'
            time.tzset()
            chicago = self.parse(self.packets)['datetime']
        finally:
            if old_tz is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = old_tz
            time.tzset()
        pd.testing.assert_series_equal(tokyo, chicago)

    def test_time_normed_from_earliest_packet(self):
        """e. Expected: seconds since the earliest packet, so the minimum is 0 and no value is
        negative: [0.294004, 0.500001, 0.0, 1.75].
        Before the fix: seconds since the first row: [0.0, 0.205997, -0.294004, 1.455996]."""
        expected = np.asarray(PKT_TIMES) - min(PKT_TIMES)
        self.assertEqual(self.df['time_normed'].dtype, np.float64)
        np.testing.assert_allclose(self.df['time_normed'], expected, rtol=0, atol=1e-6)


class EmptyPcapTests(PandasTestCase):
    """Regression guard: a pcap without packets."""

    def test_empty_pcap(self):
        """Expected: an empty DataFrame that has all the columns, before and after the fix."""
        df = self.parse([])
        self.assertEqual(len(df), 0)
        self.assertIn('time_normed', df.columns)


class RealCaptureTests(RealCaptureTestCase):
    """Real captures from the course (hands-on notebooks, Assignment 1, Assignment 3)."""

    def _pandas(self, name):
        return capture_dataframe(require_capture(self, name))

    def _assert_protocol_counts(self, name, expected):
        counts = self._pandas(name)['protocol'].value_counts(dropna=False).to_dict()
        counts = {(None if pd.isna(k) else k): v for (k, v) in counts.items()}
        self.assertEqual(counts, expected)

    def test_protocols_google_home(self):
        """google_home.pcap (Assignment 3). Expected: the counts below; 5 packets of protocols
        without a name remain None. Before the fix: ARP, EAPOL, ICMPv6 and IGMP packets were
        None (337 packets in all)."""
        self._assert_protocol_counts('google_home.pcap', {
            'TCP': 93117, 'UDP': 1405, 'ICMP': 324, 'ARP': 295, 'EAPOL': 22, 'ICMPv6': 11,
            None: 5, 'IGMP': 4})

    def test_protocols_tplink_switch(self):
        """tplink_switch.pcap (Assignment 3). Expected: the counts below. Before the fix: the
        ARP and EAPOL packets were None (34 packets in all)."""
        self._assert_protocol_counts('tplink_switch.pcap', {
            'TCP': 142, 'UDP': 32, 'ARP': 18, 'EAPOL': 14, None: 2, 'ICMP': 1})

    def test_protocols_nestcam_live(self):
        """nestcam_live.pcap (hands-on notebooks, Assignment 3). Expected: no packet is None.
        Before the fix: 80 packets (76 ARP, 4 EAPOL) were None."""
        self._assert_protocol_counts('nestcam_live.pcap', {'TCP': 29389, 'ARP': 76, 'EAPOL': 4})

    def test_dns_columns_are_strings(self):
        """http.pcap (hands-on notebooks). Expected: all 282 DNS queries are strings.
        Before the fix: all 282 were 1-element tuples."""
        queries = self._pandas('http.pcap')['dns_query'].dropna()
        self.assertEqual(len(queries), 282)
        self.assertTrue(all(isinstance(q, str) for q in queries))

    def test_datetime_keeps_fraction_of_second(self):
        """http.pcap. Expected: each datetime equals the packet's time (to the microsecond), in
        UTC. Before the fix: every datetime was cut to whole seconds, in local time."""
        df = self._pandas('http.pcap')
        expected = pd.to_datetime(df['time'], unit='s', utc=True)
        difference = (df['datetime'] - expected).abs().max()
        self.assertLessEqual(difference, pd.Timedelta(microseconds=1))
        self.assertTrue((df['datetime'].dt.microsecond != 0).any())


if __name__ == '__main__':
    unittest.main()
