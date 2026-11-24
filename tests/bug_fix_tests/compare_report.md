# netml before vs after the four bug fixes

- **before**: `tests/before_bug_fix/src (upstream netml master, commit ef55aef)`
- **after**: `src (this repository, with the four fixes)`

## Fix 1: Packet size

**The bug:** Packet sizes were measured with len(pkt), the number of bytes saved in the pcap. A capture recorded with a small snaplen saves only the first bytes of each packet, so every larger packet was measured as the snaplen (e.g. 200 bytes).

**Tests** (`tests/bug_fix_tests/test_1_packet_size.py`): before 4 pass, 6 fail; after 10 pass.

| test | before | after |
|---|---|---|
| `FullCaptureTests.test_packets_built_in_memory` | pass | pass |
| `FullCaptureTests.test_untruncated_lengths_unchanged` | pass | pass |
| `RealCaptureTests.test_http_lengths_unchanged` | pass | pass |
| `RealCaptureTests.test_netflix_lengths_match_wireshark` | fail: AssertionError: 87199 of 141471 packet lengths differ (first: #8: 96 != 386, #10: 107 != 389, #11: 85 != 346, #12: 104 != 400, #13: 108 != 398) | pass |
| `TruncatedCaptureTests.test_iat_size_features` | fail: AssertionError: Lists differ: [200, 200, 200] != [1054, 1054, 1054] First differing element 0: 200 1054 - [200, 200, 200] + [1054, 1054, 1054] | pass |
| `TruncatedCaptureTests.test_pcap2pandas_length` | fail: AssertionError: Lists differ: [200, 200, 200] != [1054, 1054, 1054] First differing element 0: 200 1054 - [200, 200, 200] + [1054, 1054, 1054] | pass |
| `TruncatedCaptureTests.test_samp_size_features` | fail: AssertionError: Lists differ: [[600]] != [[3162]] First differing element 0: [600] [3162] - [[600]] + [[3162]] | pass |
| `TruncatedCaptureTests.test_size_features` | fail: AssertionError: Lists differ: [[200, 200, 200]] != [[1054, 1054, 1054]] First differing element 0: [200, 200, 200] [1054, 1054, 1054] - [[200, 200, 200]] + [[1054, 1054, 1054]] | pass |
| `TruncatedCaptureTests.test_stats_features` | fail: AssertionError: np.float64(200.0) != 1054 | pass |
| `TruncatedCaptureTests.test_test_data_is_truncated` | pass | pass |

**[synthetic] three 1054-byte TCP packets, saved at 200 bytes each (snaplen 200)**

- before: `length column: [200, 200, 200]`
- after: `length column: [1054, 1054, 1054]`
- Why the original output is wrong: each packet is 1054 bytes on the wire; 200 is only how much of it was saved, so every packet size was under-reported by 854 bytes.

**[synthetic] the same flow: SIZE features and STATS total bytes**

- before: `SIZE features [200, 200, 200]; STATS total bytes 600`
- after: `SIZE features [1054, 1054, 1054]; STATS total bytes 3,162`
- Why the original output is wrong: the features describe the 200-byte fragments in the file, not the traffic: the flow carried 3,162 bytes, not 600.

**[real] netflix.pcap (Assignment 1), recorded with a 200-byte snaplen, vs Wireshark**

- before: `87,199 of 141,471 packet lengths differ from Wireshark; largest 200 bytes; 21,131,157 bytes in all (Wireshark: 134,630,034)`
- after: `0 of 141,471 packet lengths differ from Wireshark; largest 1,514 bytes; 134,630,034 bytes in all (Wireshark: 134,630,034)`
- Why the original output is wrong: Wireshark reports the length on the wire, which the pcap records for every packet; the original sizes capped every large packet at 200 bytes and counted only a small fraction of the bytes really sent.

**[real] control: http.pcap (hands-on), recorded without a snaplen limit, vs Wireshark**

- before: `0 of 24,798 packet lengths differ from Wireshark; largest 1,514 bytes; 29,171,688 bytes in all (Wireshark: 29,171,688)`
- after: `0 of 24,798 packet lengths differ from Wireshark; largest 1,514 bytes; 29,171,688 bytes in all (Wireshark: 29,171,688)`
- No change, as expected (control: no packet was cut short, so the saved length is the length on the wire and both versions must agree with Wireshark)

## Fix 2: Flow splitting

**The bug:** Splitting a flow (by idle timeout in _pcap2flows, by interval in _flows2subflows) dropped its final subflow, and TCP/UDP was chosen with "6 in fid", which also matches port 6.

**Tests** (`tests/bug_fix_tests/test_2_flow_splitting.py`): before 3 pass, 7 fail; after 10 pass.

| test | before | after |
|---|---|---|
| `IntervalSplitTests.test_final_subflow_kept` | fail: AssertionError: Lists differ: [[0.0, 1.0, 2.0], [12.0, 13.0, 14.0]] != [[0, 1, 2], [12, 13, 14], [24, 25, 26]] Second list contains 1 additional elements. First extra element 2: [24, 25, 26] - [[0.... | pass |
| `IntervalSplitTests.test_no_packet_lost` | fail: AssertionError: Lists differ: [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 24.0, 25.0, 26.0] != [0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41] Second list contains 2 additional elements. First extra element 9: 40 ... | pass |
| `IntervalSplitTests.test_unsplit_flow_unchanged` | pass | pass |
| `RealCaptureTests.test_demo_interval_split_loses_no_packet` | fail: AssertionError: 5388 != 5984 | pass |
| `RealCaptureTests.test_log4j_timeout_split_matches_reference` | fail: AssertionError: Tuples differ: (4766, 24864) != (4790, 24947) First differing element 0: 4766 4790 - (4766, 24864) + (4790, 24947) | pass |
| `TimeoutSplitTests.test_final_subflow_kept` | fail: AssertionError: Lists differ: [[0.0, 1.0, 2.0]] != [[0, 1, 2], [702, 703, 704]] Second list contains 1 additional elements. First extra element 1: [702, 703, 704] - [[0.0, 1.0, 2.0]] + [[0, 1, 2], ... | pass |
| `TimeoutSplitTests.test_short_final_subflow_dropped` | pass | pass |
| `TimeoutSplitTests.test_split_in_three` | fail: AssertionError: Lists differ: [[0.0, 1.0], [700.0, 701.0]] != [[0, 1], [700, 701], [1400, 1401]] Second list contains 1 additional elements. First extra element 2: [1400, 1401] - [[0.0, 1.0], [700.... | pass |
| `TimeoutSplitTests.test_udp_port_6_uses_udp_timeout` | fail: AssertionError: Lists differ: [[0.0, 1.0, 101.0, 102.0]] != [[0, 1], [101, 102]] First differing element 0: [0.0, 1.0, 101.0, 102.0] [0, 1] Second list contains 1 additional elements. First extra e... | pass |
| `TimeoutSplitTests.test_unsplit_flow_unchanged` | pass | pass |

**[synthetic] one TCP flow with packets at 0, 1, 2, 702, 703, 704 s; timeout 600 s**

- before: `packet times per flow: [[0.0, 1.0, 2.0]]`
- after: `packet times per flow: [[0.0, 1.0, 2.0], [702.0, 703.0, 704.0]]`
- Why the original output is wrong: the flow is idle for 700 s and is correctly split in two, but the second subflow (702-704 s) was then silently dropped.

**[synthetic] one TCP flow with packets at 0-2, 12-14 and 24-26 s; interval 5 s**

- before: `packet times per subflow: [[0.0, 1.0, 2.0], [12.0, 13.0, 14.0]]`
- after: `packet times per subflow: [[0.0, 1.0, 2.0], [12.0, 13.0, 14.0], [24.0, 25.0, 26.0]]`
- Why the original output is wrong: the flow is split in three, but the third subflow (24-26 s) was dropped.

**[synthetic] a UDP flow from port 6, packets at 0, 1, 101, 102 s; TCP timeout 600 s, UDP 50 s**

- before: `packet times per flow: [[0.0, 1.0, 101.0, 102.0]]`
- after: `packet times per flow: [[0.0, 1.0], [101.0, 102.0]]`
- Why the original output is wrong: "6 in fid" matched the port, so this UDP flow got the TCP timeout (600 s) and the 100-second gap did not split it.

**[real] log4j.pcap (hands-on): flows split by the 600 s timeout, vs an independent implementation of the same split**

- before: `4,766 flows, 24,864 packets (the split itself: 4,790 flows, 24,947 packets)`
- after: `4,790 flows, 24,947 packets (the split itself: 4,790 flows, 24,947 packets)`
- Why the original output is wrong: the original kept fewer flows and packets than the split itself produces: the final subflow of every split flow was lost.

**[real] demo.pcap (Assignment 3): pcap2flows(q_interval=0.9)**

- before: `196 subflows; 5,388 of 5,984 packets kept (1,933,334 bytes)`
- after: `210 subflows; 5,984 of 5,984 packets kept (2,000,406 bytes)`
- Why the original output is wrong: none of its subflows is too short to keep, so every packet should be kept; the missing packets are the dropped final subflows.

**[real] srcIP_10.42.0.1_normal.pcap (netml example): pcap2flows(q_interval=0.9)**

- before: `4,979 subflows; 34,169 of 35,355 packets kept (3,162,865 bytes)`
- after: `5,362 subflows; 35,340 of 35,355 packets kept (3,242,484 bytes)`
- Why the original output is wrong: the dropped final subflows: fewer subflows, packets and bytes reach the features and the model.

## Fix 3: IPv6

**The bug:** The network layer was looked up with scapy's IP class, which matches IPv4 only: IPv6 connections were merged under one placeholder flow id, IPv6 addresses were left empty, and header features crashed.

**Tests** (`tests/bug_fix_tests/test_3_ipv6.py`): before 1 pass, 6 fail, 4 error; after 11 pass.

| test | before | after |
|---|---|---|
| `ExtensionHeaderTests.test_header_features_with_extension_header` | error: AttributeError: proto | pass |
| `FlowIdTests.test_ipv4_unchanged` | pass | pass |
| `FlowIdTests.test_ipv6_tcp` | fail: AssertionError: Tuples differ: ('', '', -1, -1, -1) != ('2001:db8::1', '2001:db8::2', 1000, 80, 6) First differing element 0: '' '2001:db8::1' - ('', '', -1, -1, -1) + ('2001:db8::1', '2001:db8::2'... | pass |
| `FlowIdTests.test_ipv6_udp` | fail: AssertionError: Tuples differ: ('', '', -1, -1, -1) != ('2001:db8::3', '2001:db8::1', 5000, 53, 17) First differing element 0: '' '2001:db8::3' - ('', '', -1, -1, -1) + ('2001:db8::3', '2001:db8::1... | pass |
| `IPv6PcapTests.test_connections_not_merged` | fail: AssertionError: Lists differ: [('', '', -1, -1, -1)] != [('2001:db8::1', '2001:db8::2', 1000, 80, 6)[42 chars] 17)] First differing element 0: ('', '', -1, -1, -1) ('2001:db8::1', '2001:db8::2', 10... | pass |
| `IPv6PcapTests.test_header_features` | error: AttributeError: proto | pass |
| `IPv6PcapTests.test_pcap2pandas_addresses` | fail: AssertionError: Tuples differ: (None, None) != ('2001:db8::1', '2001:db8::2') First differing element 0: None '2001:db8::1' - (None, None) + ('2001:db8::1', '2001:db8::2') | pass |
| `IPv6PcapTests.test_pipeline_with_header_features` | error: AttributeError: proto | pass |
| `RealCaptureTests.test_header_features_run` | error: AttributeError: proto | pass |
| `RealCaptureTests.test_ipv6_flows_not_merged` | fail: AssertionError: ('', '', -1, -1, -1) unexpectedly found in [('192.168.43.72', '172.217.18.195', 58443, 443, 6), ('192.168.43.72', '216.58.209.228', 58444, 443, 6), ('192.168.43.72', '216.58.209.228... | pass |
| `RealCaptureTests.test_pcap2pandas_ipv6_addresses` | fail: AssertionError: 0 != 105 | pass |

**[synthetic] _get_fid() of one IPv6 TCP packet, 2001:db8::1:1000 -> 2001:db8::2:80**

- before: `flow id ('', '', -1, -1, -1)`
- after: `flow id ('2001:db8::1', '2001:db8::2', 1000, 80, 6)`
- Why the original output is wrong: an IPv6 TCP packet has a real five-tuple; the placeholder means "not TCP/UDP".

**[synthetic] one IPv6 TCP connection and one unrelated IPv6 UDP connection**

- before: `flow ('', '', -1, -1, -1): 6 packets`
- after: `flow ('2001:db8::1', '2001:db8::2', 1000, 80, 6): 3 packets; flow ('2001:db8::3', '2001:db8::1', 5000, 53, 17): 3 packets`
- Why the original output is wrong: two unrelated connections were merged into one fake flow, mixing their packets (and their features).

**[synthetic] header features (TCP flags, TTL) of the IPv6 TCP flow**

- before: `AttributeError: proto`
- after: `TCP flow: [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 57.0, 57.0, 57.0]`
- Why the original output is wrong: IPv6 has no "proto" or "ttl" field (it has "nh" and "hlim"), so the original crashed on any IPv6 packet.

**[real] netflix.pcap (Assignment 1): IPv6 flows found by _pcap2flows()**

- before: `0 IPv6 flows; 1 placeholder flow ('', '', -1, -1, -1) holding 61 packets`
- after: `5 IPv6 flows; no placeholder flow`
- Why the original output is wrong: all the IPv6 connections of the capture were merged into one placeholder flow.

**[real] netflix.pcap: pcap2pandas() rows of the IPv6 packets with an address**

- before: `0 of the 105 IPv6 packets have their addresses`
- after: `105 of the 105 IPv6 packets have their addresses`
- Why the original output is wrong: IPv6 packets were left without source and destination addresses.

**[real] netflix.pcap: pcap2flows() then flow2features(header=True)**

- before: `AttributeError: proto`
- after: `features extracted: 188 flows x 710`
- Why the original output is wrong: the IPv6 packets crashed the header features, so no features at all could be extracted from this capture.

## Fix 4: pcap2pandas

**The bug:** pcap2pandas() labeled only TCP/UDP/ICMP, stored DNS values as tuples (and crashed on invalid UTF-8), kept time as scapy EDecimal, cut datetime to whole seconds in local time, and measured time_normed from the first row instead of the earliest packet.

**Tests** (`tests/bug_fix_tests/test_4_pcap2pandas.py`): before 2 pass, 12 fail, 2 error; after 16 pass.

| test | before | after |
|---|---|---|
| `DnsColumnTests.test_non_dns_packet` | pass | pass |
| `DnsColumnTests.test_query_is_string` | fail: AssertionError: Lists differ: [('example.com.',), ('example.com.',), ('www.example.com.',), ('example.com.',)] != ['example.com.', 'example.com.', 'www.example.com.', 'example.com.'] First differin... | pass |
| `DnsColumnTests.test_response_is_string` | fail: AssertionError: Lists differ: [None, ('93.184.216.34',), ('example.com.',), ([b'hello', b'world'],)] != [None, '93.184.216.34', 'example.com.', 'hello world'] First differing element 1: ('93.184.21... | pass |
| `EmptyPcapTests.test_empty_pcap` | pass | pass |
| `InvalidDnsNameTests.test_invalid_utf8_does_not_crash` | error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte | pass |
| `ProtocolLabelTests.test_explicit_protocols` | fail: (packet='Ether / IPv6 / ICMPv6 Echo Request (id: 0x0 seq: 0x0)'): AssertionError: None != 'ICMPv6' | pass |
| `ProtocolLabelTests.test_protocols_named_from_their_number` | fail: (packet='Ether / 127.0.0.1 > 127.0.0.1 igmp / Raw'): AssertionError: Lists differ: [None] != ['IGMP'] First differing element 0: None 'IGMP' - [None] + ['IGMP'] | pass |
| `RealCaptureTests.test_datetime_keeps_fraction_of_second` | error: ValueError: unit='s' not valid with non-numerical val='1664501082.043476', at position 0 | pass |
| `RealCaptureTests.test_dns_columns_are_strings` | fail: AssertionError: False is not true | pass |
| `RealCaptureTests.test_protocols_google_home` | fail: AssertionError: {'TCP[18 chars]405, None: 337, 'ICMP': 324} != {'TCP[18 chars]405, 'ICMP': 324, 'ARP': 295, 'EAPOL': 22, 'IC[25 chars]': 4} - {None: 337, 'ICMP': 324, 'TCP': 93117, 'UDP': 1405} + {... | pass |
| `RealCaptureTests.test_protocols_nestcam_live` | fail: AssertionError: {'TCP': 29389, None: 80} != {'TCP': 29389, 'ARP': 76, 'EAPOL': 4} - {None: 80, 'TCP': 29389} + {'ARP': 76, 'EAPOL': 4, 'TCP': 29389} | pass |
| `RealCaptureTests.test_protocols_tplink_switch` | fail: AssertionError: {'TCP': 142, None: 34, 'UDP': 32, 'ICMP': 1} != {'TCP': 142, 'UDP': 32, 'ARP': 18, 'EAPOL': 14, None: 2, 'ICMP': 1} - {None: 34, 'ICMP': 1, 'TCP': 142, 'UDP': 32} ? ^ + {None: 2, 'A... | pass |
| `TimeColumnTests.test_datetime_full_precision_utc` | fail: AssertionError: 'datetime64[ns]' != 'datetime64[ns, UTC]' - datetime64[ns] + datetime64[ns, UTC] ? +++++ | pass |
| `TimeColumnTests.test_datetime_independent_of_local_time_zone` | fail: AssertionError: Series are different Series values are different (100.0 %) [index]: [0, 1, 2, 3] [left]: [1667523662000000000, 1667523663000000000, 1667523662000000000, 1667523664000000000] [right]... | pass |
| `TimeColumnTests.test_time_is_float` | fail: AssertionError: dtype('O') != <class 'numpy.float64'> | pass |
| `TimeColumnTests.test_time_normed_from_earliest_packet` | fail: AssertionError: dtype('O') != <class 'numpy.float64'> | pass |

**[synthetic] one ARP, ICMPv6, EAPOL, IGMP (IP protocol 2) and TCP packet**

- before: `labels: [None, None, None, None, 'TCP']`
- after: `labels: ['ARP', 'ICMPv6', 'EAPOL', 'IGMP', 'TCP']`
- Why the original output is wrong: only TCP/UDP/ICMP were labeled: real, common protocols were reported as None.

**[synthetic] a DNS response: example.com A 93.184.216.34**

- before: `dns_query = ('example.com.',); dns_resp = ('93.184.216.34',)`
- after: `dns_query = 'example.com.'; dns_resp = '93.184.216.34'`
- Why the original output is wrong: DNS values were 1-element tuples, which do not compare or group as strings (e.g. df.dns_query == "example.com." never matches).

**[synthetic] a DNS query whose name is not valid UTF-8 (malformed traffic)**

- before: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte`
- after: `dns_query = '��.example.com.'`
- Why the original output is wrong: one malformed packet aborted the parse of the whole capture.

**[synthetic] a packet received at 2022-11-03 16:01:02.794004 UTC**

- before: `time is EDecimal; datetime = 2022-11-03 11:01:02 (datetime64[ns]) [this machine's time zone: CST]`
- after: `time is float64; datetime = 2022-11-03 16:01:02.794004+00:00 (datetime64[ns, UTC]) [this machine's time zone: CST]`
- Why the original output is wrong: time was a column of scapy objects, and datetime lost the fraction of a second and was shifted to the local time zone of the machine running netml.

**[synthetic] packets stored out of time order, at 62.794, 63.000, 62.5, 64.25 s**

- before: `time_normed: [0.0, 0.205997, -0.294004, 1.455996]`
- after: `time_normed: [0.294004, 0.500001, 0.0, 1.75]`
- Why the original output is wrong: time_normed was measured from the first row, not the earliest packet, giving a negative time and a wrong origin.

**[real] packets labeled None, per course capture**

- before: `labeled None: google_home 337, tplink_switch 34, nestcam_live 80, netflix 80, log4j 3, http 0`
- after: `labeled None: google_home 5, tplink_switch 2, nestcam_live 0, netflix 0, log4j 0, http 0`
- Why the original output is wrong: ARP, EAPOL, ICMPv6, IGMP, SCTP, ... packets were labeled None; the remaining None are protocols without any name.

**[real] http.pcap (hands-on): DNS queries stored as tuples**

- before: `282 of 282 DNS queries are tuples`
- after: `0 of 282 DNS queries are tuples`
- Why the original output is wrong: every DNS query was a tuple instead of a string.

**[real] http.pcap: datetimes that lost the fraction of a second**

- before: `24,798 of 24,798 datetimes lost their fraction of a second; time zone: none (local time)`
- after: `0 of 24,798 datetimes lost their fraction of a second; time zone: UTC`
- Why the original output is wrong: every packet's datetime was cut to whole seconds (in local time), so packets within the same second could not be told apart or ordered by datetime.

**[real] control: course captures whose first packet is not the earliest**

- before: `negative time_normed values: netflix 0, http 0, log4j 0, nestcam_live 0, google_home 0, tplink_switch 0`
- after: `negative time_normed values: netflix 0, http 0, log4j 0, nestcam_live 0, google_home 0, tplink_switch 0`
- No change, as expected (control: in every course capture the first packet is also the earliest, so time_normed is the same on both versions; the bug needs an out-of-order capture, shown by the synthetic case)

## Fix 5: Isolation Forest (fixed upstream)

**The bug:** netml's IF inverted pyod's anomaly scores, which pyod had already inverted, so normal flows got the highest scores and the AUC fell below 0.5. Fixed upstream in noise-lab/netml#40, so it is already fixed in "before": its test and case are a check that the fix holds, and must give the same result on both versions.

**Tests** (`tests/bug_fix_tests/test_5_isolation_forest.py`): before 3 pass; after 3 pass.

| test | before | after |
|---|---|---|
| `IsolationForestTests.test_anomalies_ranked_above_normal_flows` | pass | pass |
| `IsolationForestTests.test_scores_match_pyod` | pass | pass |
| `IsolationForestTests.test_test_data` | pass | pass |

**[real] control: IF on the labeled flows of demo.pcap (Assignment 3), on identical features**

- before: `AUC 0.595 (5 random seeds: 0.579 to 0.607; 0.5 is chance)`
- after: `AUC 0.595 (5 random seeds: 0.579 to 0.607; 0.5 is chance)`
- No change, as expected (control: the fix was merged upstream in noise-lab/netml#40, so both versions rank anomalies above normal flows, with an AUC above 0.5; netml 0.7.1 gave the mirror image, 1 - AUC)

## Summary

| fix | tests that fail before and pass after | cases whose output changed |
|---|---|---|
| 1. Packet size | 6 of 10 | 3 of 4 |
| 2. Flow splitting | 7 of 10 | 6 of 6 |
| 3. IPv6 | 10 of 11 | 6 of 6 |
| 4. pcap2pandas | 14 of 16 | 8 of 9 |
| 5. Isolation Forest (fixed upstream) | 0 of 3 | 0 of 1 |

