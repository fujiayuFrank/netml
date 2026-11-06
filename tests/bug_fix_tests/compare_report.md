# netml before vs after the bug fixes

before: /home/frank/netml/before_bug_fix (original code, commit 106d5a1)  
after:  /home/frank/netml (current code)

## 1. Behaviour on inputs that trigger each bug

### packet length with a small snaplen
- input:  3 IPv4 TCP packets of 1054 bytes, recorded with snaplen 200
- before: length column: [200, 200, 200]
- after:  length column: [1054, 1054, 1054]

### IPv6 flows
- input:  1 IPv6 TCP + 1 IPv6 UDP connection
- before: 1 flow(s): [('', '', -1, -1, -1)]
- after:  2 flow(s): [('2001:db8::1', '2001:db8::2', 1000, 80, 6), ('2001:db8::3', '2001:db8::1', 5000, 53, 17)]

### IPv6 header features
- input:  same pcap, flow2features(header=True)
- before: AttributeError: proto
- after:  features shape (2, 12)

### pcap2pandas protocol labels
- input:  TCP, UDP, ICMP, ARP, ICMPv6, GRE packets
- before: protocol column: ['TCP', 'UDP', 'ICMP', None, None, None]
- after:  protocol column: ['TCP', 'UDP', 'ICMP', 'ARP', 'ICMPv6', 'GRE']

### pcap2pandas columns
- input:  DNS query for example.com at t=1700000000.123456
- before: dns_query=('example.com.',); time is EDecimal; datetime=2023-11-14 16:13:20
- after:  dns_query='example.com.'; time is float64; datetime=2023-11-14 22:13:20.123456+00:00

### pcap2pandas time_normed
- input:  packets at 10, 5, 12 s (earliest is not first)
- before: time_normed: [0.0, -5.0, 2.0]
- after:  time_normed: [5.0, 0.0, 7.0]

### idle-timeout split drops the last subflow
- input:  TCP packets at 0,1,2,702,703,704 s; timeout 600 s
- before: packet times per flow: [[0.0, 1.0, 2.0]]
- after:  packet times per flow: [[0.0, 1.0, 2.0], [702.0, 703.0, 704.0]]

### interval split drops the last subflow
- input:  same packets, split into 100 s intervals
- before: packet times per subflow: [[0.0, 1.0, 2.0]]
- after:  packet times per subflow: [[0.0, 1.0, 2.0], [702.0, 703.0, 704.0]]

### "6 in fid": UDP flow from port 6 treated as TCP
- input:  UDP sport 6 at 0,1,2,302,303,304 s; tcp_timeout 600, udp_timeout 200
- before: packet times per flow: [[0.0, 1.0, 2.0, 302.0, 303.0, 304.0]]
- after:  packet times per flow: [[0.0, 1.0, 2.0], [302.0, 303.0, 304.0]]

### header features assume Ethernet/IPv4/TCP
- input:  VLAN-tagged TCP flow: SYN, ACK, PSH+ACK
- before: header features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 64.0, 64.0, 64.0]
- after:  header features: [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 64.0, 64.0, 64.0]

### learn with several pcaps
- input:  netml learn --pcap-normal normal.pcap --pcap-abnormal anomaly.pcap
- before: exit 1: [ValueError] all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 12 and...
- after:  exit 0 (ok)

### classify a pcap other than the training one
- input:  learn on demo.pcap, classify srcIP_10.42.0.1_normal.pcap
- before: exit 1: [ValueError] X has 12 features, but OCSVM is expecting 51 features as input.
- after:  exit 0 (ok)

### classify with IF
- input:  learn + classify demo.pcap, --algorithm if
- before: exit 1: [NotFittedError] This IF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
- after:  classified 210 flows: 21 ANOMALY

### classify with KDE
- input:  same, --algorithm kde
- before: exit 1: [NotFittedError] This KDE instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
- after:  classified 210 flows: 21 ANOMALY

### classify with GMM
- input:  same, --algorithm gmm
- before: exit 1: [AttributeError] 'GMM' object has no attribute 'means_'
- after:  classified 210 flows: 21 ANOMALY

### classify with PCA
- input:  same, --algorithm pca
- before: exit 1: [NotFittedError] This PCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
- after:  classified 210 flows: 21 ANOMALY

### IF anomaly scores inverted
- input:  200 normal points + 20 obvious anomalies (same data for both)
- before: AUC 0.004 (1 = anomalies scored highest, 0.5 = chance, 0 = inverted)
- after:  AUC 0.996 (1 = anomalies scored highest, 0.5 = chance, 0 = inverted)

### learn train -f FILE empties FILE
- input:  learn extract -f F, then learn train -f F
- before: exit 1: [EOFError] Ran out of input; features file 80133 -> 0 bytes
- after:  exit 0 (ok); features file 85972 -> 85972 bytes

### pcap without flows
- input:  netml learn -p arp_only.pcap
- before: exit 1: [IndexError] index -1 is out of bounds for axis 0 with size 0
- after:  exit 2: compare_before_after.py learn: error: no flows to extract features from: no TCP/UDP flow with at least 2 packets in /tmp/tmpctccizce/arp_only.pcap

### --param FILE
- input:  --param params.yaml containing "nu: 0.2"
- before: exit 1: [AttributeError] 'str' object has no attribute 'get'
- after:  exit 0 (ok); model nu = 0.2

### OCSVM default nu (fraction of flows flagged)
- input:  netml learn --algorithm ocsvm -p demo.pcap
- before: OCSVM nu = 0.5
- after:  OCSVM nu = 0.1

### learn --contamination
- input:  netml learn --contamination 0.05
- before: exit 2: compare_before_after.py learn: error: argument action: invalid choice: '0.05' (choose from 'extract', 'train')
- after:  exit 0 (ok)

### examples/capture/capture_pkts.py flow bookkeeping
- input:  capture() on packets A@0 s, B@1 s, A@4000 s
- before: RuntimeError: dictionary changed size during iteration
- after:  ok, 3 packets written

### shipped examples/out/data/OCSVM-results.dat
- input:  load_data(...)
- before: ModuleNotFoundError: No module named 'ndm'
- after:  file removed (ndm_demo.py writes it)

24 of 24 cases give a different output.

## 2. tests/bug_fix_tests/test_my_changes.py against each version

- before: 113 tests: 14 pass, 54 fail, 45 error, 0 skip
- after: 113 tests: 113 pass, 0 fail, 0 error, 0 skip

### Fail before, pass after (99 tests)

| test | before | why it failed before |
|---|---|---|
| CLIAllAlgorithmsTests.test_classify_old_pyod_model_file | error | AttributeError: 'KDE' object has no attribute 'threshold_' |
| CLIAllAlgorithmsTests.test_learn_and_classify | fail | (algorithm='ocsvm'): AssertionError: netml ('-v', '0', 'learn', '--algorithm', 'ocsvm', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '--pcap-abnormal', ... |
| CLIAllAlgorithmsTests.test_test_split_with_one_class_warns | fail | AssertionError: 'warning' not found in '' |
| CLIContaminationTests.test_default_is_one_tenth_for_every_algorithm | fail | (algorithm='ocsvm'): AssertionError: 0.5 != 0.1 |
| CLIContaminationTests.test_flagged_fraction_on_training_data | error | TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| CLIContaminationTests.test_option_and_param_together_is_an_error | fail | AssertionError: '--contamination' not found in "usage: compare_before_after.py learn [-h] [-p FILE] [-l FILE]\n [--pcap-normal FILE]\n [--pcap-abnormal FILE] [-f FILE]\n [-o FILE]\n [--algorithm {o... |
| CLIContaminationTests.test_out_of_range | fail | (value='0'): AssertionError: '--contamination' not found in "usage: compare_before_after.py learn [-h] [-p FILE] [-l FILE]\n [--pcap-normal FILE]\n [--pcap-abnormal FILE] [-f FILE]\n [-o FILE]\n [--algorithm {o... |
| CLIContaminationTests.test_param_value_out_of_range | fail | AssertionError: 0 != 2 |
| CLIContaminationTests.test_prints_value_and_meaning | fail | AssertionError: netml ('-v', '1', 'learn', '--algorithm', 'kde', '--contamination', '0.05', '-p', '/home/frank/netml/before_bug_fix/examples/data/demo.pcap', '-o', '/tmp/tmpzd3otqna/model') exited ... |
| CLIContaminationTests.test_sets_every_algorithm | fail | (algorithm='ocsvm'): AssertionError: 2 != 0 : usage: compare_before_after.py learn [-h] [-p FILE] [-l FILE] [--pcap-normal FILE] [--pcap-abnormal FILE] [-f FILE] [-o FILE] [--algorithm {ocsvm,kde,if,ae,gmm,pca}] [--hel... |
| CLIFeatureFileTests.test_bad_arguments_keep_existing_files | fail | AssertionError: b'' != b'\x80\x04\x95\x84\x00\x00\x00\x00\x00\x00\[313546 chars]x94.' |
| CLIFeatureFileTests.test_extract_and_train_with_feature_file | error | AttributeError: 'OCSVM' object has no attribute 'extraction_params' |
| CLIFeatureFileTests.test_failed_extract_keeps_existing_files | fail | AssertionError: b'' != b'\x80\x04\x95\x84\x00\x00\x00\x00\x00\x00\[313546 chars]x94.' |
| CLIFeatureFileTests.test_train_from_old_feature_file | fail | AssertionError: netml ('learn', 'train', '-f', '/tmp/tmp424zkcj5/features', '-o', '/tmp/tmp424zkcj5/model') exited with 1: |
| CLIFeatureFileTests.test_train_in_isolation | fail | AssertionError: netml ('learn', 'train', '-f', '/tmp/tmprhn2k788/features', '-o', '/tmp/tmprhn2k788/model') exited with 1: |
| CLIFeatureFileTests.test_train_missing_feature_file | fail | AssertionError: 1 != 2 |
| CLIParamTests.test_param_file | fail | AssertionError: 1 != 0 : |
| CLIParamTests.test_param_invalid_yaml | fail | AssertionError: 1 != 2 |
| CLIParamTests.test_param_json_file_per_algorithm | fail | AssertionError: 1 != 0 : |
| CLIParamTests.test_param_missing_file | fail | AssertionError: 1 != 2 |
| CLIParamTests.test_param_not_a_mapping | fail | AssertionError: 1 != 2 |
| CLIParamTests.test_param_unknown_name | fail | AssertionError: 'no_such_param' not found in '' |
| CaptureExampleTests.test_capture_long_idle_gap | error | RuntimeError: dictionary changed size during iteration |
| CaptureExampleTests.test_clean_up_ends_the_long_flow_not_the_current_one | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| CaptureExampleTests.test_idle_gap_longer_than_timeout_ends_the_flow | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| CaptureExampleTests.test_long_flow_ended_while_others_continue | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| CaptureExampleTests.test_new_flow | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| CaptureExampleTests.test_short_gaps_continue_the_flow | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| CaptureExampleTests.test_start_time_is_the_flows_own | error | AttributeError: module 'capture_pkts' has no attribute 'update_flows' |
| ExamplesTests.test_committed_demo_features_match_parser | fail | AssertionError: Arrays are not equal (shapes (88, 2), (88, 71) mismatch) ACTUAL: array([[0.000000e+00, 5.960464e-06], [0.000000e+00, 4.053116e-06], [0.000000e+00, 2.861023e-06],... DESIRED: array([... |
| ExamplesTests.test_committed_example_outputs_load | error | (path='examples/out/data/OCSVM-results.dat'): ModuleNotFoundError: No module named 'ndm' |
| ExamplesTests.test_demos_write_to_out_and_chain | fail | AssertionError: Lists differ: ['demo.csv', 'demo.pcap', 'demo_IAT.dat', 'demo_pandas.csv', 'srcIP_10.42.0.1'] != ['demo.csv', 'demo.pcap', 'srcIP_10.42.0.1'] First differing element 2: 'demo_IAT.da... |
| ExamplesTests.test_ndm_demo_missing_input_clear_message | fail | AssertionError: 'pparser_demo.py' not found in 'Traceback (most recent call last):\n File "/tmp/tmp8dijw6ns/examples/ndm/ndm_demo.py", line 91, in <module>\n main()\n File "/tmp/tmp8dijw6ns/example... |
| ExtractionSettingsCLITests.test_classify_other_pcap | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsCLITests.test_classify_with_old_model_file | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsCLITests.test_feature_file_still_unpacks_as_pair | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsCLITests.test_model_saves_settings | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsCLITests.test_multi_pcap_settings | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsCLITests.test_single_pcap_unchanged | error | setUpClass: AssertionError: netml ('learn', 'extract', '-f', '/tmp/tmprv4b_dcl/multi-features', '--pcap-normal', '/home/frank/netml/before_bug_fix/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap', '-... |
| ExtractionSettingsLibraryTests.test_pcap2flows_with_interval | error | AttributeError: 'PCAP' object has no attribute 'q_interval' |
| ExtractionSettingsLibraryTests.test_pkts_dim_fixes_feature_length | error | (feat_type='IAT'): TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ExtractionSettingsLibraryTests.test_shared_interval_is_quantile_of_all_flows | error | AttributeError: 'PCAP' object has no attribute 'extract_flows' |
| ExtractionSettingsLibraryTests.test_two_steps_equal_pcap2flows | error | AttributeError: 'PCAP' object has no attribute 'extract_flows' |
| HeaderLayoutTests.test_header_features | error | (layout='Ether/IPv6'): AttributeError: proto |
| HeaderLayoutTests.test_pcap2pandas | fail | (layout='Ether/IPv6'): AssertionError: np.False_ is not true |
| HeaderLayoutTests.test_pipeline_with_header | fail | (layout='Ether/IPv6'): AssertionError: Tuples differ: (-1, -1, -1) != (1111, 80, 6) First differing element 0: -1 1111 - (-1, -1, -1) + (1111, 80, 6) |
| ModelPredictTests.test_feature_length_mismatch_clear_error | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_if_auc_direction | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_old_pyod_model_clear_error | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_predict_anomalies_all_models | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_predictions_follow_scores | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_pyod_threshold_from_contamination | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| ModelPredictTests.test_scores_match_reference_implementations | error | setUpClass: TypeError: PCAP.flow2features() got an unexpected keyword argument 'pkts_dim' |
| NetflixTests.test_flow_sizes_exceed_snaplen | fail | AssertionError: 200 not greater than 200 |
| NetflixTests.test_ipv6_addresses_present | fail | AssertionError: 0 != 105 |
| NetflixTests.test_ipv6_flows_extracted | fail | AssertionError: 0 not greater than 0 |
| NetflixTests.test_lengths_exceed_snaplen | fail | AssertionError: np.int64(200) not greater than 200 |
| NetflixTests.test_lengths_match_wireshark | fail | AssertionError: Arrays are not equal Mismatched elements: 87199 / 141471 (61.6%) Max absolute difference among violations: 1314 Max relative difference among violations: 0.8678996 ACTUAL: array([77... |
| NetflixTests.test_pipeline_with_header | error | AttributeError: proto |
| NetflixTests.test_protocol_counts | fail | AssertionError: {'TCP[18 chars]232, None: 80, 'ICMP': 16} != {'TCP[18 chars]232, 'ICMPv6': 44, 'ARP': 26, 'ICMP': 16, 'IGM[13 chars]': 2} - {None: 80, 'ICMP': 16, 'TCP': 141143, 'UDP': 232} + {'ARP... |
| NoFlowsTests.test_classify_empty_pcap | fail | AssertionError: netml ('classify', '-m', '/tmp/tmph1tn2pr5/model', '-p', '/tmp/tmph1tn2pr5/arp-only.pcap') exited with 1: |
| NoFlowsTests.test_cli_empty_pcap_skipped_with_warning | fail | AssertionError: netml ('learn', 'extract', '-p', '/home/frank/netml/before_bug_fix/examples/data/demo.pcap', '-p', '/tmp/tmph1tn2pr5/arp-only.pcap', '-f', '/tmp/tmph1tn2pr5/features-mixed') exited ... |
| NoFlowsTests.test_cli_only_empty_pcap | fail | AssertionError: 1 != 2 |
| NoFlowsTests.test_flow2features_without_flows_is_2d | error | AttributeError: 'PCAP' object has no attribute 'extract_flows' |
| NoFlowsTests.test_label_file_matching_nothing_warns | fail | AssertionError: UserWarning not triggered |
| NoFlowsTests.test_library_clear_error | error | IndexError: index -1 is out of bounds for axis 0 with size 0 |
| PcapPandasTests.test_datetime_full_precision_utc | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_datetime_independent_of_local_time_zone | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_dns_invalid_utf8_does_not_crash | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_dns_query_is_string | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_dns_resp_is_string | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_empty_pcap | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_non_dns_packet | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_time_is_float | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| PcapPandasTests.test_time_normed_from_earliest_packet | error | setUpClass: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte |
| ProtocolTests.test_get_protocol | error | (expected='TCP', packet='Ether / IP / TCP 127.0.0.1:ftp_data > 127.0.0.1:http S'): AttributeError: module 'netml.pparser.parser' has no attribute '_get_protocol' |
| ProtocolTests.test_pcap2pandas_protocol_column | fail | AssertionError: Lists differ: ['TCP[13 chars]MP', None, None, None, None, None, None, None,[82 chars]None] != ['TCP[13 chars]MP', 'ICMPv6', 'ICMPv6', 'SCTP', 'GRE', 'ESP',[115 chars]None] First dif... |
| RealDataProtocolTests.test_log4j | fail | AssertionError: {'TCP': 62155, 'ICMP': 15682, 'UDP': 3003, None: 3} != {'TCP': 62155, 'ICMP': 15682, 'UDP': 3003, 'SCTP': 3} - {None: 3, 'ICMP': 15682, 'TCP': 62155, 'UDP': 3003} ? --------- + {'IC... |
| RealDataProtocolTests.test_nestcam | fail | AssertionError: {'TCP': 29389, None: 80} != {'TCP': 29389, 'ARP': 76, 'EAPOL': 4} - {None: 80, 'TCP': 29389} + {'ARP': 76, 'EAPOL': 4, 'TCP': 29389} |
| RealDataProtocolTests.test_uchicago_cs | fail | AssertionError: {'TCP': 26774, 'UDP': 294, None: 12} != {'TCP': 26774, 'UDP': 294, 'IGMP': 9, 'ARP': 3} - {None: 12, 'TCP': 26774, 'UDP': 294} + {'ARP': 3, 'IGMP': 9, 'TCP': 26774, 'UDP': 294} |
| SubflowSplitTests.test_interval_split_keeps_last_subflow | fail | AssertionError: Lists differ: [[0.0, 1.0, 2.0], [12.0, 13.0, 14.0]] != [[0, 1, 2], [12, 13, 14], [24, 25, 26]] Second list contains 1 additional elements. First extra element 2: [24, 25, 26] - [[0.... |
| SubflowSplitTests.test_no_packets_lost | fail | AssertionError: Lists differ: [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 24.0, 25.0, 26.0] != [0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41] Second list contains 2 additional elements. First extra element 9: 40 ... |
| SubflowSplitTests.test_tcp_port_17_uses_tcp_timeout | fail | AssertionError: Lists differ: [[0.0, 1.0]] != [[0, 1], [101, 102]] Second list contains 1 additional elements. First extra element 1: [101, 102] - [[0.0, 1.0]] + [[0, 1], [101, 102]] |
| SubflowSplitTests.test_timeout_split_into_three | fail | AssertionError: Lists differ: [[0.0, 1.0], [700.0, 701.0]] != [[0, 1], [700, 701], [1400, 1401]] Second list contains 1 additional elements. First extra element 2: [1400, 1401] - [[0.0, 1.0], [700.... |
| SubflowSplitTests.test_timeout_split_keeps_last_subflow | fail | AssertionError: Lists differ: [[0.0, 1.0, 2.0]] != [[0, 1, 2], [702, 703, 704]] Second list contains 1 additional elements. First extra element 1: [702, 703, 704] - [[0.0, 1.0, 2.0]] + [[0, 1, 2], ... |
| SubflowSplitTests.test_udp_port_6_uses_udp_timeout | fail | AssertionError: Lists differ: [[0.0, 1.0]] != [[0, 1, 101, 102]] First differing element 0: [0.0, 1.0] [0, 1, 101, 102] - [[0.0, 1.0]] + [[0, 1, 101, 102]] |
| SubflowSplitTests.test_unrelated_ipv6_connections_not_merged | fail | AssertionError: Lists differ: [('', '', -1, -1, -1)] != [('2001:db8::1', '2001:db8::2', 1000, 80, 6)[42 chars] 17)] First differing element 0: ('', '', -1, -1, -1) ('2001:db8::1', '2001:db8::2', 10... |
| SyntheticTests.test_get_IAT_SIZE_uses_wire_length | fail | AssertionError: Lists differ: [200, 200, 200] != [1054, 1054, 1054] First differing element 0: 200 1054 - [200, 200, 200] + [1054, 1054, 1054] |
| SyntheticTests.test_get_SAMP_SIZE_uses_wire_length | fail | AssertionError: Lists differ: [600] != [3162] First differing element 0: 600 3162 - [600] + [3162] |
| SyntheticTests.test_get_SIZE_uses_wire_length | fail | AssertionError: Lists differ: [200, 200, 200] != [1054, 1054, 1054] First differing element 0: 200 1054 - [200, 200, 200] + [1054, 1054, 1054] |
| SyntheticTests.test_get_STATS_uses_wire_length | fail | AssertionError: 600 != 3162 |
| SyntheticTests.test_get_fid_ipv6_tcp | fail | AssertionError: Tuples differ: ('', '', -1, -1, -1) != ('2001:db8::1', '2001:db8::2', 2222, 443, 6) First differing element 0: '' '2001:db8::1' - ('', '', -1, -1, -1) + ('2001:db8::1', '2001:db8::2... |
| SyntheticTests.test_get_fid_ipv6_udp | fail | AssertionError: Tuples differ: ('', '', -1, -1, -1) != ('2001:db8::1', '2001:db8::3', 5353, 53, 17) First differing element 0: '' '2001:db8::1' - ('', '', -1, -1, -1) + ('2001:db8::1', '2001:db8::3... |
| SyntheticTests.test_header_features_ipv6_tcp | error | KeyError: ('2001:db8::1', '2001:db8::2', 2222, 443, 6) |
| SyntheticTests.test_header_features_ipv6_udp | error | KeyError: ('2001:db8::1', '2001:db8::3', 5353, 53, 17) |
| SyntheticTests.test_pcap2flows_keeps_ipv6_flows_separate | fail | AssertionError: Items in the first set but not the second: ('', '', -1, -1, -1) Items in the second set but not the first: ('2001:db8::1', '2001:db8::3', 5353, 53, 17) ('2001:db8::1', '2001:db8::2'... |
| SyntheticTests.test_pcap2flows_packet_counts | fail | AssertionError: {('10[35 chars]3, ('', '', -1, -1, -1): 5, ('10.0.0.1', '10.0[19 chars]): 2} != {('10[35 chars]3, ('2001:db8::1', '2001:db8::2', 2222, 443, 6[92 chars]): 2} - {('', '', -1, -1, -1):... |
| SyntheticTests.test_pcap2pandas | fail | AssertionError: Lists differ: [200, 174, 112, 200, 42, 200, 174, 112, 200, 200, 174] != [1054, 174, 112, 542, 42, 1054, 174, 112, 542, 1054, 174] First differing element 0: 200 1054 - [200, 174, 11... |
| SyntheticTests.test_pipeline_with_header | error | AttributeError: proto |

### Pass before, fail after (0 tests)

none

### Pass in both versions (14 tests)

Behaviour that was already right and is kept: CLIContaminationTests.test_param_value_used_when_option_not_given, CLIContaminationTests.test_quiet_at_verbosity_0, CLIParamTests.test_param_inline, CaptureExampleTests.test_capture_end_to_end, HttpRegressionTests.test_lengths_match_wireshark, HttpRegressionTests.test_wire_length_equals_captured_length, SubflowSplitTests.test_interval_not_split, SubflowSplitTests.test_interval_single_packet_tail_dropped, SubflowSplitTests.test_timeout_not_split, SubflowSplitTests.test_timeout_single_packet_tail_dropped, SyntheticTests.test_fixture_is_truncated, SyntheticTests.test_get_SAMP_NUM_unchanged, SyntheticTests.test_get_fid_ipv4_unchanged, SyntheticTests.test_header_features_ipv4_tcp
