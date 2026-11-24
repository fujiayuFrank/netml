"""Tests of the four bug fixes (see "Bug fixes" in README.md), and of the Isolation Forest fix that
was merged upstream (noise-lab/netml#40).

One module per fix:

    test_1_packet_size        packet sizes use the length on the wire, not the captured length
    test_2_flow_splitting     splitting a flow keeps its final subflow; TCP/UDP by protocol number
    test_3_ipv6               IPv6 flow ids, addresses and header features
    test_4_pcap2pandas        pcap2pandas protocol labels and DNS/time columns
    test_5_isolation_forest   Isolation Forest ranks anomalies above normal flows (fixed upstream,
                              noise-lab/netml#40: passes on upstream master too)

Each module has synthetic tests (small pcaps built in the test, with exact expected values) and
tests on real captures from the course, which are shipped in tests/data (see tests/data/README.md).

Run them with `python -m tests.bug_fix_tests` (see __main__.py); add --synthetic-only to run only
the synthetic tests, in a few seconds.
"""
