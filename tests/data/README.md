# Test data

All the data the tests read: packet captures and their label / reference files. The tests read no data from anywhere else. Every file is an unchanged copy of its source.

## Course captures

Captures from the University of Chicago course *Machine Learning for Computer Systems* (CMSC 25422), used by the tests in `tests/bug_fix_tests/` to check the five bug fixes on real traffic.

| File | Size | Course source | Used by | Why |
|---|---|---|---|---|
| `netflix.pcap` | 23 MB | Assignment 1; hands-on notebooks data | fixes 1, 3 | Recorded with a 200-byte snaplen: 87,199 of its 141,471 packets were cut short. Contains IPv6 traffic (5 TCP/UDP flows, 105 packets). |
| `netflix.csv.gz` | 1.8 MB | hands-on notebooks data | fix 1 | Wireshark's packet list of `netflix.pcap`; its `Length` column is the reference packet length. |
| `http.pcap` | 29 MB | hands-on notebooks data | fixes 1, 4 | Recorded without a snaplen limit (the control for fix 1); 282 DNS queries. |
| `http.csv.gz` | 276 KB | hands-on notebooks data | fix 1 | Wireshark's packet list of `http.pcap`. |
| `log4j.pcap` | 8.0 MB | hands-on notebooks data | fix 2 | Has flows idle for longer than the 600 s timeout. |
| `nestcam_live.pcap` | 9.7 MB | hands-on notebooks data; Assignment 3 | fix 4 | ARP and EAPOL traffic. |
| `google_home.pcap` | 75 MB | Assignment 3 | fix 4 | ARP, EAPOL, ICMPv6 and IGMP traffic. |
| `tplink_switch.pcap` | 52 KB | Assignment 3 | fix 4 | ARP and EAPOL traffic. |
| `demo.pcap` | 2.1 MB | Assignment 3 (identical to netml's `examples/data/demo.pcap`) | fixes 2, 5; `tests/cli_tests` | Flows whose final subflows the interval split used to drop. |
| `demo.csv` | 3.1 MB | Assignment 3 (identical to netml's `examples/data/demo.csv`) | fix 5; `tests/cli_tests` | Labels (`BENIGN`, `ANOMALY`, `BOT`) of the flows of `demo.pcap`, in the CICIDS 2017 format. |

## netml example captures

Copies of netml's `examples/data/srcIP_10.42.0.1/` pcaps, used by the command-line tests in `tests/cli_tests/`.

| File | Size | Used by |
|---|---|---|
| `srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap` | 4.3 MB | `tests/cli_tests` (`--pcap-normal`) |
| `srcIP_10.42.0.1/srcIP_10.42.0.119_anomaly.pcap` | 604 KB | `tests/cli_tests` (`--pcap-abnormal`) |
