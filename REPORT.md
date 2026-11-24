# Fixing four bugs in NetML, and proving each fix on real traffic

**Jiayu Fu and Jimmy Li** · CMSC 25422 *Machine Learning for Computer Systems*, University of Chicago · Fall 2026
Repository: [github.com/fujiayuFrank/netml](https://github.com/fujiayuFrank/netml) (`master` branch)

---

## Summary

[NetML](https://github.com/noise-lab/netml) turns packet captures (pcaps) into flows, flows into feature vectors, and feature vectors into anomaly-detection models. We used it throughout the course, and found that it quietly loses or misreports information at several steps. We fixed four bugs, on top of the current upstream NetML (`master`, commit `ef55aef`), and for each one we built three kinds of evidence:

1. **a small synthetic pcap** that triggers the bug, with a test that fails on the original code and passes on ours;
2. **a measurement on real captures** from the course, comparing the original and the fixed library on the same data;
3. **one command** that reproduces all of it, side by side, and explains why each original output is wrong.

| # | Bug | Real-capture measurement, before → after |
|---|---|---|
| 1 | Packet sizes capped at the capture's snaplen | `netflix.pcap`: 87,199 of 141,471 packet sizes wrong; 21.1 MB of 134.6 MB counted → all correct |
| 2 | Splitting a flow dropped its final piece | `demo.pcap`: 596 of 5,984 packets lost → none; `log4j.pcap`: 24 flows lost → none |
| 3 | IPv6 traffic merged into one fake flow, or crashing | `netflix.pcap`: 5 IPv6 connections merged into 1, header features crash → 5 flows, features extracted |
| 4 | `pcap2pandas()` mislabels protocols and corrupts columns | 534 packets labeled `None` on 5 captures → 7; 282 DNS names stored as tuples → strings; every timestamp cut to the second → exact |

We also found a fifth bug, Isolation Forest ranking anomalies backwards (AUC 0.405 on the Assignment 3 demo, worse than chance, instead of 0.595). It has since been fixed upstream ([noise-lab/netml#40](https://github.com/noise-lab/netml/pull/40)), so it is not part of our changes; we keep its test as a regression check, skipped by default and run with its own command, which passes against upstream `master` (see [Fix 5](#fix-5--isolation-forest-ranked-anomalies-backwards-fixed-upstream)).

Of the 47 tests we wrote for the four fixes, 37 fail on the original code and pass on ours; the other 10 pass on both, on purpose (they guard behavior that was already correct). The 3 tests of the upstream Isolation Forest fix are a separate regression check (`python -m tests.bug_fix_tests --isolation-forest`), and pass on both. Everything reproduces with:

```
python -m tests.bug_fix_tests.compare_before_after
```

---

## 1. Background: how NetML works

```
 pcap ──► flows ──► subflows ──► feature vectors ──► model ──► anomaly scores
         (5-tuple,  (split at    (inter-arrival       (OCSVM, IF,
          timeout)   idle gaps)   times, sizes, ...)    KDE, GMM, ...)
```

- A **flow** is all packets with the same five-tuple *(source IP, destination IP, source port, destination port, protocol)*.
- Flows are cut into **subflows** wherever they are idle too long: first by a fixed timeout (600 s), then by an interval learned from the data.
- Each subflow becomes a **feature vector**, e.g. the gaps between its packets (`IAT`), the packet sizes (`SIZE`), or summary statistics (`STATS`).
- A separate function, `pcap2pandas()`, turns a pcap into a table with one row per packet, which the course used for exploration.

A bug early in this pipeline propagates to every later step, silently: nothing crashes, the numbers are just wrong. That is what made these bugs worth fixing, and what made *proving* the fixes the main part of the work.

---

## 2. How we worked

**The original code is kept next to the fixed code.** `tests/before_bug_fix/src/netml/` is an unchanged copy of upstream NetML `master` (commit `ef55aef`), the code our fixes are applied to. The comparison runs every test and measurement twice, each in its own Python process, once against each version, so the two never mix.

**Every bug gets a synthetic test and a real-capture test.**

- *Synthetic pcaps* are built packet by packet with scapy inside the test, so the expected output is known exactly (e.g. "three 1,054-byte packets"). They isolate one bug at a time, and cover edge cases that real traffic rarely contains.
- *Real captures* are the course's own data (hands-on notebooks, Assignment 1, Assignment 3), shipped in `tests/data/`. Where possible, the expected result comes from an **independent reference**, not from our fixed code: Wireshark's packet list, a separate implementation of flow splitting, or scapy's own layer counts.

**Every test documents itself.** Each docstring states what is checked, the expected result, and what the original code produced:

```python
def test_final_subflow_kept(self):
    """A TCP flow idle for 700 s between its third and fourth packets is split in two.

    Expected: [[0, 1, 2], [702, 703, 704]]. Before the fix: [[0, 1, 2]] (the packets after
    the split were dropped).
    """
```

**Scope was deliberately narrow.** We found more bugs than we fixed (see [Section 6](#6-found-but-not-fixed)). Each fix touches only its bug, and the diff against upstream contains no unrelated changes (no reformatting), so that each fix can be reviewed, and offered upstream, on its own.

---

## 3. The four fixes (and the one fixed upstream)

### Fix 1 — Packet sizes were capped at the snaplen

**Symptom.** In `netflix.pcap` (Assignment 1), no packet was ever larger than 200 bytes, even though video streaming sends full 1,514-byte packets.

**Was it the capture or the library?** Both, in a sense, and the distinction matters. Reading the pcap's raw record headers (without scapy) shows that the capture tool saved at most 200 bytes of each packet, so the *file* really is truncated. But every pcap record stores two lengths: how many bytes were saved, and how long the packet really was on the wire.

| `netflix.pcap` | |
|---|---|
| packets | 141,471 |
| largest number of bytes saved per packet | 200 |
| packets cut short (real length > saved length) | 87,199 (61.6%) |
| bytes saved in the file / real bytes on the wire | 21.1 MB / 134.6 MB (the file holds 15.7%) |
| Wireshark's "Length" column matches the real length | 100% of packets |
| … matches the saved length (what `len(pkt)` returns) | 38.4% |

The true sizes were in the file all along; NetML read the wrong field.

**Root cause.** Every size feature used `len(pkt)`, the number of *saved* bytes:

```python
# original: _get_SIZE, _get_IAT_SIZE, _get_STATS, _get_SAMP, and pcap2pandas' "length"
sizes = [len(pkt) for pkt in pkts]
```

**Fix.** Use the recorded length on the wire, which scapy exposes as `pkt.wirelen`:

```python
def _get_pkt_len(pkt):
    if pkt.wirelen is None:
        # a packet built in memory (not read from a pcap) has no recorded wire length: all of
        # its bytes are there, so its own length is its length on the wire
        return len(pkt)
    return pkt.wirelen
```

**Evidence.**

| Case | Before | After |
|---|---|---|
| synthetic: three 1,054-byte packets saved at 200 bytes | sizes `[200, 200, 200]`; flow total 600 bytes | `[1054, 1054, 1054]`; 3,162 bytes |
| `netflix.pcap` vs Wireshark | 87,199 of 141,471 sizes differ; largest 200; 21,131,157 bytes in all | 0 differ; largest 1,514; 134,630,034 bytes (= Wireshark) |
| *control:* `http.pcap` (recorded without a snaplen) vs Wireshark | 0 differ | 0 differ |

The control matters: on a capture that was never truncated, the fix changes nothing, as it should.

---

### Fix 2 — Splitting a flow dropped its final subflow

**Symptom.** Some packets never made it into any flow. On `demo.pcap` (Assignment 3), 596 of 5,984 packets (10%) disappeared during splitting.

**Root cause.** Both splitting functions build a list of finished subflows and keep the one in progress in `subflow_tmp`. A subflow is appended only when the *next* one starts, so after the loop the last subflow is still in `subflow_tmp`, and it was kept only if the flow had never been split:

```python
# original (_pcap2flows and _flows2subflows)
if not split_flow:
    subflows.append((fid, subflow_tmp))
else:
    # discard the last subflow_tmp
    pass
```

```
packets:   0 1 2 . . . . . . . . . . . . . . 702 703 704      (idle 700 s > timeout 600 s)
           └─────┘                           └───────────┘
          subflow 1                            subflow 2  ← silently discarded
```

A second, smaller bug sat in the same loops: TCP vs UDP was decided with `6 in fid`, which is true if *any* item of the five-tuple is 6, so a UDP flow on port 6 got the TCP timeout.

**Fix.**

```python
# always keep the last subflow; too-short subflows are discarded afterwards, like any other
subflows.append((fid, subflow_tmp))
```

```diff
-            if (6 in fid) or (TCP in pkt):
+            if fid[4] == 6:      # the protocol is the last item of the five-tuple
```

**Evidence.**

| Case | Before | After |
|---|---|---|
| synthetic: packets at 0, 1, 2, 702, 703, 704 s; timeout 600 s | `[[0, 1, 2]]` | `[[0, 1, 2], [702, 703, 704]]` |
| synthetic: UDP from port 6, packets at 0, 1, 101, 102 s; UDP timeout 50 s | `[[0, 1, 101, 102]]` (TCP timeout used) | `[[0, 1], [101, 102]]` |
| `log4j.pcap` vs an independent implementation of the timeout split (4,790 flows, 24,947 packets) | 4,766 flows, 24,864 packets | 4,790 flows, 24,947 packets |
| `demo.pcap`, `pcap2flows(q_interval=0.9)` | 196 subflows; 5,388 of 5,984 packets kept | 210 subflows; 5,984 of 5,984 |
| `srcIP_10.42.0.1_normal.pcap` (NetML's own example) | 4,979 subflows; 34,169 of 35,355 packets | 5,362 subflows; 35,340 of 35,355 |

An interesting side effect: NetML's own CLI tests had the buggy counts hard-coded (e.g. 88 labeled flows in `demo.pcap`; we get 102). The original test suite was certifying the bug. We updated those counts, with a comment explaining why.

---

### Fix 3 — IPv6 was not handled

**Symptom.** All IPv6 connections in a capture appeared as a single flow with the id `('', '', -1, -1, -1)`; `pcap2pandas()` had no addresses for IPv6 packets; and extracting header features from any capture with IPv6 traffic crashed with `AttributeError: proto`.

**Root cause.** NetML looked up the network layer with scapy's `IP` class, which is IPv4 only:

```python
# original _get_fid
if IP in pkt and TCP in pkt:
    fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
...
else:  # others
    fid = ('', '', -1, -1, -1)
```

**Fix.** The network layer is IPv4 *or* IPv6 (`_get_ip_layer()`), used for flow ids and `pcap2pandas()`. Header features read the IPv6 equivalents of the IPv4 fields:

```python
if isinstance(network, IPv6):
    # IPv6 names the carried protocol in "nh" (after any extension headers, so TCP is looked
    # up as a layer), and its hop limit ("hlim") is the IPv6 equivalent of the IPv4 TTL.
    tcp = network.getlayer(TCP)
    if tcp is not None:
        flgs_lst += np.asarray(_parse_tcp_flgs(tcp.flags)).reshape(-1, 1)
    header_features.append(network.hlim)
    continue
```

**Evidence.**

| Case | Before | After |
|---|---|---|
| synthetic: one IPv6 TCP and one unrelated IPv6 UDP connection | 1 fake flow `('', '', -1, -1, -1)` of 6 packets | 2 flows with their real five-tuples, 3 packets each |
| synthetic: header features of the IPv6 TCP flow | `AttributeError: proto` | flag counts `[0, 1, 0, 1, 2, 0, 0, 0]`, hop limits `[57, 57, 57]` |
| `netflix.pcap`: IPv6 flows | 0, plus 1 placeholder flow of 61 packets | 5 (= scapy's independent count) |
| `netflix.pcap`: IPv6 packets with addresses in `pcap2pandas()` | 0 of 105 | 105 of 105 |
| `netflix.pcap`: `flow2features(header=True)` | crash | 188 flows × 710 features |

---

### Fix 4 — `pcap2pandas()` mislabeled protocols and corrupted its columns

`pcap2pandas()` is the function students use first, so its output is what people *look at*. It had five problems:

| Column | Original | Fixed |
|---|---|---|
| `protocol` | only TCP, UDP, ICMP; everything else `None` | protocols checked explicitly (ARP, ICMPv6, EAPOL, SCTP, GRE, …), then named from the packet's own protocol number, then `None` |
| `dns_query`, `dns_resp` | 1-element tuples, e.g. `('example.com.',)`; invalid UTF-8 crashed the whole parse | strings; invalid bytes replaced |
| `time` | scapy `EDecimal` objects | floats |
| `datetime` | cut to whole seconds, in the local time zone of whoever runs NetML | exact, in UTC |
| `time_normed` | seconds since the *first row* | seconds since the *earliest packet* |

The protocol labeling keeps every label NetML gave before (TCP, UDP and ICMP are checked first, so e.g. TCP tunneled over GRE is still `TCP`), and adds the rest:

```python
if TCP in pkt:
    return 'TCP'
...
if ARP in pkt:
    return 'ARP'
...
# otherwise, name the protocol from the number in the packet's own headers
ip_proto = _get_ip_proto_number(pkt)
if ip_proto is not None:
    return _lookup_protocol_name(IP_PROTOS, ip_proto)        # e.g. 2 -> 'IGMP'
return _lookup_protocol_name(ETHER_TYPES, _get_ethertype(pkt))  # e.g. 0x88cc -> 'LLDP'
```

**Evidence.**

| Case | Before | After |
|---|---|---|
| synthetic: ARP, ICMPv6, EAPOL, IGMP, TCP packets | `[None, None, None, None, 'TCP']` | `['ARP', 'ICMPv6', 'EAPOL', 'IGMP', 'TCP']` |
| synthetic: DNS answer for example.com | `('example.com.',)`, `('93.184.216.34',)` | `'example.com.'`, `'93.184.216.34'` |
| synthetic: DNS name with invalid UTF-8 | `UnicodeDecodeError` (whole parse lost) | `'��.example.com.'` |
| synthetic: packet at 16:01:02.794004 UTC | `2022-11-03 11:01:02` (fraction lost, local time) | `2022-11-03 16:01:02.794004+00:00` |
| synthetic: packets stored out of order (62.794, 63.000, 62.5, 64.25 s) | `time_normed` `[0.0, 0.206, -0.294, 1.456]` | `[0.294, 0.500, 0.0, 1.75]` |
| packets labeled `None` per course capture | google_home 337, nestcam_live 80, netflix 80, tplink_switch 34, log4j 3 | 5, 0, 0, 2, 0 |
| `http.pcap`: DNS queries stored as tuples | 282 of 282 | 0 |
| `http.pcap`: datetimes that lost their fraction of a second | 24,798 of 24,798 | 0 |

**An honest caveat on `time_normed`.** We scanned every course capture (14 files) and none is out of order at its start, so on real course data the old and new `time_normed` agree. The closest case is `google_home.pcap`, which has 3 timestamps that step backwards by up to 37 µs (typical of a network card timestamping packets on several queues), but not at the first packet. The fix is correct and tested, but its evidence is the synthetic case; the real captures serve as a control here.

---

### Fix 5 — Isolation Forest ranked anomalies backwards (fixed upstream)

*We found and fixed this bug in NetML 0.7.1; the same fix has since been merged upstream in [noise-lab/netml#40](https://github.com/noise-lab/netml/pull/40) ("ndm.iforest: stop inverting IF scores a second time"). It is therefore not part of our changes, and the "before" code already includes it. Our test for it (`tests/bug_fix_tests/test_5_isolation_forest.py`) is kept as a regression check, skipped by default and run with `python -m tests.bug_fix_tests --isolation-forest` (or `compare_before_after --fix 5`): it passes against upstream `master`, and fails on 0.7.1. The numbers below compare 0.7.1 with the fixed code.*

**Symptom.** With `--algorithm if`, NetML reported an AUC below 0.5, i.e. worse than random guessing.

**Root cause.** pyod's convention, for every model, is "higher score = more anomalous". pyod's `IForest` already turns scikit-learn's scores round to follow it. NetML's wrapper turned them round a second time:

```python
# original IF.decision_function
return invert_order(self.model_.decision_function(X))
```

**Fix** (ours, and upstream's). Return pyod's scores as they are:

```python
return self.model_.decision_function(X)
```

**Evidence.** To isolate this bug from the feature fixes, both versions of the model are trained on *identical* features (the labeled flows of Assignment 3's `demo.pcap`), over 5 random seeds:

| | AUC |
|---|---|
| `IF` in NetML 0.7.1 | 0.405 (0.393–0.421) |
| fixed `IF` (upstream `master`, and ours) | 0.595 (0.579–0.607) |

The two add up to exactly 1, the signature of scores pointing the wrong way. A test also checks that our scores now equal pyod's own `IForest`. (0.595 is the *correct* score, but it is only modestly better than chance: the fix makes Isolation Forest rank anomalies the right way round; it does not make it a strong detector on this data.)

---

## 4. Reproducing everything

**Tested environment.** All results in this report were produced on **Ubuntu 26.04 (WSL2 on Windows)**, in a **conda** environment (conda 26.7) with **Python 3.11.17** and the package versions NetML requires:

| Package | Version |
|---|---|
| scapy | 2.5.0 |
| numpy | 2.0.2 |
| pandas | 2.2.3 |
| pyod | 2.0.7 |
| scikit-learn | 1.5.2 |
| netaddr | 1.3.0 |
| argcmdr, argparse-formatter, PyYAML, terminaltables (CLI) | 0.13.1, 1.4, 6.0.1, 3.1.10 |

To set up the same environment:

```
git clone https://github.com/fujiayuFrank/netml.git
cd netml
conda create -n netml-dev python=3.11
conda activate netml-dev
pip install --editable .[cli]
```

NetML supports Python 3.9–3.12. Use an environment like this one: in an unrelated environment, e.g. a base Python 3.14 installation, pandas 2.2 crashes (segmentation fault) while parsing a pcap; the comparison script detects this and says so.

Then, from the repository root:

```
python -m tests.bug_fix_tests.compare_before_after                   # fixes 1-4 (~3 minutes)
python -m tests.bug_fix_tests.compare_before_after --synthetic-only  # synthetic only (2 s)
python -m unittest                                   # the whole test suite (~2.5 minutes; skips the IF check)
python -m tests.bug_fix_tests --isolation-forest     # the Isolation Forest regression check (seconds)
python -m tests.bug_fix_tests.compare_before_after --fix 5           # ... compared on both versions
python -m tests.bug_fix_tests.compare_before_after --fix 1 2 3 4 5 --output tests/bug_fix_tests/compare_result.md
```

All data is in the repository (`tests/data/`, 155 MB, with a [README](tests/data/README.md) giving each file's source), so nothing needs to be downloaded or configured. For each fix, the comparison prints the bug, the test results on both versions, and every case with an explanation. An excerpt:

```
[real] log4j.pcap (hands-on): flows split by the 600 s timeout, vs an independent implementation of the same split
  before: 4,766 flows, 24,864 packets (the split itself: 4,790 flows, 24,947 packets)
  after:  4,790 flows, 24,947 packets (the split itself: 4,790 flows, 24,947 packets)
  Why the original output is wrong: the original kept fewer flows and packets than the split
  itself produces: the final subflow of every split flow was lost.
```

By default (fixes 1–4) it ends with this summary:

```
fix                      tests fail before, pass after     cases changed
1. Packet size                                 6 of 10            3 of 4
2. Flow splitting                              7 of 10            6 of 6
3. IPv6                                       10 of 11            6 of 6
4. pcap2pandas                                14 of 16            8 of 9

No unexpected results.
```

The two cases that did not change are the controls (`http.pcap` for fix 1, `time_normed` on in-order captures for fix 4), which must not change. The Isolation Forest check is left out by default; with `--fix 5` its 3 tests pass and its AUC is the same on both versions, since that fix is already in the "before" code. The script flags any unexpected result: a case that should change but did not, a control that changed, or a test failing on the fixed code. The full output of a run with all five (the last command above, which adds the Isolation Forest check: 0 of 3 tests and 0 of 1 cases change, as expected) is [`tests/bug_fix_tests/compare_result.md`](tests/bug_fix_tests/compare_result.md).

---

## 5. Making the tests trustworthy

A few design choices turned out to matter as much as the fixes:

- **Regression guards.** 9 of the tests pass on both versions by design: they check behavior that was already correct and must stay so. For example, "a flow with no long gaps stays whole" protects against the obvious-but-wrong fix for bug 2 (always appending the last subflow could duplicate an unsplit flow).
- **Independent references.** A test should always find the correct answer from external reference (e.g., from the pcap file directly) instead of comparing to the code's own output. Otherwise, the test is just comparing if the result is the same as last time instead of if the result is correct.
- **Tests must never hang.** Our first run froze for 40 minutes: comparing two 141,471-element lists with `assertEqual` makes unittest compute a full diff when they differ. Now long sequences are compared by counting mismatches, and every test has a time limit (60 s synthetic, 15 min real), so a slow test fails instead of hanging.
- **Fast feedback.** `--synthetic-only` runs the synthetic tests in about a second, so the fix logic can be checked on every edit; the real captures are for the final check.

---

## 6. Found but not fixed

We kept the deliverable to four fixes. These were found along the way and are documented in the [README](README.md#found-but-not-fixed) for future work, among them:

- `classify` only works with OCSVM: the other models never set their decision threshold.
- The autoencoder calls `nn.LeakyReLU(True)`, which sets the negative slope to 1, so the network is linear.
- Within a pcap, most feature values are padding: the feature length is the 90th percentile of packets per flow, so about 90% of flows are padded with zeros and the longest ones are cut.

Prototype fixes for several of these live on the `Experimental_Bug_Fix` branch. One more bug we had listed here, `netml learn` crashing on several pcaps with different feature lengths, has since been fixed upstream ([noise-lab/netml#42](https://github.com/noise-lab/netml/pull/42)); with it, two of NetML's CLI tests that we had marked as known failures now pass.

---

## 7. What we learned

- **"No crash" doesn't mean "correct".** Four of the five bugs we found produced plausible-looking output. They were found by asking *where did the bytes go?* and checking totals against an outside reference.
- **Check the data before blaming the code, and the code before blaming the data.** The 200-byte cap *was* in the capture, but the true sizes were in the same file. Both halves of that sentence were needed to fix it correctly.
- **Tests can encode bugs.** NetML's own tests had the wrong flow counts hard-coded.
- **Evidence is part of the fix.** For each bug, the most convincing artifact was not the code change (often one line) but the side-by-side output on real traffic.

---

## Acknowledgments

NetML is by Kun Yang, under the direction of Professor Samory Kpotufe, based on the [`odet`](https://github.com/Learn-Live/odet) library. The course captures come from CMSC 25422 at the University of Chicago. We used Claude (Anthropic's AI assistant, via Claude Code) to help investigate the bugs, write the fixes and tests, and draft this report; every result here comes from code in the repository and can be reproduced with the commands above.
