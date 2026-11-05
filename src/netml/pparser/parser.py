"""PCAP parser

'pparser' parses pcaps to flow features by Scapy.

"""
import ipaddress
import itertools
import warnings
from collections import OrderedDict

import netaddr
import numpy as np
import pandas as pd

# scapy.all is imported for its side effect: it loads all of scapy's default protocol layers,
# which decides how packets are dissected (e.g. which encapsulations are decoded). The names
# used here are imported explicitly (no 'import *', which hid where names came from).
import scapy.all  # noqa: F401
from scapy.layers.dns import DNSQR, DNSRR
from scapy.utils import PcapReader

from scapy.data import ETHER_TYPES, IP_PROTOS
from scapy.layers.eap import EAPOL
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6, _ICMPv6, _IPv6ExtHdr
from scapy.layers.ipsec import AH, ESP
from scapy.layers.l2 import ARP, GRE, STP, CookedLinux, Dot1Q, Ether
from scapy.layers.ppp import PPPoE, PPPoED
from scapy.layers.sctp import SCTP
from scapy.layers.vrrp import VRRP, VRRPv3

from netml.utils.tool import data_info, timing


def _get_pkt_len(pkt):
    """Get the original length of a packet on the wire.

    ``len(pkt)`` is the number of *captured* bytes, which is smaller than the real packet
    when the pcap was recorded with a small snaplen (e.g. 200 bytes). The pcap record keeps
    the original length, which scapy exposes as ``pkt.wirelen``.

    Parameters
    ----------
    pkt:
        a packet
    Returns
    -------
        pkt_len: int
    """
    if pkt.wirelen is None:
        # packets built in memory (not read from a pcap) have no wire length
        return len(pkt)
    return pkt.wirelen


def _get_ip_layer(pkt):
    """Get the IPv4 or IPv6 layer of a packet, or None if it has neither."""
    # (getlayer looks each layer up once; "X in pkt" followed by pkt[X] would do it twice)
    ip = pkt.getlayer(IP)
    if ip is None:
        ip = pkt.getlayer(IPv6)
    return ip


def _get_ttl(pkt):
    """Get the IPv4 TTL, or the IPv6 hop limit (its IPv6 equivalent)."""
    ip = pkt.getlayer(IP)
    if ip is not None:
        return ip.ttl
    ipv6 = pkt.getlayer(IPv6)
    if ipv6 is not None:
        return ipv6.hlim
    return 0


def _get_fid(pkt):
    """Extract fid (five-tuple) from an IPv4 or IPv6 packet
    Parameters
    ----------

    Returns
    -------
        fid: five-tuple
    """
    ip = _get_ip_layer(pkt)
    if ip is None:  # others
        return ('', '', -1, -1, -1)

    tcp = pkt.getlayer(TCP)
    if tcp is not None:
        # flow_type = 'TCP'
        return (ip.src, ip.dst, tcp.sport, tcp.dport, 6)

    udp = pkt.getlayer(UDP)
    if udp is not None:
        # flow_type = 'UDP'
        return (ip.src, ip.dst, udp.sport, udp.dport, 17)

    return ('', '', -1, -1, -1)  # others


def _has_icmpv6(pkt):
    """True if the packet has an ICMPv6 layer.

    scapy has one class per ICMPv6 message type (ICMPv6EchoRequest, ICMPv6ND_NS, ...), all
    subclasses of _ICMPv6, and ``X in pkt`` only matches the exact class, so check the base class.
    """
    return any(issubclass(layer, _ICMPv6) for layer in pkt.layers())


def _get_ip_proto_number(pkt):
    """Get the IP protocol number (IPv4 'proto', IPv6 'nh'), or None if the packet is not IP.

    For IPv6, extension headers (hop-by-hop, routing, fragment, ...) are skipped: each one names
    the next header in its own 'nh' field, and the last one names the actual protocol.
    """
    if IP in pkt:
        return pkt[IP].proto
    if IPv6 not in pkt:
        return None

    ipv6 = pkt[IPv6]
    proto = ipv6.nh
    layer = ipv6.payload
    while isinstance(layer, _IPv6ExtHdr):
        proto = layer.nh
        layer = layer.payload
    return proto


def _get_ethertype(pkt):
    """Get the EtherType (the protocol carried by the link layer), or None.

    For VLAN-tagged frames, the EtherType inside the (first) VLAN tag is used.
    """
    if Dot1Q in pkt:
        return pkt[Dot1Q].type
    if Ether in pkt:
        return pkt[Ether].type
    if CookedLinux in pkt:  # Linux "any" interface captures
        return pkt[CookedLinux].proto
    return None


def _lookup_protocol_name(table, number):
    """Look up a protocol number in one of scapy's name tables (loaded from /etc/protocols and
    /etc/ethertypes). Returns the upper-cased name, or None if the number is not in the table."""
    if number is None or number not in table:
        return None
    return table[number].upper()


def _get_protocol_from_number(pkt):
    """Name the protocol from the numbers in the packet's own headers.

    IP packets: the IP protocol number, e.g. 2 -> 'IGMP', 89 -> 'OSPF'.
    Other frames: the EtherType, e.g. 0x88cc -> 'LLDP'.
    Returns None if the number has no name.
    """
    ip_proto = _get_ip_proto_number(pkt)
    if ip_proto is not None:
        # an IP packet: don't fall back to the EtherType, which would only say 'IPV4'/'IPV6'
        return _lookup_protocol_name(IP_PROTOS, ip_proto)

    return _lookup_protocol_name(ETHER_TYPES, _get_ethertype(pkt))


def _get_protocol(pkt):
    """Get the protocol name of a packet.

    1. Protocols checked explicitly, in this order (the first match wins). TCP and UDP come
       first, so tunneled TCP/UDP (e.g. over GRE) is labeled by the inner protocol, as before.
    2. Otherwise, the name of the protocol number in the packet's headers
       (see _get_protocol_from_number). This covers protocols scapy cannot dissect.
    3. Otherwise, None.
    """
    if TCP in pkt:
        protocol = 'TCP'
    elif UDP in pkt:
        protocol = 'UDP'
    elif ICMP in pkt:
        protocol = 'ICMP'
    elif _has_icmpv6(pkt):
        protocol = 'ICMPv6'
    elif SCTP in pkt:
        protocol = 'SCTP'
    elif GRE in pkt:
        protocol = 'GRE'
    elif ESP in pkt:
        protocol = 'ESP'
    elif AH in pkt:
        protocol = 'AH'
    elif (VRRP in pkt) or (VRRPv3 in pkt):
        protocol = 'VRRP'
    elif ARP in pkt:
        protocol = 'ARP'
    elif EAPOL in pkt:
        protocol = 'EAPOL'
    elif STP in pkt:
        protocol = 'STP'
    elif (PPPoED in pkt) or (PPPoE in pkt):
        protocol = 'PPPoE'
    else:
        protocol = _get_protocol_from_number(pkt)

    return protocol


def _get_frame_time(pkt):
    """Get packet arrival time

    Parameters
    ----------
    pkt:
        a packet
    Returns
    -------
        pkt_time: float
    """
    return float(pkt.time)


def _get_flow_duration(pkts):
    """Get flow duration

    Parameters
    ----------
    pkts: list
        a list of packets of a flow
    Returns
    -------
        flow_duration: float
    """
    pkt_times = [_get_frame_time(pkt) for pkt in pkts]
    flow_duration = max(pkt_times) - min(pkt_times)
    return flow_duration


def _pcap2flows(pcap_file, flow_pkts_thres=2, *, tcp_timeout=600, udp_timeout=600, verbose=0):
    """Extract flows. Only keep TCP and UDP flows, others are discarded.

    Parameters
    ----------
    pcap_file: str
        a pcap needed to processed.

    flow_pkts_thres: int (default is 2)
        the minimum number of packets of each flow is to control which flow should be kept
        and which one should be discarded. The default value is 2, i.e., all the flows which have
        less than 2 packets
        are discarded. It must be >= 2.

    tcp_timeout: int (default is 600s)
        a timeout is to split flow

    ucp_timeout: int (default is 600s)
        a timeout is to split flow

    verbose: int (default is 0)
        a print level is to control what information should be printed according to the given
        value.
        The higher the value is, the more info is printed.

    Returns
    -------
        flows: list

    """

    if verbose:
        print(f'pcap_file: {pcap_file}')

    # store all extracted flows into a dictionary, whose key is flow id ('fid': five-tuple) and
    # value is packtes
    # that belongs to the flow.
    flows = OrderedDict()
    try:
        # iteratively get each packet from the pcap; "with" closes the file when done
        with PcapReader(pcap_file) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                if (verbose > 3) and (i % 10000 == 0):
                    print(f'ith_packets: {i}')

                if (TCP in pkt) or (UDP in pkt):

                    # this function treats bidirection flows as two sessions (hereafter, we use
                    # sessions and flows interchangeably).
                    fid = _get_fid(pkt)

                    if fid not in flows.keys():
                        flows[fid] = [pkt]
                    else:
                        flows[fid].append(pkt)
                else:
                    continue

    except Exception as e:
        msg = f'Parse PCAP error: {e}!'
        raise RuntimeError(msg)

    if verbose > 3:
        print(f'len(flows): {len(flows.keys())}')

    # split flows by TIMEOUT and discard flows that have less than "flow_pkts_thres" packets.
    n_pkts = 0
    new_flows = []  # store the preprocessed flows
    for i, (fid, pkts) in enumerate(flows.items()):
        n_pkts += len(pkts)
        if len(pkts) < max(2, flow_pkts_thres):
            # discard flows that have less than "max(2, flow_pkts_thres)" packets
            continue

        # Is it necessary to sort packets by arrival_times?
        pkts = sorted(pkts, key=_get_frame_time, reverse=False)

        subflows = []
        # split flows by TIMEOUT
        for j, pkt in enumerate(pkts):
            pkt_time = _get_frame_time(pkt)
            if j == 0:
                subflow_tmp = [pkt]
                continue
            # the protocol is the last item of the fid: "6 in fid" would also match a port number
            if fid[4] == 6:
                # handle TCP packets, TCP is 6
                # a timeout (the idle time) is the duration between the previous pkt and the
                # current one.
                if pkt_time - _get_frame_time(subflow_tmp[-1]) > tcp_timeout:
                    subflows.append((fid, subflow_tmp))
                    # create a new subflow and store the current packet as the first packet of it.
                    subflow_tmp = [pkt]
                else:
                    subflow_tmp.append(pkt)
            elif fid[4] == 17:
                # handle UDP packets, UDP is 17
                if pkt_time - _get_frame_time(subflow_tmp[-1]) > udp_timeout:
                    subflows.append((fid, subflow_tmp))
                    subflow_tmp = [pkt]
                else:
                    subflow_tmp.append(pkt)
            else:  # it's not possible, because flows only include TCP and UDP flows
                pass

        # the last subflow (the whole flow, if it was not split by TIMEOUT); subflows with too few
        # packets are discarded below, like the others
        subflows.append((fid, subflow_tmp))

        new_flows.extend(subflows)

    new_flows = [(fid, pkts) for (fid, pkts) in new_flows if len(pkts) >= flow_pkts_thres]
    if verbose > 3:
        n_lt_2 = len([len(pkts) for fid, pkts in flows.items() if len(pkts) < flow_pkts_thres])
        n_gt_2 = len([len(pkts) for fid, pkts in flows.items() if len(pkts) >= flow_pkts_thres])
        print(f'total number of flows: {len(flows.keys())}. Num of flows < {flow_pkts_thres} '
              f'pkts: {n_lt_2}, and >={flow_pkts_thres} pkts: {n_gt_2} without timeout splitting.')
        print(
            f'kept flows: {len(new_flows)}. Each of them has at least {flow_pkts_thres} pkts '
            'after timeout splitting.')

    return new_flows


def _flows2subflows(flows, interval=10, *, flow_pkts_thres=2, verbose=0):
    """Split flows to subflows by interval

    Parameters
    ----------
    flows: list
      all flows needed to be split

    interval: float (default is 5.0s)
       a window is to split each flow

    flow_ptks_thres: int (default is 2)
        the minimum number of packets of each flow is to control which flow should be kept
        and which one should be discarded. The default value is 2, i.e., all the flows which have
        less than 2 packets
        are discarded. It must be >= 2.

    verbose: int (default is 0)
        a print level is to control what information should be printed according to the given
        value.
        The higher the value is, the more info is printed.

    Returns
    -------
    subflows: list
        each of subflow has at least "flow_ptks_thres" packets
    """

    new_flows = []  # store all subflows
    for i, (fid, pkts) in enumerate(flows):
        if (verbose > 3) and (i % 1000) == 0:
            print(f'{i}th_flow: len(pkts): {len(pkts)}')

        # Is it necessary to sort packets by arrival_times ?
        pkts = sorted(pkts, key=_get_frame_time, reverse=False)

        subflows = []
        # split flows by interval
        for j, pkt in enumerate(pkts):
            pkt_time = _get_frame_time(pkt)
            if j == 0:
                subflow_tmp_start_time = pkt_time
                subflow_tmp = [(subflow_tmp_start_time, pkt)]
                continue

            # the protocol is the last item of the fid: "6 in fid" would also match a port number
            if fid[4] == 6:
                # handle TCP packets, TCP is 6
                # a timeout (the idle time) is the duration between the previous pkt and the
                # current one.
                if pkt_time - subflow_tmp[-1][0] > interval:
                    subflows.append((fid, subflow_tmp))
                    subflow_tmp_start_time += (int((pkt_time - subflow_tmp_start_time) // interval)
                                               * interval)
                    # create a new subflow and store "subflow_tmp_start_time" as the time. Here,
                    # it will has a tiny difference of packet time between
                    # "subflow_tmp_start_time" and the current packet time.
                    subflow_tmp = [(subflow_tmp_start_time, pkt)]
                else:
                    subflow_tmp.append((pkt_time, pkt))

            elif fid[4] == 17:
                # handle UDP packets, UDP is 17
                if pkt_time - subflow_tmp[-1][0] > interval:
                    subflows.append((fid, subflow_tmp))
                    subflow_tmp_start_time += (int((pkt_time - subflow_tmp_start_time) // interval)
                                               * interval)
                    subflow_tmp = [(subflow_tmp_start_time, pkt)]
                else:
                    subflow_tmp.append((pkt_time, pkt))
            else:  # it's not possible, because flows only include TCP and UDP flows
                pass

        # the last subflow (the whole flow, if it was not split by interval); subflows with too few
        # packets are discarded below, like the others
        subflows.append((fid, subflow_tmp))

        new_flows.extend(subflows)

    # sort all flows by packet arrival time, each flow must have at least two packets
    subflows = []
    for fid, subflow_tmp in new_flows:
        if len(subflow_tmp) < max(2, flow_pkts_thres):
            continue
        subflows.append((fid, [pkt for pkt_time, pkt in subflow_tmp]))

    new_flows = subflows
    if verbose > 1:
        print(f'After splitting flows, the number of subflows: {len(new_flows)} and each of them '
              f'has at least {flow_pkts_thres} packets.')

    return new_flows


# bit i of the TCP flags field, for i = 0..7, is the flag F S R P A U E C (FIN, SYN, RST, PSH, ACK,
# URG, ECE, CWR): the order of the flag counts in the header features
_TCP_FLAG_BITS = np.arange(8)


def _count_tcp_flags(flag_values):
    """Count how often each TCP flag (F S R P A U E C) is set.

    Parameters
    ----------
    flag_values: list of int
        the TCP flags field of each packet

    Returns
    -------
        counts: numpy array of 8 floats
    """
    values = np.asarray(flag_values, dtype=np.int64).reshape(-1, 1)
    is_set = (values >> _TCP_FLAG_BITS) & 1
    return is_set.sum(axis=0).astype(float)


def _get_header_features(flows):
    """Extract header features which includes TCP Flags and TTL
    Parameters
    ----------

    Returns
    -------
       headers: a list
    """
    features = []
    for fid, pkts in flows:
        tcp_flag_values = []
        header_features = []
        for pkt in pkts:
            tcp = pkt.getlayer(TCP)
            if tcp is not None:
                tcp_flag_values.append(int(tcp.flags))
            header_features.append(_get_ttl(pkt))

        flag_counts = _count_tcp_flags(tcp_flag_values)  # 8 TCP flags, for the whole flow
        features.append(list(flag_counts) + header_features)

    return features


def _get_IAT(flows):
    """Extract interarrival times (IAT) features  from flows.
    Parameters
    ----------

    Returns
    -------
    features: a list
        iats
    fids: a list
        each value is five-tuple
    """
    features = []
    fids = []
    for fid, pkts in flows:
        pkt_times = [_get_frame_time(pkt) for pkt in pkts]
        # some packets have the same time, please double check the pcap.
        iats = list(np.diff(pkt_times))
        features.append(iats)
        fids.append(fid)

    return features, fids


def _get_SIZE(flows):
    """Extract packet sizes features from flows
    Parameters
    ----------

    Returns
    -------
    features: a list
        sizes
    fids: a list
        each value is five-tuple
    """

    features = []
    fids = []
    for fid, pkts in flows:
        sizes = [_get_pkt_len(pkt) for pkt in pkts]
        features.append(sizes)
        fids.append(fid)

    return features, fids


def _get_IAT_SIZE(flows):
    """Extract iats and sizes features from flows
    Parameters
    ----------

    Returns
    -------
    features: a list
        iats_sizes
    fids: a list
        each value is five-tuple
    """

    features = []
    fids = []
    for fid, pkts in flows:
        pkt_times = [_get_frame_time(pkt) for pkt in pkts]
        iats = list(np.diff(pkt_times))
        sizes = [_get_pkt_len(pkt) for pkt in pkts]
        iats_sizes = []
        for j in range(len(iats)):
            iats_sizes.extend([iats[j], sizes[j]])
        iats_sizes.append(sizes[-1])
        features.append(iats_sizes)
        fids.append(fid)

    return features, fids


def _get_STATS(flows):
    """get basic stats features, which includes duration, pkts_rate, bytes_rate, mean,
    median, std, q1, q2, q3, min, and max.

    Parameters
    ----------
    flows:

    Returns
    -------
    features: a list
        stats
    fids: a list
        each value is five-tuple
    """

    features = []
    fids = []
    for fid, pkts in flows:
        sizes = [_get_pkt_len(pkt) for pkt in pkts]

        sub_duration = _get_flow_duration(pkts)
        num_pkts = len(sizes)  # number of packets in the flow
        num_bytes = sum(sizes)  # all bytes in sub_duration: sum of wire lengths
        if sub_duration == 0:
            pkts_rate = 0.0
            bytes_rate = 0.0
        else:
            # it will be very larger due to the very small sub_duration
            pkts_rate = num_pkts / sub_duration
            bytes_rate = num_bytes / sub_duration

        # q should be [0,1] and q2 is np.median(data)
        q1, q2, q3 = np.quantile(sizes, q=[0.25, 0.5, 0.75])
        base_features = [sub_duration, pkts_rate, bytes_rate, np.mean(sizes), np.std(sizes),
                         q1, q2, q3, np.min(sizes), np.max(sizes), num_pkts, num_bytes]

        features.append(base_features)

        fids.append(fid)

    return features, fids


def _get_SAMP(flows, sampling_feature='SAMP_NUM', sampling_rate=0.1, verbose=0):
    """Extract sampling IATs from subwindows obtained by splitting each flow.

    For example, sampling_feature = 'SAMP_NUM'
        The length in time of the sub window is what we’re calling sampling rate.
           features obtained on sampling_rate = 0.1 means that:
            1) split each flow into small windows, each window has 0.1 duration
               (the length in time of each small window)
            2) obtain the number of packets from each window (0.1s).
            3) all the values obtained form each window make up of the features (SAMP_NUM).

    Parameters
    ----------
    flows: list
        all flows.

    sampling_feature: str
        'SAMP_NUM' or 'SAMP_SIZE'
        the feature we wants to extract from each flow.

    sampling_rate: float
        the duration of the window

    Returns
    -------

    features: a list
       SAMP features

    fids: a list
        each value is five-tuple

    """
    features = []
    fids = []
    for fid, pkts in flows:
        # for each flow
        each_flow_features = []
        pkt_times = [_get_frame_time(pkt) for pkt in pkts]
        pkt_sizes = [_get_pkt_len(pkt) for pkt in pkts]
        samp_sub = -1
        for i in range(len(pkts)):
            if i == 0:
                current = pkt_times[0]
                if sampling_feature == 'SAMP_NUM':
                    samp_sub = 1
                elif sampling_feature == 'SAMP_SIZE':
                    samp_sub = pkt_sizes[0]
                continue
            if pkt_times[i] - current <= sampling_rate:  # interval
                if sampling_feature == 'SAMP_NUM':
                    samp_sub += 1
                elif sampling_feature == 'SAMP_SIZE':
                    samp_sub += pkt_sizes[i]
                else:
                    print(f'{sampling_feature} is not implemented yet')
            else:  # if times[i]-current > sampling_rate:    # interval
                current += sampling_rate
                each_flow_features.append(samp_sub)
                # the time diff between times[i] and times[i-1] will be larger than mutli-samplings
                # for example, times[i]=10.0s, times[i-1]=2.0s, sampling=0.1,
                # for this case, we should insert int((10.0-2.0)//0.1) * [0]
                num_intervals = int(np.floor((pkt_times[i] - current) // sampling_rate))
                if num_intervals > 0:
                    num_intervals = min(num_intervals, 500)
                    each_flow_features.extend([0] * num_intervals)
                    current += num_intervals * sampling_rate
                # avoid num_features too large to excess the memory.
                if len(each_flow_features) > 500:
                    # return fid, each_flow_features[:500]
                    samp_sub = -1
                    each_flow_features = each_flow_features[:500]
                    break

                if sampling_feature == 'SAMP_NUM':
                    samp_sub = 1
                elif sampling_feature == 'SAMP_SIZE':
                    samp_sub = pkt_sizes[i]

        if samp_sub > 0:  # handle the last sub period in the flow.
            each_flow_features.append(samp_sub)

        features.append(each_flow_features)
        fids.append(fid)

    # if verbose:
    #     show_len = 10  # only show the first 20 difference
    #     samp_lens = np.asarray([len(samp_features) for samp_features in features])[:show_len]

    return features, fids


def _get_SAMP_NUM(flows, sampling_rate=1):
    """Extract sampling the number of packets from subwindows obtained by splitting each flow.

     The length in time of the sub window is what we’re calling sampling rate.
        features obtained on sampling_rate = 0.1 means that:
         1) split each flow into small windows, each window has 0.1 duration
            (the length in time of each small window)
         2) obtain the number of packets from each window (0.1s).
         3) all values obtained from each window make up of the features (SAMP_NUM).

    Parameters
    ----------
        flows: list

        sampling_rate: float
           the duration of subwindow (interval)
    Returns
    -------
    features: a list
        sizes
    fids: a list
        each value is five-tuple
    """

    features, fids = _get_SAMP(flows, sampling_feature='SAMP_NUM', sampling_rate=sampling_rate)

    return features, fids


def _get_SAMP_SIZE(flows, sampling_rate=1):
    """Extract sampling total sizes of packets from each subwindow after splitting each flow.

     The length in time of the subwindow is what we’re calling sampling rate.
        features obtained on sampling_rate = 0.1 means that:
         1) split each flow into small windows, each window has 0.1 duration
            (the length in time of each small window)
         2) obtain the total size of packets in each window (0.1s).
         3) all the values obtained from each window make up of the features (SAMP_SIZE).

    Parameters
    ----------
        flows:list

        sampling_rate: float
           the duration of subwindow (interval)

    Returns
    -------
    features: a list
        sizes
    fids: a list
        each value is five-tuple
    """
    features, fids = _get_SAMP(flows, sampling_feature='SAMP_SIZE', sampling_rate=sampling_rate)

    return features, fids


def _get_split_interval(flow_durations, q_interval=0.9):
    if len(flow_durations) == 0:
        raise ValueError('no flows: no TCP/UDP flow with at least flow_pkts_thres packets '
                         '(default 2) in the pcap(s)')
    interval = np.quantile(flow_durations, q=q_interval)

    return interval


def _get_flow_durations(flows):
    return [_get_flow_duration(pkts) for fid, pkts in flows]


def _get_pkts_dim(flows, q_interval):
    """Number of packets per flow that features are fixed to: the q_interval quantile of the
    number of packets per flow."""
    num_pkts = [len(pkts) for fid, pkts in flows]
    if not num_pkts:
        raise ValueError('no flows: no TCP/UDP flow with at least flow_pkts_thres packets '
                         '(default 2) in the pcap(s)')
    return int(np.floor(np.quantile(num_pkts, q_interval)))


def _get_sampling_rate(flows):
    """Window length (seconds) for SAMP_* features: the 0.3 quantile of flow durations."""
    # To obtain different samp_features, you should change q_interval ((0, 1))
    return _get_split_interval(_get_flow_durations(flows), q_interval=0.3)


def get_shared_interval(pcaps, q_interval=0.9):
    """Split interval shared by several PCAPs: the q_interval quantile of the durations of all
    their (not yet split) flows. Call PCAP.extract_flows() on each PCAP first.

    Using one interval for all PCAPs makes their features comparable: every flow is then split
    by the same rule, whichever pcap it comes from.
    """
    all_flows = list(itertools.chain.from_iterable(pcap.raw_flows for pcap in pcaps))
    return _get_split_interval(_get_flow_durations(all_flows), q_interval=q_interval)


def get_shared_pkts_dim(pcaps, q_interval=0.9):
    """Feature length shared by several PCAPs: the q_interval quantile of the number of packets
    per (sub)flow over all of them. Call PCAP.split_flows() (or pcap2flows()) on each PCAP first.
    """
    all_flows = list(itertools.chain.from_iterable(pcap.flows for pcap in pcaps))
    return _get_pkts_dim(all_flows, q_interval)


def _get_FFT_data(features, fft_bin='', fft_part='real'):
    """Do fft transform of features

    Parameters
    ----------
    features: features

    fft_bin: int
        the dimension of transformed features
    fft_part: str
        'real' or 'real+imaginary' transformation

    Returns
    -------
    fft_features: a list
        transformed fft features
    """
    if fft_part == 'real':  # default
        fft_features = [list(np.real(np.fft.fft(v, n=fft_bin))) for v in features]

    elif fft_part == 'real+imaginary':
        msg = f'{fft_part}'
        raise NotImplementedError(msg)

    else:
        msg = f'fft_part: {fft_part} is not correct, please modify it and retry!'
        raise ValueError(msg)

    return fft_features


def _dns_text(value):
    """Turn a DNS name or record value from scapy into a string (or None).

    scapy gives names as bytes, A/AAAA records as str, and TXT records as a list of bytes.
    Bytes that are not valid UTF-8 (malformed or malicious packets) are replaced, not fatal.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (list, tuple)):
        return ' '.join(_dns_text(v) for v in value)
    return str(value)


class PCAP:

    def __init__(self, pcap_file, *, flow_ptks_thres=2, verbose=0, random_state=42):
        """PCAP includes all processing functions of pcaps, such as pcap2flows, flow2features,
        and label_flows .

        Parameters
        ----------
        pcap_file: str
            a pcap needed to processed.

        flow_ptks_thres: int (default is 2)
            the minimum number of packets of each flow is to control which flow should be kept
            and which one should be discarded. The default value is 2, i.e., all the flows which
            have less than 2 packets
            are discarded. It must be >= 2.

        verbose: int (default is 0)
            a print level is to control what information should be printed according to the given
            value.
            The higher the value is, the more info is printed.

        random_state: int
            a value is to make your experiments more reproducible.

        Returns
        -------
            a PCAP instance

        """
        self.pcap_file = pcap_file
        self.flow_ptks_thres = flow_ptks_thres
        self.verbose = verbose
        self.random_state = random_state

        self.labels = None
        self.df = None

    @timing
    def _pcap2flows(self, interval=0, q_interval=0.1, *, tcp_timeout=600, udp_timeout=600):
        """Extract flows from the given pcap and split each flow to subflow by "interval" or
        "q_interval".
                   It prefers to choose "interval" as the split measure if interval > 0; otherwise,
                   use q_interval to find an interval.
                    q_interval must be in [0, 1]

        Parameters
        ----------
        interval: float (default is 0.)
            an time interval is used to split a flow.

        q_interval: float (default is 0.9)
           a quntile (must be in [0, 1]) is to obtain "interval" from all flow durations.

        tcp_timeout: int (default is 600s)
            a value is to split flow by tcp_timeout.

        udp_timeout: int (default is 600s)
            a value is to split flow by udp_timeout.

        Returns
        -------
        all flows: list
            each element in the list represents a flow, and each flow includes 2 values: flow id
            (five-tuple) and packets.
        """

        # always kept: flow2features uses it for the feature dimension, even if "interval" is given
        self.q_interval = q_interval

        # extract all flows firstly and then split flows to subflows
        self.extract_flows(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout)

        if interval <= 0:
            if q_interval <= 0:
                msg = f'q_interval must be in [0, 1]! Current q_interval is {q_interval}.'
                raise ValueError(msg)

            self.flow_durations = _get_flow_durations(self.raw_flows)
            if self.verbose > 3:
                data_info(np.asarray(self.flow_durations, dtype=float).reshape(-1, 1),
                          name='flow_durations')
            interval = _get_split_interval(self.flow_durations, q_interval=q_interval)

        self.split_flows(interval)
        self.raw_flows = None  # not needed anymore; free the packets dropped by the split

    def extract_flows(self, *, tcp_timeout=600, udp_timeout=600):
        """Extract whole flows (not yet split into subflows) into self.raw_flows.

        First step of pcap2flows(). Together with split_flows(), it lets several PCAPs be split
        with one shared interval (see get_shared_interval()).

        Parameters
        ----------
        tcp_timeout: int (default is 600s)
            a value is to split flow by tcp_timeout.

        udp_timeout: int (default is 600s)
            a value is to split flow by udp_timeout.
        """
        self.raw_flows = _pcap2flows(self.pcap_file, self.flow_ptks_thres, tcp_timeout=tcp_timeout,
                                     udp_timeout=udp_timeout, verbose=self.verbose)

    def split_flows(self, interval):
        """Split self.raw_flows (from extract_flows()) into subflows, into self.flows. Second step
        of pcap2flows().

        A flow is split wherever the time between two consecutive packets is longer than
        "interval" (so it is an idle-time threshold, not a fixed window length).

        Parameters
        ----------
        interval: float
            an time interval is used to split a flow.
        """
        self.interval = interval
        self.flows = _flows2subflows(self.raw_flows, interval,
                                     flow_pkts_thres=self.flow_ptks_thres, verbose=self.verbose)

    def pcap2flows(self, interval=0.0, q_interval=0.9, *, tcp_timeout=600, udp_timeout=600):
        """Extract flows from the given pcap and split each flow to subflow by "interval" or
        "q_interval".
           It prefers to choose "interval" as the split measure if interval > 0; otherwise,
           use q_interval to find an interval.
            q_interval must be in [0, 1]

        Parameters
        ----------
        interval: float (default is 0.)
            an time interval is used to split a flow.

        q_interval: float (default is 0.9)
           a quntile (must be in [0, 1]) is to obtain "interval" from all flow durations.

        tcp_timeout: int (default is 600s)
            a value is to split flow by tcp_timeout.

        udp_timeout: int (default is 600s)
            a value is to split flow by udp_timeout.
        Returns
        -------
            self
        """
        _, tot_time = self._pcap2flows(interval, q_interval, tcp_timeout=tcp_timeout,
                                       udp_timeout=udp_timeout)
        self.pcap2flows.__dict__['tot_time'] = tot_time

    @timing
    def _flow2features(self, feat_type='IAT', *, fft=False, header=False, pkts_dim=None,
                       sampling_rate=None):
        """Extract features from each flow according to feat_type, fft and header.

        Parameters
        ----------
        feat_type: str (default is 'IAT')
            which features do we want to extract from flows

        fft: boolean (default is False)
            if we need fft-features

        header: boolean (default is False)
            if we need header+features

        Returns
        -------
            self
        """
        self.feat_type = feat_type

        if pkts_dim is None:
            q_interval = getattr(self, 'q_interval', None)
            if q_interval is None:
                raise ValueError('pkts_dim is required when the flows were not split by '
                                 'pcap2flows()')
            # use the same q_interval to get the dimension
            pkts_dim = _get_pkts_dim(self.flows, q_interval)
        self.pkts_dim = pkts_dim
        dim = pkts_dim

        is_samp = feat_type in ['SAMP_NUM', 'FFT-SAMP_NUM', 'SAMP_SIZE', 'FFT-SAMP_SIZE']
        if is_samp and sampling_rate is None:
            sampling_rate = _get_sampling_rate(self.flows)
        self.sampling_rate = sampling_rate

        if feat_type in ['IAT', 'FFT-IAT']:
            self.dim = dim - 1
            self.features, self.fids = _get_IAT(self.flows)
        elif feat_type in ['SIZE', 'FFT-SIZE']:
            self.dim = dim
            self.features, self.fids = _get_SIZE(self.flows)
        elif feat_type in ['IAT_SIZE', 'FFT-IAT_SIZE']:
            self.dim = 2 * dim - 1
            self.features, self.fids = _get_IAT_SIZE(self.flows)
        elif feat_type in ['STATS']:
            self.dim = 12
            self.features, self.fids = _get_STATS(self.flows)
        elif feat_type in ['SAMP_NUM', 'FFT-SAMP_NUM']:
            self.dim = dim - 1
            self.features, self.fids = _get_SAMP_NUM(self.flows, sampling_rate)
        elif feat_type in ['SAMP_SIZE', 'FFT-SAMP_SIZE']:
            # here the dim of "SAMP_SIZE" is dim -1, which equals to the dimension of 'SAMP_NUM'
            self.dim = dim - 1
            self.features, self.fids = _get_SAMP_SIZE(self.flows, sampling_rate)
        else:
            msg = f'feat_type ({feat_type}) is not correct! '
            raise ValueError(msg)

        if fft:
            self.features = _get_FFT_data(self.features, fft_bin=self.dim)
        else:
            # fix each flow to the same feature dimension (cut off the flow or append 0 to it)
            self.features = [v[:self.dim] if len(v) > self.dim else v + [0] * (self.dim - len(v))
                             for v in self.features]

        if header:
            _headers = _get_header_features(self.flows)
            h_dim = 8 + self.dim  # 8 TCP flags
            if fft:
                fft_headers = _get_FFT_data(_headers, fft_bin=h_dim)
                self.features = [h + f for h, f in zip(fft_headers, self.features)]
            else:
                # fix header dimension firstly
                headers = [h[:h_dim] if len(h) > h_dim else h + [0] * (h_dim - len(h))
                           for h in _headers]
                self.features = [h + f for h, f in zip(headers, self.features)]

        # change list to numpy array
        self.features = np.asarray(self.features, dtype=float)
        if not self.flows:
            # no flows: still a 2-D array (0 rows), so it can be concatenated with others
            width = self.dim + (8 + self.dim if header else 0)
            self.features = self.features.reshape(0, width)
        if self.verbose > 5:
            print(np.all(self.features >= 0))

    def flow2features(self, feat_type='IAT', *, fft=False, header=False, pkts_dim=None,
                      sampling_rate=None):
        """Extract features from each flow according to feat_type, fft and header.

        Parameters
        ----------
        feat_type: str (default is 'IAT')
            which features do we want to extract from flows

        fft: boolean (default is False)
            if we need fft-features

        header: boolean (default is False)
            if we need header+features

        pkts_dim: int (default is None)
            the number of packets per flow that features are fixed to (e.g. IAT features have
            pkts_dim - 1 values). By default, the q_interval quantile of the number of packets per
            flow in this pcap. Pass a fixed value (e.g. from get_shared_pkts_dim(), or the value
            used for training) so that features from different pcaps have the same length.

        sampling_rate: float (default is None)
            window length (seconds) for SAMP_* features. By default, the 0.3 quantile of the flow
            durations in this pcap.

        Returns
        -------
            self
        """
        _, tot_time = self._flow2features(feat_type, fft=fft, header=header, pkts_dim=pkts_dim,
                                          sampling_rate=sampling_rate)
        self.flow2features.__dict__['tot_time'] = tot_time

    @timing
    def _label_flows(self, label_file=None, label=0):
        """label each flow by label_file (only for CICIDS_2017 label_file) or label.
        If you want to parse other label file, you have to override "label_flows()" with your own
        one.
        (normal=0,and abnormal = 1)

        Parameters
        ----------
        label_file: str (path) or file
            a file that includes flow labels

        label: int
            if label_file is None, then use "label" to label each flow

        Returns
        -------
        self
        """

        if label_file:
            NORMAL_LABELS = [v.upper() for v in ['benign'.upper(), 'normal'.upper()]]

            # load CSV with pandas
            csv = pd.read_csv(label_file)

            true_labels = {}
            cnt_anomaly = 0
            cnt_nomral = 0

            for i, r in enumerate(csv.index):
                if (self.verbose > 3) and (i % 10000 == 0):
                    print(f"Label CSV {i}th row")
                row = csv.loc[r]
                fid = (str(row[" Source IP"]), str(row[" Destination IP"]),
                       int(row[" Source Port"]), int(row[" Destination Port"]),
                       int(row[" Protocol"]))
                # ensure all 5-tuple flows have same label
                label_i = row[" Label"].upper()
                if label_i in NORMAL_LABELS:
                    label_i = 0
                    cnt_nomral += 1
                else:
                    label_i = 1
                    cnt_anomaly += 1

                # it will overwrite the previous label for the same fid.
                true_labels[fid] = label_i

            # label flows with ture_labels
            new_labels = []
            not_existed_fids = []
            new_flows = []
            for i, (fid, pkts) in enumerate(self.flows):
                if fid in true_labels.keys():
                    new_labels.append(true_labels[fid])
                    new_flows.append((fid, pkts))
                else:
                    # the fid does not exist in labels.csv
                    not_existed_fids.append(fid)
            if self.verbose > 3:
                print(
                    f'Number of labelled flows: {len(new_labels)}; number of not existed flows: '
                    f'{len(not_existed_fids)}')

            if self.flows and not new_flows:
                warnings.warn(f'none of the {len(self.flows)} flows of {self.pcap_file} matched a '
                              f'row of the label file {label_file}: all its flows are dropped; '
                              'check that the label file belongs to this pcap')

            self.labels = new_labels
            self.flows = new_flows
        else:
            self.labels = [label] * len(self.flows)

        self.labels = np.asarray(self.labels, dtype=int)

    def label_flows(self, label_file=None, label=0):
        """label each flow by label_file (only for CICIDS_2017 label_file) or label.
        If you want to parse other label file, you have to override "label_flows()" with your own
        one.
        (normal=0,and abnormal = 1)

        Parameters
        ----------
        label_file: str (path) or file
            a file that includes flow labels

        label: int
            if label_file is None, then use "label" to label each flow

        Returns
        -------
        self
        """
        _, tot_time = self._label_flows(label_file, label)
        self.label_flows.__dict__['tot_time'] = tot_time

    _pcap_columns = (
        'time',
        'datetime',
        'length',
        'mac_dst',
        'mac_src',
        'ip_dst',
        'ip_src',
        'protocol',
        'port_dst',
        'port_src',
        'is_dns',
        'dns_query',
        'dns_resp',
    )

    def _iter_pcap_dict(self):
        """Stream extracted dict mappings from PCAP file.

        Requires:
          self.pcap_file: string filepath of PCAP file

        Returns:
          Iterator of dicts with one dict per packet in pcap file.

            The dicts have the following key/value pairs:

              "time"      : time the packet was received, in seconds since the epoch (float)
              "datetime"  : time the packet was received, in nanoseconds since the epoch (int);
                            turned into a UTC timestamp by _pcap2pandas
              "length"    : original (wire) length of packet in bytes
              "mac_src"   : source MAC address
              "mac_dst"   : destination MAC address
              "ip_src"    : source IP address (IPv4 or IPv6)
              "ip_dst"    : destination IP address (IPv4 or IPv6)
              "protocol"  : protocol name, e.g. 'TCP', 'UDP', 'ICMP', 'ICMPv6', 'ARP', 'IGMP',
                            or None if it cannot be identified (see _get_protocol)
              "port_src"  : source port
              "port_dst"  : destination port
              "is_dns"    : True if packet is DNS packet, else False
              "dns_query" : DNS query name (str)
              "dns_resp"  : DNS answer (str) of the first answer record

        """
        with PcapReader(self.pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                pkt_dict = dict.fromkeys(self._pcap_columns)

                pkt_dict.update(
                    time=float(pkt.time),
                    # exact: pkt.time is a decimal number of seconds, as stored in the pcap
                    datetime=int(pkt.time * 1_000_000_000),
                    length=_get_pkt_len(pkt),
                    is_dns=False,
                )

                if Ether in pkt:
                    pkt_dict['mac_dst'] = pkt[Ether].dst
                    pkt_dict['mac_src'] = pkt[Ether].src

                ip = _get_ip_layer(pkt)
                if ip is not None:
                    pkt_dict['ip_dst'] = ip.dst
                    pkt_dict['ip_src'] = ip.src

                if TCP in pkt:
                    pkt_dict['port_dst'] = pkt[TCP].dport
                    pkt_dict['port_src'] = pkt[TCP].sport
                elif UDP in pkt:
                    pkt_dict['port_dst'] = pkt[UDP].dport
                    pkt_dict['port_src'] = pkt[UDP].sport

                pkt_dict['protocol'] = _get_protocol(pkt)

                if (dnsqr := pkt.getlayer(DNSQR)) is not None:
                    pkt_dict.update(
                        is_dns=True,
                        dns_query=_dns_text(dnsqr.qname),
                    )

                if (dnsrr := pkt.getlayer(DNSRR)) is not None:
                    pkt_dict.update(
                        is_dns=True,
                        dns_query=_dns_text(dnsrr.rrname),
                        dns_resp=_dns_text(dnsrr.rdata),
                    )

                yield pkt_dict

    @timing
    def _pcap2pandas(self):
        """Parse PCAP file into pandas DataFrame.

        Requires:
            self.pcap_file: string filepath of PCAP file

        Returns:
          DataFrame with one packet per row.
            column names are those of _pcap_columns plus
            'ip_dst_int', 'ip_src_int', 'mac_dst_int', 'mac_dst_int'

        """
        self.df = pd.DataFrame(self._iter_pcap_dict(), columns=self._pcap_columns)

        # float, also for an empty pcap
        self.df['time'] = self.df['time'].astype(float)

        # full precision, in UTC: the same on every machine, whatever its local time zone
        self.df['datetime'] = pd.to_datetime(self.df['datetime'].astype('int64'), unit='ns',
                                             utc=True)

        self.df['ip_dst_int'] = self.df['ip_dst'].apply(
            lambda x: None if x is None else int(ipaddress.ip_address(x)))

        self.df['ip_src_int'] = self.df['ip_src'].apply(
            lambda x: None if x is None else int(ipaddress.ip_address(x)))

        self.df['mac_dst_int'] = self.df['mac_dst'].apply(
            lambda x: None if x is None else int(netaddr.EUI(x)))

        self.df['mac_src_int'] = self.df['mac_src'].apply(
            lambda x: None if x is None else int(netaddr.EUI(x)))

        # seconds since the earliest packet (packets are not always stored in time order);
        # computed from the exact datetimes
        time_since_start = self.df['datetime'] - self.df['datetime'].min()
        self.df['time_normed'] = time_since_start.dt.total_seconds()

        self.df.sort_index(axis=1, inplace=True)

    def pcap2pandas(self):
        """Parse PCAP file into pandas DataFrame.

        Requires:
            self.pcap_file: string filepath of pcap file

        Sets:
            self.df

        """
        (_, self.pcap2pandas.__dict__['tot_time']) = self._pcap2pandas()
