"""Capture packets by Scapy

    pip3 install netifaces
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import os
from datetime import datetime
from scapy.layers.inet import IP, TCP, UDP
from scapy.sendrecv import sniff
from scapy.utils import wrpcap


FLOW_TIMEOUT = 600  # seconds: a flow ends when it has been idle for longer than this
MAX_FLOW_DURATION = 60 * 60  # seconds: a flow that has lasted longer than this is ended


def _get_fid(pkt):
    """Extract fid (five-tuple) from a packet: only focus on IPv4
    Parameters
    ----------

    Returns
    -------
        fid: five-tuple
    """

    if IP in pkt and TCP in pkt:
        fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
    elif IP in pkt and UDP in pkt:
        fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
    else:
        fid = ('', '', -1, -1, -1)

    return fid


def get_device_interfaces(verbose=10):
    """Get the device interfaces (netcard)

    Parameters
    ----------
    verbose

    Returns
    -------

    """
    # not work on mac
    # from scapy.arch.windows import IFACES
    # Print list of available network interfaces
    # print(IFACES.show(resolve_mac))
    # print(IFACES.show())

    import netifaces
    from pprint import pprint

    ifaces = netifaces.interfaces()
    pprint(ifaces)
    if verbose >= 5:
        for intf in ifaces:
            print(f'\n***{intf}')
            pprint(netifaces.ifaddresses(intf))

    return ifaces


def update_flows(pkt_time, fid, flow_buffer, flows, timeout=FLOW_TIMEOUT,
                 max_duration=MAX_FLOW_DURATION):
    """Track flows with one more packet.

    Parameters
    ----------
    pkt_time: float
        arrival time of the packet

    fid: five-tuple
        flow id of the packet

    flow_buffer: dict
        flows in progress: fid -> (number of packets, start time, time of the last packet)

    flows: dict
        ended flows, as in flow_buffer (a newer flow with the same fid replaces an older one)

    timeout: float
        a flow ends when it has been idle for longer than this (seconds)

    max_duration: float
        flows that have lasted longer than this are ended (seconds)
    """
    if fid not in flow_buffer:
        flow_buffer[fid] = (1, pkt_time, pkt_time)
        return

    (pkt_cnt, start_time, pre_pkt_time) = flow_buffer[fid]
    if pkt_time - pre_pkt_time > timeout:
        # idle for longer than the timeout: the flow has ended, and this packet starts a new one
        flows[fid] = flow_buffer[fid]
        flow_buffer[fid] = (1, pkt_time, pkt_time)
    else:
        flow_buffer[fid] = (pkt_cnt + 1, start_time, pkt_time)

    # end the flows that have lasted too long, to keep flow_buffer small
    # (loop over a copy of the items: entries are removed from flow_buffer in the loop)
    for (k, (_pkt_cnt, k_start_time, k_pre_pkt_time)) in list(flow_buffer.items()):
        if k_pre_pkt_time - k_start_time > max_duration:
            flows[k] = flow_buffer.pop(k)


def capture(iface='en0', out_file='capture.pcap'):
    """
    https://0xbharath.github.io/art-of-packet-crafting-with-scapy/scapy/sniffing/index.html
    Parameters
    ----------
    iface: str
        netcard interface

    out_file: file name
        store all captured packets to out_file

    Returns
    -------
    out_file:

    """
    start_capture_time = datetime.now()

    print(f'\ncapture starts at {start_capture_time}...')

    if os.path.exists(out_file):
        os.remove(out_file)

    pkt_buffer = []  # buffer size of packets
    flow_buffer = {}  # flows in progress (see update_flows)
    src_ip = '192.168.143.76'

    flows = {}  # ended flows
    for pkt in sniff(iface=iface, count=2000):
        pkt_buffer.append(pkt)

        pkt_time = pkt.time
        if len([fid for fid in flows.keys() if src_ip == fid[0]]) > 1000:  # number of flows
            wrpcap(out_file, pkt_buffer, append=True)  # appends packet to output file
            pkt_buffer = []
            break
        fid = _get_fid(pkt)
        update_flows(pkt_time, fid, flow_buffer, flows)

        if len(pkt_buffer) > 10000:  # buffer size of packets
            wrpcap(out_file, pkt_buffer, append=True)  # appends packet to output file
            pkt_buffer = []

    wrpcap(out_file, pkt_buffer, append=True)  # appends packet to output file
    print(f'out_file: {os.path.abspath(out_file)}')

    end_capture_time = datetime.now()
    total_time = (end_capture_time - start_capture_time).total_seconds()
    print(f'capture finished at {end_capture_time}, and the total time is {total_time} s')


if __name__ == '__main__':
    get_device_interfaces(verbose=0)
    capture(iface='en0')
