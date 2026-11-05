"""A demo for parsing pcap and labelling the extracted flows.

"""
import pathlib

from netml.pparser.parser import PCAP
from netml.utils.tool import dump_data


RANDOM_STATE = 42

EXAMPLES_PATH = pathlib.Path(__file__).parent.parent

# outputs go here, never into examples/data/ (the inputs)
OUT_DATA_PATH = EXAMPLES_PATH / 'out' / 'data'

PCAP_FILE = EXAMPLES_PATH / 'data' / 'demo.pcap'

LABEL_FILE = EXAMPLES_PATH / 'data' / 'demo.csv'


def main(pcap_file=PCAP_FILE, label_file=LABEL_FILE, out_dir=OUT_DATA_PATH):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pp = PCAP(
        str(pcap_file),    # scapy doesn't appear to support Path
        flow_ptks_thres=2,
        verbose=10,
        random_state=RANDOM_STATE,
    )

    pp.pcap2pandas()

    pp.df.to_csv(out_dir / 'demo_pandas.csv')
    # extract flows from pcap
    pp.pcap2flows(q_interval=0.9)

    # label each flow with a label
    pp.label_flows(label_file=str(label_file))

    # extract features from each flow given feat_type
    # feat_type in ['IAT', 'SIZE', 'STATS', 'SAMP_NUM', 'SAMP_SIZE']
    feat_type = 'IAT'
    print(f'feat_type: {feat_type}')
    pp.flow2features(feat_type, fft=False, header=False)

    # dump data to disk
    X, y = pp.features, pp.labels
    # (read by examples/ndm/ndm_demo.py)
    out_file = out_dir / f'demo_{feat_type}.dat'
    dump_data((X, y), out_file=out_file)
    print(f'features written to {out_file}')

    print(pp.features.shape, pp.pcap2flows.tot_time, pp.flow2features.tot_time)


if __name__ == '__main__':
    main()
