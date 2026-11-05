"""Command-line interface for netml library."""
import argparse
import contextlib
import datetime
import functools
import importlib
import inspect
import itertools
import pathlib
import pkgutil
import sys
import textwrap

import argcmdr
import argparse_formatter
import numpy as np
import sklearn.model_selection
import terminaltables
import yaml
from plumbum import colors

import netml.ndm
import netml.ndm.model
import netml.pparser.parser
from netml.utils.tool import dump_data, load_data, FeatureData, ManualDependencyError


# TODO: (sub)-command to quickly/easily get command-line completion?
#
# (likely shouldn't do anything to system, but could at least call out to
# register-python-argcomplete?)


LABEL_NORMAL = 0
LABEL_ABNORMAL = 1


class NoFlowsError(ValueError):
    """None of the given pcaps has a flow to extract features from."""


# interface for setup.py

def execute():
    """launch the netml cli"""
    argcmdr.main(Main)


# utilities

# NOTE: FileType('rb') with "-" doesn't respect binary flag and so fails --
# NOTE: https://bugs.python.org/issue14156

def binary_input_file_type(value, allow_stdin=False):
    """Open given file path for binary reading.

    Optionally return STDIN given the value "-".

    """
    if allow_stdin and value == '-':
        return sys.stdin.buffer

    return open(value, 'rb')


binary_input_file_type_stdin = functools.partial(binary_input_file_type, allow_stdin=True)


def _pcap_name(pcap_file):
    """Name of a pcap (a path, or an open file such as standard input) for messages."""
    return str(getattr(pcap_file, 'name', pcap_file))


PARAM_FILE_SUFFIXES = ('.yaml', '.yml', '.json')

# expected fraction of anomalies (see Learn: --contamination)
DEFAULT_CONTAMINATION = 0.1

# model parameter that holds the expected fraction of anomalies, by order of preference
CONTAMINATION_PARAMS = ('contamination', 'nu')


def param_mapping(value):
    """Parse --param: YAML/JSON text, or the path to a YAML/JSON file, giving a mapping (dict).

    JSON is valid YAML, so both are read by the YAML parser.

    """
    path = pathlib.Path(value)
    try:
        is_file = path.is_file()
    except OSError:  # e.g. text too long to be a file name
        is_file = False

    if is_file:
        text = path.read_text()
    elif path.suffix.lower() in PARAM_FILE_SUFFIXES:
        raise argparse.ArgumentTypeError(f'file not found: {value}')
    else:
        text = value

    try:
        params = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f'invalid YAML/JSON: {exc}')

    if params is None:
        return {}

    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError(
            f"must be a mapping of parameter names to values, e.g. '{{nu: 0.1}}' or "
            f"'{{ocsvm: {{nu: 0.1}}}}' (or the path to a file with one), not: {value}"
        )

    return params


# base command class

class Command(argcmdr.Command):

    # feature type extracted from flows
    # TODO: support specification of remaining feature types
    feat_type = 'IAT'

    def extract_pcaps(self, pcap_file_labels, params=None):
        """Parse PCAP files and extract their flows and features, with shared settings.

        All pcaps are split into subflows with one interval, and their features are fixed to one
        length, so features from different pcaps are comparable (and can be concatenated). These
        settings are computed from the flows of all the given pcaps, unless they are given in
        "params" (e.g. the settings a model was trained with).

        Labels, for model-testing, are applied if specified.

        Parameters
        ----------
        pcap_file_labels: iterable of (pcap_file, label_kwargs)
            label_kwargs: dict with optional "label" or "label_file" (see PCAP.label_flows)

        params: dict (default is None)
            optional "feat_type", "interval" and "pkts_dim"; missing or None values are computed

        Returns
        -------
        (pcaps, params): the PCAP objects, and the settings used (dict)

        Pcaps left without any flow are skipped, with a warning; NoFlowsError is raised if no pcap
        has a flow.

        """
        params = dict(params or {})
        q_interval = params.get('q_interval', self.args.q_interval)
        feat_type = params.get('feat_type', self.feat_type)
        min_pkts = self.args.flow_pkts_threshold

        # 1. extract whole flows from every pcap
        all_names = []
        pcaps = []
        pcap_label_kwargs = []
        for (pcap_file, label_kwargs) in pcap_file_labels:
            pcap = netml.pparser.parser.PCAP(
                pcap_file,
                flow_ptks_thres=min_pkts,
                verbose=self.args.verbosity,
                random_state=self.args.random_state,
            )
            pcap.extract_flows()
            all_names.append(_pcap_name(pcap_file))

            if not pcap.raw_flows:
                print(f'warning: {_pcap_name(pcap_file)} has no TCP/UDP flow with at least '
                      f'{min_pkts} packets: skipped', file=sys.stderr)
                continue

            pcaps.append(pcap)
            pcap_label_kwargs.append(label_kwargs)

        if not pcaps:
            raise NoFlowsError(f'no flows to extract features from: no TCP/UDP flow with at least '
                               f'{min_pkts} packets in {", ".join(all_names)}')

        # 2. split them into subflows with one shared interval
        interval = params.get('interval')
        if interval is None:
            interval = netml.pparser.parser.get_shared_interval(pcaps, q_interval)

        for (pcap, label_kwargs) in zip(pcaps, pcap_label_kwargs):
            pcap.split_flows(interval)
            pcap.raw_flows = None  # not needed anymore

            # label each flow
            label = label_kwargs.get('label')
            label_file = label_kwargs.get('label_file')
            if label is not None or label_file is not None:
                pcap.label_flows(label_file=label_file, label=label)

        # pcaps left without flows: subflows too short, or none matched the label file
        empty_names = [_pcap_name(pcap.pcap_file) for pcap in pcaps if not pcap.flows]
        for name in empty_names:
            print(f'warning: no flow of {name} is left after splitting and labeling '
                  '(see any warning above): skipped', file=sys.stderr)
        pcaps = [pcap for pcap in pcaps if pcap.flows]

        if not pcaps:
            raise NoFlowsError(f'no flows to extract features from: none is left after splitting '
                               f'and labeling, in {", ".join(all_names)}')

        # 3. extract features with one shared feature length
        pkts_dim = params.get('pkts_dim')
        if pkts_dim is None:
            pkts_dim = netml.pparser.parser.get_shared_pkts_dim(pcaps, q_interval)

        for pcap in pcaps:
            pcap.flow2features(feat_type, fft=False, header=False, pkts_dim=pkts_dim)

        params_used = {
            'feat_type': feat_type,
            'q_interval': q_interval,
            'interval': float(interval),
            'pkts_dim': int(pkts_dim),
        }
        return (pcaps, params_used)

    @staticmethod
    def get_model_params(model):
        """Get the extraction settings a model was trained with (see Learn.perform_train).

        Models saved by older versions of netml have no settings: then the feature length is
        taken from the model itself (if it records it), and the split interval from the pcap.

        """
        params = getattr(model, 'extraction_params', None)
        if params is not None:
            return params

        print("warning: the model file has no saved feature-extraction settings "
              "(it was made by an older version of netml); "
              "the flow split interval is computed from this pcap instead",
              file=sys.stderr)

        n_features = getattr(model, 'n_features_in_', None)
        if n_features is None:
            return None

        # the CLI extracts IAT features, which have pkts_dim - 1 values
        return {'pkts_dim': n_features + 1}

    def vprint(self, level, *args, **kwargs):
        """Print IFF verbosity level meets or exceeds `level`."""
        if self.args.verbosity >= level:
            print(*args, **kwargs)

    def vtable(self, level, rows, title=None):
        """Print tabular data IFF verbosity level meets or exceeds `level`."""
        self.vprint(level, terminaltables.AsciiTable(rows, title=title).table)


# commands

class Main(argcmdr.RootCommand):
    """detect novel or anomalous network activity from packet captures"""

    def __init__(self, parser):
        # FIXME: internals are way too verbose
        parser.add_argument(
            '-v', '--verbosity',
            type=int,
            default=1,
            metavar='INT',
            help="verbosity level (default: 1)",
        )
        parser.add_argument(
            '--flow-pkts-threshold',
            type=int,
            metavar='INT',
            default=2,
            help="minimum packets per extracted flow (default: 2)",
        )
        parser.add_argument(
            '--q-interval',
            type=float,
            metavar='FLOAT',
            default=0.9,
            help="quantile between [0, 1] by which flow duration interval is determined "
                 "(default: 0.9)",
        )
        parser.add_argument(
            '--random-state',
            metavar='INT',
            type=int,
            default=42,
            help="random state used in extraction and modeling (default: 42)",
        )


@Main.register
class Classify(Command):
    """classify network activity"""

    def __init__(self, parser):
        parser.add_argument(
            '--report-all',
            action='store_true',
            default=None,
            dest='report_all',
            help="report non-anomalous packet flows "
                 "(by default only reported at verbosity greater than 1)",
        )
        parser.add_argument(
            '--no-report-all',
            action='store_false',
            default=None,
            dest='report_all',
            help="do NOT report non-anomalous packet flows "
                 "(by default only reported at verbosity greater than 1)",
        )

        # TODO: might be cool to be able to "follow" input, classifying it in chunks
        parser.add_argument(
            '-p', '--pcap',
            metavar='FILE',
            # NOTE: FileType('rb') with "-" doesn't respect binary flag and so fails --
            # NOTE: https://bugs.python.org/issue14156
            type=binary_input_file_type_stdin,
            help='path to packet capture (pcap) file '
                 '(the value "-" or the omission of this option indicates standard input)',
        )

        parser.add_argument(
            '-m', '--model',
            metavar='FILE',
            required=True,
            type=binary_input_file_type,
            help='path to trained novelty-detection model',
        )

    def __call__(self, args, parser):
        pcap_file = args.pcap

        if pcap_file is None:
            if sys.stdin.isatty():
                parser.error("the following arguments are required "
                             "when standard input is not specified: -p/--pcap")

            pcap_file = sys.stdin.buffer

        # TODO: catch model file issues such as EOFError
        (model, train_history) = load_data(args.model)

        # extract features with the same settings the model was trained with
        try:
            ((pcap,), _params) = self.extract_pcaps([(pcap_file, {})],
                                                    params=self.get_model_params(model))
        except NoFlowsError as exc:
            # not an error for classify: there is simply nothing to report
            print(f'warning: {exc}; nothing to classify', file=sys.stderr)
            return

        # True where the model flags an anomaly (whichever convention the model's predict() has)
        try:
            is_anomaly = netml.ndm.model.predict_anomalies(model, pcap.features)
        except ValueError as exc:
            parser.error(str(exc))

        for ((flow_key, flow_packets), anomaly) in zip(pcap.flows, is_anomaly):
            if anomaly:
                class_tag = 'ANOMALY' | colors.red | colors.bold
            else:
                if args.report_all is None:
                    if args.verbosity <= 1:
                        continue
                elif not args.report_all:
                    continue

                class_tag = 'NORMAL'

            if flow_key[4] == 6:
                packet_type = 'TCP'
            elif flow_key[4] == 17:
                packet_type = 'UDP'
            else:
                packet_type = '[protocol-other]'

            (packet_datetime0, packet_datetime1) = packet_datetimes = [
                datetime.datetime.fromtimestamp(float(packet.time))
                for packet in (flow_packets[0], flow_packets[-1])
            ]
            packet_date = packet_datetime0.date()
            (packet_time0, packet_time1) = (
                packet_datetime.time()
                for packet_datetime in packet_datetimes
            )

            print(
                f'[{packet_date}] [{packet_time0} – {packet_time1}]',
                f'{flow_key[0]}:{flow_key[2]} → {flow_key[1]}:{flow_key[3]} [{packet_type}]',
                class_tag,
            )


@Main.register
class Learn(Command):
    """train & test anomaly-detection models"""

    action_extract = 'extract'
    action_train = 'train'
    actions = (action_extract, action_train)

    train_algorithms = ('ocsvm', 'kde', 'if', 'ae', 'gmm', 'pca')

    formatter_class = argparse_formatter.ParagraphFormatter

    def __init__(self, parser):
        # input/output paths

        pcap_labeling = parser.add_argument_group(
            "pcap extraction & labeling",

            "Specify packet capture (pcap) file(s) from which to extract features.\n\n"

            "To test the model during the training phase, "
            "optionally specify labels (csv) file(s) with -l/--label, "
            "by which to label pcap files specified with -p/--pcap. "
            "Alternatively, specify NORMAL & ABNORMAL pcaps with "
            "--pcap-normal and --pcap-abnormal.\n\n"

            f'Note: at least one pcap file is required to "{self.action_extract}".'
        )
        pcap_labeling.add_argument(
            '-p', '--pcap',
            action='append',
            metavar='FILE',
            # TODO: support optionless input like classify
            type=argparse.FileType('rb'),
            help='path(s) to packet capture (pcap) file(s)',
        )
        pcap_labeling.add_argument(
            '-l', '--label',
            action='append',
            metavar='FILE',
            type=argparse.FileType('r'),
            help='path(s) to labels (csv) file(s) to pair with pcap files specified by --pcap '
                 f'(optional for "{self.action_extract}" such that "{self.action_train}" '
                 'may subsequently test model)',
            # TODO: help=f'path to labels (csv) file (required to "{self.action_test}")',
        )
        pcap_labeling.add_argument(
            '--pcap-normal',
            action='append',
            metavar='FILE',
            type=argparse.FileType('rb'),
            help='path(s) to packet capture (pcap) file(s) '
                 'that will be labeled "normal" during testing',
        )
        pcap_labeling.add_argument(
            '--pcap-abnormal',
            action='append',
            metavar='FILE',
            type=argparse.FileType('rb'),
            help='path(s) to packet capture (pcap) file(s) '
                 'that will be labeled "abnormal" during testing',
        )

        feature_modeling = parser.add_argument_group(
            "features & model",
            "Specify paths at which to store extracted features and the trained model."
        )
        # TODO: support optionless output (like classify)
        # NOTE: -f and -o are paths, not argparse.FileType: FileType opens (and so empties) the
        # NOTE: file while parsing arguments, which destroyed the input of "train" in isolation,
        # NOTE: and the previous output of any run that then failed. They are opened only once
        # NOTE: their data is ready.
        feature_modeling.add_argument(
            '-f', '--feature',
            metavar='FILE',
            type=pathlib.Path,
            help=f"path at which extracted features are stored "
                 f'(written by "{self.action_extract}", read by "{self.action_train}"; '
                 f'required only to "{self.action_extract}" OR '
                 f'"{self.action_train}" *in isolation*)',
        )
        feature_modeling.add_argument(
            '-o', '--output',
            '-m', '--model',
            metavar='FILE',
            type=pathlib.Path,
            help="path at which to store trained model "
                 f'(required to "{self.action_train}")',
        )

        model_params = parser.add_argument_group(
            "modeling",
            "Configure model training."
        )
        model_params.add_argument(
            '--algorithm',
            choices=self.train_algorithms,
            default=self.train_algorithms[0],
            help=f"model-training algorithm to use (default: {self.train_algorithms[0]})",
        )
        model_params.add_argument(
            '--help-algorithm',
            action='store_true',
            help="describe model-training algorithms and exit",
        )
        model_params.add_argument(
            '--param',
            metavar='YAML',
            type=param_mapping,
            help="algorithmic parameters to override defaults: YAML/JSON text, e.g. "
                 "'{nu: 0.1}' or '{ocsvm: {nu: 0.1}, kde: {bandwidth: 2}}', or the path to a "
                 f"{'/'.join(PARAM_FILE_SUFFIXES)} file with such text",
        )
        model_params.add_argument(
            '--help-param',
            action='store_true',
            help="show all model parameters & defaults and exit",
        )
        model_params.add_argument(
            '--contamination',
            metavar='FRACTION',
            type=float,
            default=None,
            help="expected fraction of anomalies, in (0, 0.5]: classify will flag about this "
                 "fraction of the flows similar to the training data (fewer false alarms with "
                 "a smaller value, more anomalies caught with a larger one); sets "
                 "'contamination' of the if, kde, gmm, pca and ae algorithms and 'nu' of ocsvm "
                 f"(default: {DEFAULT_CONTAMINATION})",
        )
        model_params.add_argument(
            '--test-size',
            metavar='FLOAT',
            type=float,
            default=0.33,
            help="test size used in train-test-split (default: 0.33)",
        )

        # specify step (optionally)
        parser.add_argument(
            'action',
            choices=self.actions,
            nargs='?',
            help="action to take (default: all)",
        )

    def __call__(self, args, parser):
        actions = (args.action,) if args.action else self.actions

        # add'l help
        if args.help_algorithm:
            self.perform_help_algorithm()
            return

        if args.help_param:
            self.perform_help_param()
            return

        # dynamic argument checks
        if (self.action_extract in actions and not args.pcap
                and not args.pcap_normal
                and not args.pcap_abnormal):
            parser.error(f'at least one of the following arguments are required to '
                         f'"{self.action_extract}": -p/--pcap, --pcap-normal or --pcap-abnormal')

        if args.label and len(args.label) != len(args.pcap):
            parser.error("the number of files optionally specified by -l/--label must match those "
                         "specified by -p/--pcap")

        if args.pcap and not args.label and (args.pcap_normal or args.pcap_abnormal):
            parser.error("may not specify --pcap and --pcap-normal/--pcap-abnormal without also "
                         "specifying --label (both labeled and unlabeled pcaps)")

        if actions != self.actions and not args.feature:
            parser.error(f'the following arguments are required to "{self.action_extract}" or '
                         f'"{self.action_train}" in isolation: -f/--feature')

        if self.action_train in actions and not args.output:
            parser.error(f'the following arguments are required to "{self.action_train}": '
                         f'-o/--output/-m/--model')

        # checked now, rather than after a possibly long extraction
        if self.action_train in actions:
            self.model_params = self.check_model_params(parser)

        train_only = self.action_extract not in actions
        if train_only and not args.feature.is_file():
            parser.error(f'features file not found: {args.feature}')

        # actions
        with contextlib.ExitStack() as stack:
            # ensure file descriptors close
            # (otherwise closed at exit; but, at least, this doesn't work for tests)
            for file_descriptor in itertools.chain(
                args.pcap or (),
                args.pcap_normal or (),
                args.pcap_abnormal or (),
                args.label or (),
            ):
                stack.enter_context(file_descriptor)

            # extract
            if self.action_extract in actions:
                pcaps = args.pcap or ()
                labels = ({'label_file': label} for label in args.label or itertools.repeat(None))

                pcaps_normal = args.pcap_normal or ()
                labels_normal = ({'label': LABEL_NORMAL} for _count in itertools.count())

                pcaps_abnormal = args.pcap_abnormal or ()
                labels_abnormal = ({'label': LABEL_ABNORMAL} for _count in itertools.count())

                try:
                    feature_data = self.perform_extract(
                        itertools.chain(
                            zip(pcaps, labels),
                            zip(pcaps_normal, labels_normal),
                            zip(pcaps_abnormal, labels_abnormal),
                        ),
                    )
                except NoFlowsError as exc:
                    parser.error(str(exc))

                # written only now that extraction succeeded
                if args.feature:
                    dump_data(feature_data, out_file=args.feature)
            else:
                feature_data = load_data(args.feature)

            # train (on the features in memory: no need to re-read them from the file)
            if self.action_train in actions:
                self.perform_train(feature_data, args.output)

    def perform_extract(self, pcap_file_labels):
        """Extract features (and labels, if any) from the pcaps.

        Returns a FeatureData: unpacks as a plain (features, labels) pair, and also carries the
        extraction settings in .params.

        """
        # all pcaps share one split interval and one feature length (see extract_pcaps), so
        # their features are comparable and can simply be concatenated.
        #
        # NOTE: features are not scaled/normalized; a scaler would have to be fit on the training
        # NOTE: data and saved with the model, like the extraction settings.
        (pcaps, params) = self.extract_pcaps(pcap_file_labels)

        if not pcaps:
            raise ValueError("nothing to extract")

        features = np.concatenate([pcap.features for pcap in pcaps])

        if any(pcap.labels is None for pcap in pcaps):
            labels = None
        else:
            labels = np.concatenate([pcap.labels for pcap in pcaps])

        self.vprint(2, f'feature-extraction settings: {params}')

        return FeatureData(features, labels, params)

    def check_model_params(self, parser):
        """Get the --param values for the chosen algorithm, and check them against the model.

        --param may be:
        * unspecified (None)
        * just a dict for this model
        * a dict with sub-dicts for several models (keys are algorithm names)

        """
        params = self.args.param or {}
        algorithm = self.args.algorithm

        is_per_algorithm = bool(params) and all(
            key in self.train_algorithms and isinstance(value, dict)
            for (key, value) in params.items()
        )
        if is_per_algorithm:
            params = params.get(algorithm, {})

        if 'random_state' in params:
            parser.error('--param: set random_state with --random-state instead')

        model_class = self.load_algorithmic_model(algorithm)
        if isinstance(model_class, Exception):
            parser.error(f'algorithm {algorithm} is not available: {model_class}')

        accepted = [name for name in inspect.signature(model_class).parameters
                    if name != 'random_state']
        unknown = [name for name in params if name not in accepted]
        if unknown:
            parser.error(f'--param: {", ".join(map(str, unknown))} is not a parameter of '
                         f'algorithm {algorithm}; its parameters are: {", ".join(accepted)}')

        return self.add_contamination(parser, dict(params), accepted)

    def add_contamination(self, parser, params, accepted):
        """Set the model parameter holding the expected fraction of anomalies (pyod models:
        "contamination", OCSVM: "nu") from --contamination, from --param, or to the default.

        Then report the value and what it means for classify.

        """
        algorithm = self.args.algorithm
        names = [name for name in CONTAMINATION_PARAMS if name in accepted]
        if not names:
            parser.error(f'algorithm {algorithm} has no parameter for the expected fraction of '
                         'anomalies, so --contamination does not apply to it')
        name = names[0]

        if self.args.contamination is not None and name in params:
            parser.error(f'--contamination and --param both set {name} (the expected fraction of '
                         'anomalies): set it once')

        if name in params:
            (value, option) = (params[name], f'--param {name}')
        elif self.args.contamination is not None:
            (value, option) = (self.args.contamination, '--contamination')
        else:
            (value, option) = (DEFAULT_CONTAMINATION, '--contamination')

        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not is_number or not 0 < value <= 0.5:
            parser.error(f'{option} must be a fraction in (0, 0.5], not {value!r}: it is the '
                         'expected fraction of anomalies (0 would flag almost nothing: the '
                         'threshold would be the most anomalous training flow)')

        params[name] = value

        self.vprint(1, f'contamination {value:g} ({algorithm}: {name}={value:g}): classify will '
                       f'flag about {value:.0%} of flows similar to the training data '
                       f'(set with --contamination)')

        return params

    def perform_train(self, feature_data, output_path):
        """Train (and test, if labeled) a model, and save it to output_path.

        feature_data: a FeatureData, or a plain (features, labels) pair from older netml versions.

        """
        (features, labels) = feature_data

        # feature files saved by older versions of netml are plain tuples without settings
        extraction_params = getattr(feature_data, 'params', None)

        # train (and test split)
        #
        # TODO: might make sense to split testing into a separate "action," such that
        # TODO: it's done by default, but also can be applied modularly (on its own)
        #
        if labels is None:
            features_train = features
            features_test = labels_test = _labels_train = None

            self.vtable(2, [
                ('', 'features', 'labels'),
                ('train', features_train.shape, 'n/a'),
                ('test', 'n/a', 'n/a'),
            ], title='data shapes')
        elif len(np.unique(labels)) < 2:
            print('warning: all flows have the same label (only one class): the model is trained, '
                  'but not tested -- testing (AUC) needs both normal and abnormal flows',
                  file=sys.stderr)
            features_train = features
            features_test = labels_test = _labels_train = None
        else:
            (
                features_train,
                features_test,
                _labels_train,
                labels_test,
            ) = sklearn.model_selection.train_test_split(features,
                                                         labels,
                                                         test_size=self.args.test_size,
                                                         random_state=self.args.random_state)

            if len(np.unique(labels_test)) < 2:
                print('warning: the test split has flows of only one class: the model is not '
                      'tested -- testing (AUC) needs both normal and abnormal flows; try a larger '
                      '--test-size', file=sys.stderr)
                features_test = labels_test = None

            self.vtable(2, [
                ('', 'features', 'labels'),
                ('train', features_train.shape, 'n/a'),
                ('test', getattr(features_test, 'shape', 'n/a'),
                 getattr(labels_test, 'shape', 'n/a')),
            ], title='data shapes')

        self.vprint(2, f'model name: {self.args.algorithm}')

        params = self.model_params
        self.vprint(2, f'param override: {params}')

        model_class = self.load_algorithmic_model(self.args.algorithm)

        # create detection model
        model = model_class(
            random_state=self.args.random_state,
            **params
        )
        model.name = self.args.algorithm  # FIXME

        # saved with the model, so "classify" extracts features the same way
        model.extraction_params = extraction_params
        ndm = netml.ndm.model.MODEL(
            model,
            score_metric='auc',
            verbose=self.args.verbosity,
            random_state=self.args.random_state,
        )

        # train the model from the train set
        ndm.train(features_train)
        time_train = ndm.train.tot_time

        # evaluate the model
        if features_test is not None and labels_test is not None:
            ndm.test(features_test, labels_test)
            time_test = ndm.test.tot_time
        else:
            time_test = None

        # dump data to disk
        # written only now that training succeeded
        dump_data((model, ndm.history), out_file=output_path)

        self.vtable(1, [
            ('train time (m)', 'test time (m)', 'model score (auc)'),
            (
                time_train,
                'n/a' if time_test is None else time_test,
                getattr(ndm, 'score', 'n/a'),
            ),
        ], title='training performance')

    def perform_help_algorithm(self, title_addendum='', include_docs=True):
        print(f"{self.args.__parser__.prog}: model-training algorithms", title_addendum)

        for (model_name, model_class) in self.load_algorithmic_model().items():
            print('\n', model_name, ':', sep='')

            if isinstance(model_class, ManualDependencyError):
                print('  [Algorithm could not be loaded due to missing dependencies]', end='\n\n')
                print(textwrap.indent(str(model_class), '  '))

                continue

            # defaults as they might be overridden in yaml
            model_signature = inspect.signature(model_class)
            model_defaults = {
                parameter.name: parameter.default
                for parameter in model_signature.parameters.values()
            }
            print(textwrap.indent(yaml.dump(model_defaults).strip(), '  '))

            if include_docs:
                print('\n  ---', end='\n\n')

                # documentation
                doc = model_class.__doc__ or model_class.__init__.__doc__

                # fix indentation
                #
                # first line may not be indented at all
                if doc[0] in (' ', '\n'):
                    doc = textwrap.dedent(doc)
                else:
                    line_end = doc.find('\n') + 1
                    if line_end > 1:
                        doc = doc[:line_end] + textwrap.dedent(doc[line_end:])

                print(textwrap.indent(doc.strip(), '  '))

    def perform_help_param(self):
        self.perform_help_algorithm(
            title_addendum='(params only)',
            include_docs=False,
        )

    def load_algorithmic_model(self, model_name=None):
        models = {}

        # module names *should* be same as algorithm & class but we allow them to differ
        # NOTE: could simplify this by just mandating that module names equal to model class names
        for module_info in pkgutil.iter_modules(netml.ndm.__path__):
            if module_info.name == 'model':
                # NOTE: could just move this module aside
                continue

            try:
                module = importlib.import_module(f'netml.ndm.{module_info.name}')
            except ManualDependencyError as exc:
                # in the case of printing help we'll just report this algo as unconfigured
                # so: defer to caller
                models[module_info.name] = exc
                found = module_info.name
            else:
                try:
                    # optimistic naming for most models
                    models[module_info.name] = getattr(module, module_info.name.upper())
                    found = module_info.name
                except AttributeError:
                    # fallback to iteration over all possibilities
                    for algorithm in self.train_algorithms:
                        if algorithm in models:
                            continue

                        try:
                            models[algorithm] = getattr(module, algorithm.upper())
                        except AttributeError:
                            pass
                        else:
                            found = algorithm
                            break

            if found == model_name:
                return models[model_name]
        else:
            if model_name:
                raise LookupError(f"could not find algorithmic model: {model_name}")

        missing_models = set(self.train_algorithms) - models.keys()
        if missing_models:
            raise LookupError(f"failed to load algorithm models: {missing_models}")

        unexpected_models = models.keys() - set(self.train_algorithms)
        if unexpected_models:
            # this isn't really a big deal -- can exclude -- but shouldn't happen for now
            raise RuntimeError(f"unanticipated models loaded: {unexpected_models}")

        return models
