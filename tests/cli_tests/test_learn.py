"""Tests for the CLI sub-command `learn`"""
import contextlib
import io
import pathlib
import tempfile
import unittest
import warnings

import argcmdr

from netml import cli
from netml.utils.tool import load_data


# TODO: use smaller data files to speed up tests?

# the test data (copies of netml's examples/data pcaps; see tests/data/README.md)
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'

TEMPDIR_PATH = pathlib.Path(tempfile.gettempdir())

# Number of flows (subflows) that netml extracts from the example pcaps with the default
# q_interval of 0.9. Bug fix (2/5) changed them: splitting a flow used to drop its final subflow,
# and these counts used to encode that loss (88, 4,979 and 310 respectively).
N_DEMO_LABELED = 102     # demo.pcap flows that demo.csv labels
N_NORMAL = 5362          # srcIP_10.42.0.1_normal.pcap
N_ANOMALY = 328          # srcIP_10.42.0.119_anomaly.pcap

# Known bug, not fixed (see "Found but not fixed" in README.md): each pcap gets its own split
# interval and feature length, so features of pcaps with different lengths cannot be combined
# ("all the input array dimensions ... must match exactly"). demo.pcap gives 70 IAT values per
# flow and the other example pcaps 12, so the tests that combine them fail.
KNOWN_BUG_MULTI_PCAP = unittest.expectedFailure


def setUpModule():
    warnings.simplefilter('ignore', ResourceWarning)


class CLITestCase(unittest.TestCase):

    def try_execute(self, *argv):
        argcmdr.main(cli.Main, argv=map(str, argv))


class TestLearnExtractRequirements(CLITestCase):

    def test_extract_requires_pcap(self):
        """extract requires a pcap path"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                'extract',
                '-f', TEMPDIR_PATH / 'netml-test-features-a',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r'arguments are required .+: -p/--pcap')

    def test_extract_requires_features(self):
        """extract-only requires a features path"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                'extract',
                '-p', DATA_PATH / 'demo.pcap',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r'arguments are required .+: -f/--feature')

    def test_extract_requires_model(self):
        """extract-train requires a model output path"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                '-p', DATA_PATH / 'demo.pcap',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r'arguments are required .+: -o/--output/-m/--model')


class TestLearnExtractPcapOnly(CLITestCase):

    def test_extract_only(self):
        """extract features to path"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',
            '-p', DATA_PATH / 'demo.pcap',
            '-f', feature_path,
        )

        (features, labels) = load_data(feature_path)

        self.assertGreater(len(features), 0)
        self.assertIsNone(labels)

    def test_extract_train(self):
        """extract features and train model"""
        model_path = TEMPDIR_PATH / 'netml-test-output-a'

        self.try_execute(
            'learn',
            '-p', DATA_PATH / 'demo.pcap',
            '-o', model_path,
        )

        (model, train_history) = load_data(model_path)

        self.assertTrue(hasattr(model, 'predict'))
        self.assertFalse(train_history)

    def test_extract_multiple(self):
        """extract features from multiple unlabeled pcap files"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',
            '-f', feature_path,

            # (filenames reflect what's in them but here we specify no labels for testing)
            '-p', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.119_anomaly.pcap',
            '-p', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
        )

        (features, labels) = load_data(feature_path)

        self.assertEqual(len(features), N_ANOMALY + N_NORMAL)
        self.assertIsNone(labels)


class TestLearnExtractPcapLabels(CLITestCase):

    def test_extract_only(self):
        """extract features & labels to path"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',
            '-p', DATA_PATH / 'demo.pcap',
            '-l', DATA_PATH / 'demo.csv',
            '-f', feature_path,
        )

        (features, labels) = load_data(feature_path)

        self.assertEqual(len(features), N_DEMO_LABELED)
        self.assertEqual(len(labels), N_DEMO_LABELED)

    def test_extract_train(self):
        """extract features & labels and train & test model"""
        model_path = TEMPDIR_PATH / 'netml-test-output-a'

        self.try_execute(
            'learn',
            '-p', DATA_PATH / 'demo.pcap',
            '-l', DATA_PATH / 'demo.csv',
            '-o', model_path,
        )

        (model, train_history) = load_data(model_path)

        self.assertTrue(hasattr(model, 'predict'))

        self.assertIsNot(train_history, None)
        self.assertIn('score', train_history)
        self.assertGreater(train_history['score'], 0)

    def test_extract_multiple(self):
        """extract features & labels from multiple pcap & label files"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',
            '-f', feature_path,

            '-p', DATA_PATH / 'demo.pcap',
            '-p', DATA_PATH / 'demo.pcap',

            '-l', DATA_PATH / 'demo.csv',
            '-l', DATA_PATH / 'demo.csv',
        )

        (features, labels) = load_data(feature_path)

        self.assertEqual(len(features), 2 * N_DEMO_LABELED)
        self.assertEqual(len(labels), 2 * N_DEMO_LABELED)

    def test_extract_mismatch_length_0(self):
        """cannot specify too many label files for pcap files"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                'extract',
                '-f', TEMPDIR_PATH / 'netml-test-features-a',

                '-p', DATA_PATH / 'demo.pcap',

                '-l', DATA_PATH / 'demo.csv',
                '-l', DATA_PATH / 'demo.csv',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r'number of files .+ must match')

    def test_extract_mismatch_length_1(self):
        """cannot specify too few label files for pcap files"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                'extract',
                '-f', TEMPDIR_PATH / 'netml-test-features-a',

                '-p', DATA_PATH / 'demo.pcap',
                '-p', DATA_PATH / 'demo.pcap',

                '-l', DATA_PATH / 'demo.csv',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r'number of files .+ must match')


class TestLearnExtractPcapNormalAbnormal(CLITestCase):

    def test_extract_only(self):
        """extract flag-labeled features to path"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',

            '--pcap-normal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
            '--pcap-abnormal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.119_anomaly.pcap',

            '-f', feature_path,
        )

        (features, labels) = load_data(feature_path)

        self.assertEqual(len(features), N_NORMAL + N_ANOMALY)
        self.assertEqual(len(labels), N_NORMAL + N_ANOMALY)

        # 0 is normal
        self.assertFalse(labels[:N_NORMAL].any())

        # 1 is abnormal
        self.assertTrue(labels[N_NORMAL:].all())

    def test_extract_train(self):
        """extract flag-labeled features and train & test model"""
        model_path = TEMPDIR_PATH / 'netml-test-output-a'

        self.try_execute(
            'learn',

            '--pcap-normal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
            '--pcap-abnormal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.119_anomaly.pcap',

            '-o', model_path,
        )

        (model, train_history) = load_data(model_path)

        self.assertTrue(hasattr(model, 'predict'))

        self.assertIsNot(train_history, None)
        self.assertIn('score', train_history)
        self.assertGreater(train_history['score'], 0)


class TestLearnExtractPcapMixed(CLITestCase):

    @KNOWN_BUG_MULTI_PCAP
    def test_extract_only(self):
        """extract file-labeled & flag-labeled features to path"""
        feature_path = TEMPDIR_PATH / 'netml-test-features-a'

        self.try_execute(
            'learn',
            'extract',

            '-p', DATA_PATH / 'demo.pcap',
            '-l', DATA_PATH / 'demo.csv',

            '--pcap-normal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
            '--pcap-abnormal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.119_anomaly.pcap',

            '-f', feature_path,
        )

        (features, labels) = load_data(feature_path)

        self.assertEqual(len(features), N_DEMO_LABELED + N_NORMAL + N_ANOMALY)
        self.assertEqual(len(labels), N_DEMO_LABELED + N_NORMAL + N_ANOMALY)

        # 0 is normal; 1 is abnormal
        self.assertTrue(labels[:N_DEMO_LABELED].any())
        self.assertFalse(labels[:N_DEMO_LABELED].all())

        self.assertFalse(labels[N_DEMO_LABELED:N_DEMO_LABELED + N_NORMAL].any())

        self.assertTrue(labels[(N_DEMO_LABELED + N_NORMAL):].all())

    @KNOWN_BUG_MULTI_PCAP
    def test_extract_train(self):
        """extract file-labeled & flag-labeled features and train & test model"""
        model_path = TEMPDIR_PATH / 'netml-test-output-a'

        self.try_execute(
            'learn',

            '-p', DATA_PATH / 'demo.pcap',
            '-l', DATA_PATH / 'demo.csv',

            '--pcap-normal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
            '--pcap-abnormal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.119_anomaly.pcap',

            '-o', model_path,
        )

        (model, train_history) = load_data(model_path)

        self.assertTrue(hasattr(model, 'predict'))

        self.assertIsNot(train_history, None)
        self.assertIn('score', train_history)
        self.assertGreater(train_history['score'], 0)

    def test_unlabeled_pcap(self):
        """cannot specify unlabeled pcap & flag-labeled pcap(s)"""
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit) as context:
            self.try_execute(
                'learn',
                'extract',
                '-f', TEMPDIR_PATH / 'netml-test-features-a',

                '-p', DATA_PATH / 'demo.pcap',
                '--pcap-normal', DATA_PATH / 'srcIP_10.42.0.1' / 'srcIP_10.42.0.1_normal.pcap',
            )

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            'may not specify --pcap and --pcap-normal/--pcap-abnormal '
            'without also specifying --label',
            stderr.getvalue()
        )
