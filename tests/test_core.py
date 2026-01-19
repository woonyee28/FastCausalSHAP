import json
import os
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from fast_causal_shap.core import FastCausalSHAP


class TestFastCausalSHAP(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        np.random.seed(42)

        # Create synthetic regression data
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
        self.regression_data = pd.DataFrame(
            X, columns=["feature1", "feature2", "feature3", "feature4"]
        )
        self.regression_data["target"] = y

        # Create synthetic classification data
        X_clf, y_clf = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )
        self.classification_data = pd.DataFrame(
            X_clf, columns=["feature1", "feature2", "feature3", "feature4"]
        )
        self.classification_data["target"] = y_clf

        # Create mock models
        self.regression_model = LinearRegression()
        self.regression_model.fit(
            self.regression_data[["feature1", "feature2", "feature3", "feature4"]],
            self.regression_data["target"],
        )

        self.classification_model = RandomForestClassifier(random_state=42)
        self.classification_model.fit(
            self.classification_data[["feature1", "feature2", "feature3", "feature4"]],
            self.classification_data["target"],
        )

        # Create test instance
        self.fast_causal_shap = FastCausalSHAP(
            data=self.regression_data,
            model=self.regression_model,
            target_variable="target",
        )

        # Create sample causal effects for testing
        self.sample_causal_effects = [
            {"Pair": "feature1->target", "Mean_Causal_Effect": 0.5},
            {"Pair": "feature2->target", "Mean_Causal_Effect": 0.3},
            {"Pair": "feature3->feature1", "Mean_Causal_Effect": 0.2},
            {"Pair": "feature4->feature2", "Mean_Causal_Effect": 0.1},
        ]

    def _create_test_graph(self, edges=None):
        """Helper method to create a test graph with all nodes from data."""
        G = nx.DiGraph()
        all_nodes = self.regression_data.columns.tolist()
        G.add_nodes_from(all_nodes)
        if edges:
            G.add_edges_from(edges)
        return G

    def test_init(self):
        """Test FastCausalSHAP initialization."""
        fcs = FastCausalSHAP(
            data=self.regression_data,
            model=self.regression_model,
            target_variable="target",
        )

        self.assertEqual(fcs.target_variable, "target")
        self.assertIsNone(fcs.gamma)
        self.assertIsNone(fcs.ida_graph)
        self.assertEqual(fcs.regression_models, {})
        self.assertEqual(fcs.feature_depths, {})
        self.assertEqual(fcs.path_cache, {})
        self.assertEqual(fcs.causal_paths, {})

    def test_load_causal_strengths(self):
        """Test loading causal strengths from JSON file.

        This test verifies that the class can correctly load causal strengths from
        a JSON file, compute the gamma values, and create the causal graph.

        It tests the following:
        - Loading the JSON file correctly.
        - Graph construction from causal effects.
        - Normalization of causal strengths. (gamma values should sum to 1)
        - Proper handling of the ida_graph attribute.
        """
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.sample_causal_effects, f)
            temp_file = f.name

        try:
            # Test loading causal strengths
            gamma = self.fast_causal_shap.load_causal_strengths(temp_file)

            # Check that gamma is computed
            self.assertIsInstance(gamma, dict)
            self.assertIsNotNone(self.fast_causal_shap.gamma)
            self.assertIsNotNone(self.fast_causal_shap.ida_graph)

            # Check that graph is created
            self.assertIsInstance(self.fast_causal_shap.ida_graph, nx.DiGraph)

            # Check that gamma values sum to 1 (approximately)
            total_gamma = sum(abs(v) for v in gamma.values())
            self.assertAlmostEqual(total_gamma, 1.0, places=5)

        finally:
            os.unlink(temp_file)

    def test_remove_cycles(self):
        """Test cycle removal functionality.

        This test ensures that the class can detect and remove cycles from causal
        graph to create a Directed Acyclic Graph (DAG).
        """
        # Create a graph with cycles
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=0.5)
        G.add_edge("B", "C", weight=0.3)
        G.add_edge("C", "A", weight=0.1)  # Creates a cycle

        self.fast_causal_shap.ida_graph = G

        # Test cycle removal
        removed_edges = self.fast_causal_shap.remove_cycles()

        # Check that cycles are removed
        self.assertTrue(len(removed_edges) > 0)
        self.assertTrue(nx.is_directed_acyclic_graph(self.fast_causal_shap.ida_graph))

    def test_get_topological_order(self):
        """Test topological ordering."""
        # Set up a simple DAG
        G = nx.DiGraph()
        G.add_edges_from([("feature1", "feature2"), ("feature2", "target")])
        self.fast_causal_shap.ida_graph = G

        # Test topological order
        order = self.fast_causal_shap.get_topological_order([])
        self.assertIsInstance(order, list)

        # Test with intervention
        order_intervened = self.fast_causal_shap.get_topological_order(["feature1"])
        self.assertIsInstance(order_intervened, list)

    def test_get_parents(self):
        """Test getting parent nodes."""
        # Set up a simple graph
        G = nx.DiGraph()
        G.add_edges_from([("feature1", "feature2"), ("feature3", "feature2")])
        self.fast_causal_shap.ida_graph = G

        parents = self.fast_causal_shap.get_parents("feature2")
        self.assertEqual(set(parents), {"feature1", "feature3"})

        parents_no_parent = self.fast_causal_shap.get_parents("feature1")
        self.assertEqual(parents_no_parent, [])

    def test_sample_marginal(self):
        """Test marginal sampling."""
        sampled_value = self.fast_causal_shap.sample_marginal("feature1")
        self.assertIsInstance(sampled_value, (int, float, np.number))

        # Check that sampled value is within reasonable range
        feature_values = self.regression_data["feature1"]
        self.assertGreaterEqual(sampled_value, feature_values.min())
        self.assertLessEqual(sampled_value, feature_values.max())

    def test_sample_conditional(self):
        """Test conditional sampling."""
        # Set up a simple graph
        G = nx.DiGraph()
        G.add_edge("feature1", "feature2")
        self.fast_causal_shap.ida_graph = G

        parent_values = {"feature1": 0.5}
        sampled_value = self.fast_causal_shap.sample_conditional(
            "feature2", parent_values
        )
        self.assertIsInstance(sampled_value, (int, float, np.number))

    def test_compute_v_do(self):
        """Test interventional expectation computation.

        This test verifies the core do-calculus computation. This test checks:
        - Empty intervention handling.
        - Intervention with specific values.
        - Caching of results.
        """
        # Set up a simple graph with ALL nodes from data
        self.fast_causal_shap.ida_graph = self._create_test_graph(
            [("feature1", "feature2"), ("feature2", "target")]
        )

        # Test with empty intervention
        result = self.fast_causal_shap.compute_v_do([], {})
        self.assertIsInstance(result, (int, float, np.number))

        # Test with intervention
        result_intervened = self.fast_causal_shap.compute_v_do(
            ["feature1"], {"feature1": 0.5}
        )
        self.assertIsInstance(result_intervened, (int, float, np.number))

        # Test caching
        result_cached = self.fast_causal_shap.compute_v_do(
            ["feature1"], {"feature1": 0.5}
        )
        self.assertEqual(result_intervened, result_cached)

    def test_compute_modified_shap_proba_regression(self):
        """Test SHAP computation for regression."""
        # Load causal strengths first
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.sample_causal_effects, f)
            temp_file = f.name

        try:
            self.fast_causal_shap.load_causal_strengths(temp_file)

            # Test SHAP computation
            x = self.regression_data.iloc[0][
                ["feature1", "feature2", "feature3", "feature4"]
            ]
            shap_values = self.fast_causal_shap.compute_modified_shap_proba(
                x, is_classifier=False
            )

            # Check that result is a dictionary
            self.assertIsInstance(shap_values, dict)

            # Check that all features are present
            expected_features = ["feature1", "feature2", "feature3", "feature4"]
            for feature in expected_features:
                self.assertIn(feature, shap_values)
                self.assertIsInstance(shap_values[feature], (int, float, np.number))

            # Check that not all SHAP values are zero
            # At least one feature should have non-zero attribution
            total_abs_shap = sum(
                abs(shap_values[feature]) for feature in expected_features
            )
            self.assertGreater(
                total_abs_shap,
                0,
                "All SHAP values are zero - algorithm may not be working correctly",
            )

        finally:
            os.unlink(temp_file)

    def test_compute_modified_shap_proba_classification(self):
        """Test SHAP computation for classification."""
        # Create classification instance
        fcs_clf = FastCausalSHAP(
            data=self.classification_data,
            model=self.classification_model,
            target_variable="target",
        )

        # Load causal strengths
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.sample_causal_effects, f)
            temp_file = f.name

        try:
            fcs_clf.load_causal_strengths(temp_file)

            # Test SHAP computation for classification
            x = self.classification_data.iloc[0][
                ["feature1", "feature2", "feature3", "feature4"]
            ]
            shap_values = fcs_clf.compute_modified_shap_proba(x, is_classifier=True)

            # Check that result is a dictionary
            self.assertIsInstance(shap_values, dict)

            # Check that all features are present
            expected_features = ["feature1", "feature2", "feature3", "feature4"]
            for feature in expected_features:
                self.assertIn(feature, shap_values)
                self.assertIsInstance(shap_values[feature], (int, float, np.number))

            # Check that not all SHAP values are zero
            # At least one feature should have non-zero attribution
            total_abs_shap = sum(
                abs(shap_values[feature]) for feature in expected_features
            )
            self.assertGreater(
                total_abs_shap,
                0,
                "All SHAP values are zero - algorithm may not be working correctly",
            )

        finally:
            os.unlink(temp_file)

    def test_compute_feature_depths(self):
        """Test feature depth computation."""
        # Set up a graph with known depths - include ALL nodes
        self.fast_causal_shap.ida_graph = self._create_test_graph(
            [("feature1", "feature2"), ("feature2", "target"), ("feature3", "target")]
        )
        self.fast_causal_shap._compute_feature_depths()

        # Check computed depths
        self.assertEqual(self.fast_causal_shap.feature_depths["feature1"], 2)
        self.assertEqual(self.fast_causal_shap.feature_depths["feature3"], 1)

    def test_compute_causal_paths(self):
        """Test causal path computation."""
        # Set up a graph with known paths - include ALL nodes
        self.fast_causal_shap.ida_graph = self._create_test_graph(
            [("feature1", "feature2"), ("feature2", "target"), ("feature3", "target")]
        )
        self.fast_causal_shap._compute_causal_paths()

        # Check computed paths
        self.assertIn("feature1", self.fast_causal_shap.causal_paths)
        self.assertIn("feature3", self.fast_causal_shap.causal_paths)

        # Check that feature1 has a path through feature2
        feature1_paths = self.fast_causal_shap.causal_paths["feature1"]
        self.assertTrue(any("feature2" in path for path in feature1_paths))

    def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            # Implementation wraps JSONDecodeError in ValueError
            with self.assertRaises(ValueError) as context:
                self.fast_causal_shap.load_causal_strengths(temp_file)
            self.assertIn("Invalid JSON file", str(context.exception))
        finally:
            os.unlink(temp_file)

    def test_empty_causal_effects(self):
        """Test handling of empty causal effects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            temp_file = f.name

        try:
            # Implementation now raises ValueError for empty lists
            with self.assertRaises(ValueError) as context:
                self.fast_causal_shap.load_causal_strengths(temp_file)
            self.assertIn("empty list", str(context.exception))
        finally:
            os.unlink(temp_file)

    def test_compute_path_delta_v(self):
        """Test path delta V computation."""
        # Set up a simple graph with ALL nodes
        self.fast_causal_shap.ida_graph = self._create_test_graph(
            [("feature1", "target")]
        )
        self.fast_causal_shap._compute_feature_depths()
        self.fast_causal_shap._compute_causal_paths()

        # Create test data
        x = self.regression_data.iloc[0][
            ["feature1", "feature2", "feature3", "feature4"]
        ]
        path = ["feature1", "target"]

        # Test delta V computation
        delta_v = self.fast_causal_shap._compute_path_delta_v(
            "feature1", path, 0, x, False
        )
        self.assertIsInstance(delta_v, (int, float, np.number))


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
