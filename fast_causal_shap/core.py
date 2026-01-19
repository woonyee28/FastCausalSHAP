import json
from collections import defaultdict
from math import factorial

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class FastCausalSHAP:
    def __init__(self, data, model, target_variable):
        self.data = data
        self.model = model
        self.gamma = None
        self.target_variable = target_variable
        self.ida_graph = None
        self.regression_models = {}
        self.feature_depths = {}
        self.path_cache = {}
        self.causal_paths = {}

    def remove_cycles(self):
        """
        Detects cycles in the graph and removes edges causing cycles.
        Returns a list of removed edges.
        """
        G = self.ida_graph.copy()
        removed_edges = []

        # Find all cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
        except nx.NetworkXNoCycle:
            return []  # No cycles found

        while cycles:
            # Get the current cycle
            cycle = cycles[0]

            # Find the edge with the smallest weight in the cycle
            min_weight = float("inf")
            edge_to_remove = None

            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]

                if G.has_edge(source, target):
                    weight = abs(G[source][target]["weight"])
                    if weight < min_weight:
                        min_weight = weight
                        edge_to_remove = (source, target)

            if edge_to_remove:
                # Remove the edge with the smallest weight
                G.remove_edge(*edge_to_remove)
                removed_edges.append(
                    (
                        edge_to_remove[0],
                        edge_to_remove[1],
                        self.ida_graph[edge_to_remove[0]][edge_to_remove[1]]["weight"],
                    )
                )

                # Recalculate cycles after removing an edge
                try:
                    cycles = list(nx.simple_cycles(G))
                except nx.NetworkXNoCycle:
                    cycles = []  # No more cycles
            else:
                break

        # Update the graph
        self.ida_graph = G
        return removed_edges

    def _compute_causal_paths(self):
        """Compute and store all causal paths to target for each feature."""
        features = [col for col in self.data.columns if col != self.target_variable]
        for feature in features:
            try:
                # Store the actual paths instead of just the features
                paths = list(
                    nx.all_simple_paths(self.ida_graph, feature, self.target_variable)
                )
                self.causal_paths[feature] = paths
            except nx.NetworkXNoPath:
                self.causal_paths[feature] = []

    def load_causal_strengths(self, json_file_path):
        with open(json_file_path, "r") as f:
            causal_effects_list = json.load(f)

        G = nx.DiGraph()
        nodes = list(self.data.columns)
        G.add_nodes_from(nodes)

        for item in causal_effects_list:
            pair = item["Pair"]
            mean_causal_effect = item["Mean_Causal_Effect"]
            if mean_causal_effect is None:
                continue
            source, target = pair.split("->")
            source = source.strip()
            target = target.strip()
            G.add_edge(source, target, weight=mean_causal_effect)
        self.ida_graph = G.copy()

        removed_edges = self.remove_cycles()
        if removed_edges:
            print(f"Removed {len(removed_edges)} edges to make the graph acyclic:")
            for source, target, weight in removed_edges:
                print(f"  {source} -> {target} (weight: {weight})")

        self._compute_feature_depths()
        self._compute_causal_paths()
        features = self.data.columns.tolist()
        beta_dict = {}

        for feature in features:
            if feature == self.target_variable:
                continue
            try:
                paths = list(
                    nx.all_simple_paths(G, source=feature, target=self.target_variable)
                )
            except nx.NetworkXNoPath:
                continue
            total_effect = 0
            for path in paths:
                effect = 1
                for i in range(len(path) - 1):
                    edge_weight = G[path[i]][path[i + 1]]["weight"]
                    effect *= edge_weight
                total_effect += effect
            if total_effect != 0:
                beta_dict[feature] = total_effect

        total_causal_effect = sum(abs(beta) for beta in beta_dict.values())
        if total_causal_effect == 0:
            self.gamma = {k: 0.0 for k in features}
        else:
            self.gamma = {
                k: abs(beta_dict.get(k, 0.0)) / total_causal_effect for k in features
            }
        return self.gamma

    def _compute_feature_depths(self):
        """Compute minimum depth of each feature to target in causal graph."""
        features = [col for col in self.data.columns if col != self.target_variable]
        for feature in features:
            try:
                all_paths = list(
                    nx.all_simple_paths(self.ida_graph, feature, self.target_variable)
                )
                min_depth = float("inf")
                for path in all_paths:
                    depth = len(path) - 1
                    min_depth = min(min_depth, depth)
                if min_depth != float("inf"):
                    self.feature_depths[feature] = min_depth
            except nx.NetworkXNoPath:
                continue

    def get_topological_order(self, S):
        """Returns the topological order of variables after intervening on subset S."""
        G_intervened = self.ida_graph.copy()
        for feature in S:
            G_intervened.remove_edges_from(list(G_intervened.in_edges(feature)))
        missing_nodes = set(self.data.columns) - set(G_intervened.nodes)
        G_intervened.add_nodes_from(missing_nodes)

        try:
            order = list(nx.topological_sort(G_intervened))
        except nx.NetworkXUnfeasible:
            raise ValueError("The causal graph contains cycles.")

        return order

    def get_parents(self, feature):
        """Returns the parent features for a given feature in the causal graph."""
        return list(self.ida_graph.predecessors(feature))

    def sample_marginal(self, feature):
        """Sample a value from the marginal distribution of the specified feature."""
        return self.data[feature].sample(1).iloc[0]

    def sample_conditional(self, feature, parent_values):
        """Sample a value for a feature conditioned on its parent features."""
        effective_parents = [
            p for p in self.get_parents(feature) if p != self.target_variable
        ]
        if not effective_parents:
            return self.sample_marginal(feature)
        model_key = (feature, tuple(sorted(effective_parents)))
        if model_key not in self.regression_models:
            X = self.data[effective_parents].values
            y = self.data[feature].values
            reg = LinearRegression()
            reg.fit(X, y)
            residuals = y - reg.predict(X)
            std = residuals.std()
            self.regression_models[model_key] = (reg, std)
        reg, std = self.regression_models[model_key]
        parent_values_array = np.array(
            [parent_values[parent] for parent in effective_parents]
        ).reshape(1, -1)
        mean = reg.predict(parent_values_array)[0]
        sampled_value = np.random.normal(mean, std)
        return sampled_value

    def compute_v_do(self, S, x_S, is_classifier=False):
        """Compute interventional expectations with caching."""
        cache_key = (
            frozenset(S),
            tuple(sorted(x_S.items())) if len(x_S) > 0 else tuple(),
        )

        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        variables_order = self.get_topological_order(S)

        sample = {}
        for feature in S:
            sample[feature] = x_S[feature]
        for feature in variables_order:
            if feature in S or feature == self.target_variable:
                continue
            parents = self.get_parents(feature)
            parent_values = {
                p: x_S[p] if p in S else sample[p]
                for p in parents
                if p != self.target_variable
            }
            if not parent_values:
                sample[feature] = self.sample_marginal(feature)
            else:
                sample[feature] = self.sample_conditional(feature, parent_values)

        intervened_data = pd.DataFrame([sample])
        intervened_data = intervened_data[self.model.feature_names_in_]
        if is_classifier:
            probas = self.model.predict_proba(intervened_data)[:, 1]
        else:
            probas = self.model.predict(intervened_data)

        result = np.mean(probas)
        self.path_cache[cache_key] = result
        return result

    def is_on_causal_path(self, feature, S, target_feature):
        """Check if feature is on any causal path from S to target_feature."""
        if target_feature not in self.causal_paths:
            return False
        path_features = self.causal_paths[target_feature]
        return feature in path_features

    def compute_modified_shap_proba(self, x, is_classifier=False):
        """TreeSHAP-inspired computation using causal paths and dynamic programming."""
        features = [col for col in self.data.columns if col != self.target_variable]
        phi_causal = {feature: 0.0 for feature in features}

        data_without_target = self.data.drop(columns=[self.target_variable])
        if is_classifier:
            E_fX = self.model.predict_proba(data_without_target)[:, 1].mean()
        else:
            E_fX = self.model.predict(data_without_target).mean()

        x_ordered = x[self.model.feature_names_in_]
        if is_classifier:
            f_x = self.model.predict_proba(x_ordered.to_frame().T)[0][1]
        else:
            f_x = self.model.predict(x_ordered.to_frame().T)[0]

        sorted_features = sorted(features, key=lambda f: self.feature_depths.get(f, 0))
        max_path_length = max(self.feature_depths.values(), default=0)
        shapley_weights = {}
        for m in range(max_path_length + 1):
            for d in range(m + 1, max_path_length + 1):
                shapley_weights[(m, d)] = (
                    factorial(m) * factorial(d - m - 1)
                ) / factorial(d)

        # Track contributions using dynamic programming (EXTEND-like logic in TreeSHAP)
        # m_values will accumulate contributions from subsets (use combinatorial logic)
        # Essentially, values in m_values[k] represent how many ways there are
        # to select k nodes from the path seen so far.
        for feature in sorted_features:
            if feature not in self.causal_paths:
                continue
            for path in self.causal_paths[feature]:
                path_features = [n for n in path if n != self.target_variable]
                d = len(path_features)
                m_values = defaultdict(float)
                m_values[0] = 1.0

                for node in path_features:
                    if node == feature:
                        continue

                    new_m_values = defaultdict(float)
                    for m, val in m_values.items():
                        new_m_values[m + 1] += val
                        new_m_values[m] += val
                    m_values = new_m_values

                for m in m_values:
                    weight = shapley_weights.get((m, d), 0) * self.gamma.get(feature, 0)
                    delta_v = self._compute_path_delta_v(
                        feature, path, m, x, is_classifier
                    )
                    phi_causal[feature] += weight * delta_v

        sum_phi = sum(phi_causal.values())
        if sum_phi != 0:
            scaling_factor = (f_x - E_fX) / sum_phi
            phi_causal = {k: v * scaling_factor for k, v in phi_causal.items()}

        return phi_causal

    def _compute_path_delta_v(self, feature, path, m, x, is_classifier):
        """Compute Δv for a causal path using precomputed expectations."""
        S = [n for n in path[:m] if n != feature]
        x_S = {n: x[n] for n in S if n in x}
        v_S = self.compute_v_do(S, x_S, is_classifier)

        S_with_i = S + [feature]
        x_Si = {**x_S, feature: x[feature]}
        v_Si = self.compute_v_do(S_with_i, x_Si, is_classifier)

        return v_Si - v_S
