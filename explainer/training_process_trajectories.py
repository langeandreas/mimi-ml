# Compact trajectory schema: tree evolution + SHAP evolution
import json
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

ROUND_DECIMALS = 4
TOP_K_SHAP_FEATURES = 12
TOP_K_TREE_FEATURES = 10
TOP_K_INTERACTIONS = 8


class TrajectorySummarizer:
    def __init__(self, explain, classification, save: bool = False, path: str = '../data/results/'):
        """Initializes the TrajectorySummarizer with the given parameters
        :param explain: An instance of the ExplainerCallback containing the training history
        :param classification: ClassificationClass containing the classification information (see classification__class.py)
        :param save: boolean indicating whether to save the summarized trajectories globally for all generated JSONs (default is False), individual jsons can be saved seperately
        :param path: path where the summarized trajectories will be saved
        """
        self.explain = explain
        self.classification = classification
        self.save = save
        self.path = path

    def _generate_output_file(self, default_filename: str) -> Path:
        """Returns a writable output file path.
        If self.path points to a JSON file, that file is used directly
        If self.path points to a directory, default_filename is created inside it
        """
        base_path = Path(self.path)
        if base_path.suffix.lower() == ".json":
            base_path.parent.mkdir(parents=True, exist_ok=True)
            return base_path

        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / default_filename

    def _generate_output_dir(self) -> Path:
        """Returns an output directory path for generated artifacts."""
        base_path = Path(self.path)
        output_dir = base_path.parent if base_path.suffix.lower() == ".json" else base_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _build_phase_summaries(progress):
        """Builds early/mid/late split-feature summaries from iteration progress."""
        phase_summaries = []
        if not progress:
            return phase_summaries

        n = len(progress)
        split_points = [0, n // 3, (2 * n) // 3, n]
        labels = ["early", "mid", "late"]

        for idx in range(3):
            start = split_points[idx]
            end = split_points[idx + 1]
            if end <= start:
                continue

            phase_steps = progress[start:end]
            counter = Counter()
            for step in phase_steps:
                for item in step.get("top_split_features", []):
                    counter[item["feature_index"]] += item["split_count"]

            top_phase = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            phase_summaries.append(
                {
                    "phase_name": labels[idx],
                    "iteration_range": [int(phase_steps[0]["iteration"]), int(phase_steps[-1]["iteration"])],
                    "top_phase_features": [
                        {"feature_index": int(fi), "split_count": int(cnt)}
                        for fi, cnt in top_phase
                    ],
                }
            )

        return phase_summaries

    @staticmethod
    def _round_float(x):
        return round(float(x), ROUND_DECIMALS)
    


    def _summarize_shap_step(self, entry):
        shap_values = np.asarray(entry["shap_values"])

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        top_idx = [
            int(i)
            for i in np.argsort(mean_abs)[::-1][:TOP_K_SHAP_FEATURES]
            if mean_abs[i] > 0
        ]

        return {
            "iteration": int(entry["iteration"]),
            "base_value": self._round_float(np.asarray(entry["base_value"]).mean()),
            "shap_magnitude": self._round_float(mean_abs.sum()),
            "top_shap_features": [
                {
                    "feature_index": int(i),
                    "mean_abs_shap": self._round_float(mean_abs[i]),
                    "mean_signed_shap": self._round_float(mean_signed[i]),
                }
                for i in top_idx
            ],
        }

    def _walk_tree(self, node, depth, path, split_counter, split_depth_sum, pair_counter, leaf_values, feature_to_index):
        """Recursively walks through the tree structure to extract split and leaf information.
        :param node: current node in the tree (can be a dict or a leaf value)
        :param depth: current depth in the tree
        :param path: list of feature indexes used in the path to the current node
        :param split_counter: counter to count how many times each feature is used for splitting
        :param split_depth_sum: counter to sum the depths at which each feature is used for splitting
        :param pair_counter: counter to count how many times each pair of features appears together in a path
        :param leaf_values: list to store the values of the leaf nodes
        :param feature_to_index: dictionary mapping feature names to their corresponding indexes
        :return: maximum depth encountered in the tree
        """
        if not isinstance(node, dict):
            return depth

        if "leaf" in node:
            leaf_values.append(float(node["leaf"]))
            path_unique = sorted(set(path))
            for left, right in combinations(path_unique, 2):
                pair_counter[(left, right)] += 1
            return depth

        split_name = str(node.get("split", ""))
        feature_idx = feature_to_index.get(split_name, -1)
        next_path = path

        if feature_idx >= 0:
            split_counter[feature_idx] += 1
            split_depth_sum[feature_idx] += depth
            next_path = path + [feature_idx]

        max_depth = depth
        for child in node.get("children", []):
            max_depth = max(
                max_depth,
                self._walk_tree(child, depth + 1, next_path, split_counter, split_depth_sum, pair_counter, leaf_values, feature_to_index),
            )

        return max_depth

    def _summarize_tree_step(self, entry):
        """Summarizes a single tree step by extracting relevant information from the tree structure and calculating statistics on splits and leaf values.
        :param entry: dictionary containing the tree information for a specific iteration
        :return: dictionary summarizing the tree step
        """
        tree_raw = entry.get("tree")
        if tree_raw is None:
            return None

        tree_obj = json.loads(tree_raw) if isinstance(tree_raw, str) else tree_raw
        split_counter = Counter()
        split_depth_sum = Counter()
        pair_counter = Counter()
        leaf_values = []
        
        feature_names = [str(c) for c in self.classification.train_test["X_train"].columns]
        feature_to_index = {name: i for i, name in enumerate(feature_names)}

        max_depth = self._walk_tree(
            tree_obj,
            0,
            [],
            split_counter,
            split_depth_sum,
            pair_counter,
            leaf_values,
            feature_to_index,
        )

        top_features = sorted(split_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_TREE_FEATURES]
        top_pairs = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_INTERACTIONS]

        if leaf_values:
            leaf_stats = {
                "mean": self._round_float(np.mean(leaf_values)),
                "std": self._round_float(np.std(leaf_values)),
                "min": self._round_float(np.min(leaf_values)),
                "max": self._round_float(np.max(leaf_values)),
            }
        else:
            leaf_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "iteration": int(entry["iteration"]),
            "split_count": int(sum(split_counter.values())),
            "leaf_count": int(len(leaf_values)),
            "max_depth": int(max_depth),
            "top_split_features": [
                {
                    "feature_index": int(fi),
                    "split_count": int(cnt),
                    "avg_split_depth": self._round_float(split_depth_sum[fi] / cnt),
                }
                for fi, cnt in top_features
            ],
            "top_feature_interactions": [
                {
                    "left_feature_index": int(a),
                    "right_feature_index": int(b),
                    "cooccurrence_count": int(cnt),
                }
                for (a, b), cnt in top_pairs
            ],
            "leaf_value_stats": leaf_stats,
        }

    def generate_tree_json(self, save=None):
        """Generates the tree summary JSON
        :param save: override for class attribute save, saves the JSON to the path specified in the class attribute (default ../data/results)
        :return: dictionary containing the tree summary
        """
        feature_names = [str(c) for c in self.classification.train_test["X_train"].columns]
        progress = []

        for entry in self.explain.trees:
            tree_step = self._summarize_tree_step(entry)
            if tree_step is not None:
                progress.append(tree_step)
        tree_json = {
            "feature_names": feature_names,
            "phase_summaries": self._build_phase_summaries(progress),
            "iteration_progress": progress,
        }
        if save or self.save: # enable save override with parameter or class attribute
            output_file = self._generate_output_file("trajectory_tree.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(tree_json, fp, ensure_ascii=False, separators=(',', ':'))
        return tree_json

    def generate_shap_json(self, save=None):
        """Generates the SHAP summary JSON.
        :param save: If True, saves the JSON to the path specified in the class attribute. Default is None, which falls back to the class attribute.
        :return: A dictionary containing the SHAP summary."""
        shap_steps = [self._summarize_shap_step(e) for e in self.explain.shap_values_history]
        sample_size = 0
        if self.explain.shap_values_history:
            sample_size = int(np.asarray(self.explain.shap_values_history[0]["shap_values"]).shape[0])

        shap_json = {
            "shap_sample_size": sample_size,
            "iteration_progress": shap_steps,
        }
        if self.save or save: # enable save override with parameter or class attribute
            output_file = self._generate_output_file("trajectory_shap.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(shap_json, fp, ensure_ascii=False, separators=(',', ':'))

        return shap_json

    def summarize_training_process(self, country_iso: str, mn: str, save=None):
        """
        Generates the trajectory JSON by combining tree and SHAP summaries.
        :param country_iso: ISO code of the country (used for saving)
        :param mn: Model name or identifier (used for saving)
        :return: A dictionary containing the summarized training process.
        """

        tree_json = self.generate_tree_json()
        shap_json = self.generate_shap_json()

        feature_names = tree_json["feature_names"]
        shap_by_iter = {
            step["iteration"]: step
            for step in shap_json["iteration_progress"]
        }

        progress = []
        for tree_step in tree_json["iteration_progress"]:
            if tree_step is None:
                continue

            shap_step = shap_by_iter.get(tree_step["iteration"])

            if shap_step is not None:
                tree_step["base_value"] = shap_step["base_value"]
                tree_step["shap_magnitude"] = shap_step["shap_magnitude"]
                tree_step["top_shap_features"] = shap_step["top_shap_features"]
            progress.append(tree_step)

        used_feature_indexes = sorted({
            item["feature_index"]
            for step in progress
            for item in step.get("top_split_features", [])
        } | {
            item["feature_index"]
            for step in progress
            for item in step.get("top_shap_features", [])
        } | {
            item["left_feature_index"]
            for step in progress
            for item in step.get("top_feature_interactions", [])
        } | {
            item["right_feature_index"]
            for step in progress
            for item in step.get("top_feature_interactions", [])
        })

        feature_lookup = [feature_names[i] for i in used_feature_indexes]
        feature_index_map = {old_i: new_i for new_i, old_i in enumerate(used_feature_indexes)}

        for step in progress:
            step["top_split_features"] = [
                {
                    "feature_index": feature_index_map[item["feature_index"]],
                    "split_count": item["split_count"],
                    "avg_split_depth": item["avg_split_depth"],
                }
                for item in step.get("top_split_features", [])
            ]
            if "top_shap_features" in step:
                step["top_shap_features"] = [
                    {
                        "feature_index": feature_index_map[item["feature_index"]],
                        "mean_abs_shap": item["mean_abs_shap"],
                        "mean_signed_shap": item["mean_signed_shap"],
                    }
                    for item in step["top_shap_features"]
                ]
            step["top_feature_interactions"] = [
                {
                    "left_feature_index": feature_index_map[item["left_feature_index"]],
                    "right_feature_index": feature_index_map[item["right_feature_index"]],
                    "cooccurrence_count": item["cooccurrence_count"],
                }
                for item in step.get("top_feature_interactions", [])
            ]

        feature_first_seen_iteration = [-1] * len(feature_lookup)
        for step in progress:
            for item in step.get("top_split_features", []):
                feature_index = item["feature_index"]
                if feature_first_seen_iteration[feature_index] == -1:
                    feature_first_seen_iteration[feature_index] = step["iteration"]

        phase_summaries = self._build_phase_summaries(progress)

        compact_trajectory_json = {
            "metadata": {
                "schema_version": 1,
                "feature_lookup": feature_lookup,
                "round_decimals": ROUND_DECIMALS,
                "top_k": {
                    "shap_features": TOP_K_SHAP_FEATURES,
                    "tree_features": TOP_K_TREE_FEATURES,
                    "feature_interactions": TOP_K_INTERACTIONS,
                },
                "shap_sample_size": shap_json["shap_sample_size"],
                "gain_available": 0,
            },
            "feature_first_seen_iteration": feature_first_seen_iteration,
            "phase_summaries": phase_summaries,
            "iteration_progress": progress,
        }

        if self.save or save:
            output_dir = self._generate_output_dir()
            output_file = output_dir / f"trajectory_compact_{self.classification.type_target}_{country_iso}_{self.classification.sampling}_best_model_{mn}.json"
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(compact_trajectory_json, fp, ensure_ascii=False, separators=(',', ':'))

        return compact_trajectory_json

