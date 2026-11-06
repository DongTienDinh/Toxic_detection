from pathlib import Path
import unittest

try:
    import nbformat
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "nbformat unavailable")
class NotebookTests(unittest.TestCase):
    def test_notebooks_are_clean_and_configure_gpu_first(self):
        root = Path(__file__).resolve().parents[1] / "notebooks"
        paths = sorted(root.glob("*.ipynb"))
        self.assertEqual(
            {path.name for path in paths},
            {
                "00_prepare_vihsd.ipynb",
                "01_train_evaluate_visobert.ipynb",
                "02_train_evaluate_proposed_gate.ipynb",
                "03_train_evaluate_optimized_gate.ipynb",
                "04_train_evaluate_augmentation.ipynb",
                "05_train_evaluate_targeted_augmentation.ipynb",
                "06_gradio_toxic_detection_demo.ipynb",
                "07_package_seed42_checkpoint.ipynb",
            },
        )
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            self.assertTrue(code, path.name)
            first = code[0].source
            self.assertIn("CUDA_VISIBLE_DEVICES", first, path.name)
            self.assertNotRegex(
                first, r"(?m)^\s*(?:from|import)\s+torch\b", path.name
            )
            self.assertNotRegex(
                first, r"(?m)^\s*(?:from|import)\s+transformers\b", path.name
            )
            for cell in code:
                self.assertIsNone(cell.execution_count, path.name)
                self.assertEqual(cell.outputs, [], path.name)

    def test_proposed_notebook_trains_full_budget_and_loads_best_checkpoint(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "02_train_evaluate_proposed_gate.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("EPOCHS = 10.0", source)
        self.assertIn("load_best_model_at_end=True", source)
        self.assertIn("metric_for_best_model='macro_f1'", source)
        self.assertNotIn("EarlyStoppingCallback", source)

    def test_proposed_notebook_runs_locked_v3_ablation_then_tests_each(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "02_train_evaluate_proposed_gate.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("MODEL_VARIANTS = ('concat', 'gated')", source)
        self.assertIn("EVALUATE_TEST = True", source)
        self.assertIn("class ResidualDsConv", source)
        self.assertIn("groups=channels", source)
        self.assertIn("self.residual_scale_logit", source)
        self.assertIn("self.fusion_norm(g + torch.sigmoid", source)
        self.assertIn("test_frame = load_verified_frame('test')", source)
        self.assertLess(source.index("trainer.train()"), source.index("test_frame ="))
        self.assertIn("for variant in MODEL_VARIANTS:", source)
        self.assertIn("contexts.append(train_and_validate(variant))", source)
        self.assertIn("all_results.append(test_and_save(context))", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("'concat': 'ViSoBERT + Residual DS-CNN Concat v3'", source)
        self.assertIn("'gated': 'ViSoBERT + Residual DS-CNN Adaptive Gate v3'", source)
        self.assertIn("'decision_rule':'raw_argmax_no_test_tuning'", source)
        self.assertNotIn("fit_macro_f1_biases", source)

    def test_optimized_gate_notebook_runs_three_original_seeds_then_tests(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "03_train_evaluate_optimized_gate.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("RUN_SEEDS = (42, 52, 62)", source)
        self.assertIn("MODEL_VARIANT = 'original'", source)
        self.assertIn("EVALUATE_TEST = True", source)
        self.assertIn("'augmentation': 'none'", source)
        self.assertNotIn("build_controlled_aeda", source)
        self.assertIn("train_dataset = make_dataset(train_frame)", source)
        self.assertIn("class OptimizedGateViSoBERT", source)
        self.assertIn("class MultiScaleDsCnnPool", source)
        self.assertIn("class ResidualDsConv", source)
        self.assertIn("outputs.hidden_states[-self.config.last_n_layers:]", source)
        self.assertIn("self.residual_scale_logit", source)
        self.assertIn("self.toxic_head", source)
        self.assertIn("self.target_head", source)
        self.assertIn("self.config.aux_toxic_weight * toxic_loss", source)
        self.assertIn("self.config.aux_target_weight * target_loss", source)
        self.assertIn("content_special_ids=content_special_ids", source)
        self.assertIn("def fit_validation_biases", source)
        self.assertIn("'source': 'validation_only'", source)
        self.assertIn("def collect_gate_diagnostics", source)
        self.assertIn("raw_validation_metrics", source)
        self.assertIn("calibrated_validation_metrics", source)
        self.assertIn("matched_update_steps", source)
        self.assertIn("max_steps=matched_update_steps", source)
        self.assertIn(
            "validation_results = [train_and_validate(seed) for seed in RUN_SEEDS]",
            source,
        )
        self.assertIn("seed=seed, data_seed=seed", source)
        self.assertIn("def summarize_seed_frame", source)
        self.assertIn("validation_summary_mean_sd.csv", source)
        self.assertIn("load_best_model_at_end=True", source)
        self.assertIn("metric_for_best_model='macro_f1'", source)
        self.assertNotIn("EarlyStoppingCallback", source)
        self.assertLess(source.index("trainer.train()"), source.index("test_frame ="))
        self.assertIn("if not EVALUATE_TEST:", source)
        self.assertIn("for validation_result in validation_results:", source)
        self.assertIn("test_seed_results.csv", source)
        self.assertIn("test_summary_mean_sd.csv", source)
        self.assertIn("multi_seed_summary.json", source)
        self.assertIn("test_bias_source': 'validation_only'", source)

    def test_augmentation_notebook_reproduces_all_class_aeda_ablation(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "04_train_evaluate_augmentation.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("MODEL_VARIANTS = ('original', 'aeda_vi_v1')", source)
        self.assertIn("AUGMENT_RATE = 0.25", source)
        self.assertIn("AUGMENT_TARGET_LABELS = (0, 1, 2)", source)
        self.assertIn("EVALUATE_TEST = True", source)
        self.assertIn("def build_controlled_aeda", source)
        self.assertIn("qc_status': 'automated_checks_only_exploratory'", source)
        self.assertIn("class_weights_from_original", source)
        self.assertIn("augmentation_rows.to_parquet", source)
        self.assertIn("max_steps=matched_update_steps", source)
        self.assertIn("load_best_model_at_end=True", source)
        self.assertIn("metric_for_best_model='macro_f1'", source)
        self.assertLess(source.index("trainer.train()"), source.index("test_frame ="))
        self.assertIn(
            "validation_results = [train_and_validate(variant) for variant in MODEL_VARIANTS]",
            source,
        )
        self.assertIn("for variant_index, validation_result in enumerate", source)
        self.assertIn("paired_bootstrap_macro_f1_delta", source)
        self.assertIn("paired_test_comparison.json", source)
        self.assertIn("post_hoc_augmentation_ablation_after_nb03_test_observation", source)

    def test_targeted_augmentation_notebook_excludes_clean_candidates(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "05_train_evaluate_targeted_augmentation.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn(
            "MODEL_VARIANTS = ('original', 'aeda_vi_toxic50_v1')", source
        )
        self.assertIn("AUGMENT_RATE = 0.5", source)
        self.assertIn("AUGMENT_TARGET_LABELS = (1, 2)", source)
        self.assertIn("set(augmented.labels.astype(int).unique())", source)
        self.assertIn("target_labels", source)
        self.assertIn("source_train_sha256", source)
        self.assertIn("augmentation_hash", source)
        self.assertIn("train_frames = {'original': train_frame", source)
        self.assertIn("test_frame = load_verified_frame('test')", source)
        self.assertLess(source.index("trainer.train()"), source.index("test_frame ="))
        self.assertIn("raw_offensive_precision", source)
        self.assertIn("raw_hate_recall", source)
        self.assertIn("raw_clean_fpr", source)
        self.assertIn("paired_test_comparison.json", source)

    def test_gradio_demo_uses_balanced_correct_validation_examples(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "06_gradio_toxic_detection_demo.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("SAMPLES_PER_CLASS = {0: 2, 1: 2, 2: 2}", source)
        self.assertIn("TARGET_SEED = 42", source)
        self.assertIn("TARGET_VARIANT = 'original'", source)
        self.assertIn("require_exactly_one('vihsd_manifest.json')", source)
        self.assertIn("KAGGLE_INPUT.rglob('validation_result.json')", source)
        self.assertIn("payload.get('augmentation') != 'none'", source)
        self.assertIn("artifact_dir.name == f'seed_{seed}'", source)
        self.assertIn("seed_dir.name == 'original'", source)
        self.assertIn("model_type') != 'visobert-optimized-gate-v4'", source)
        self.assertIn("predictions_validation.parquet", source)
        self.assertIn("np.array_equal(truth_from_data, truth_from_predictions)", source)
        self.assertIn("sample_frame['labels'].astype(int) == sample_frame['y_pred']", source)
        self.assertIn("if len(demo_examples) != 6", source)
        self.assertIn("SHARE_APP = False", source)
        self.assertIn("raw softmax + argmax", source)
        self.assertNotIn("raw_macro_f1'].idxmax()", source)
        self.assertNotIn("predictions_test.parquet", source)

    def test_checkpoint_packager_exports_direct_input_and_zip(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "07_package_seed42_checkpoint.ipynb"
        )
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("PACKAGE_NAME = 'tmg-toxic-nb03-seed42'", source)
        self.assertIn("TARGET_SEED = 42", source)
        self.assertIn("TARGET_VARIANT = 'original'", source)
        self.assertIn("SEARCH_ROOT.rglob('model.safetensors')", source)
        self.assertIn("kagglehub.notebook_output_download(version_handle)", source)
        self.assertIn("weight_path.parent.parent.name == TARGET_VARIANT", source)
        self.assertIn("SOURCE_ORIGINAL_DIR", source)
        self.assertIn("source_augmentation is not None", source)
        self.assertIn("deployment_validation_result.update", source)
        self.assertIn("source_validation_result.json", source)
        self.assertIn("model.safetensors", source)
        self.assertIn("predictions_validation.parquet", source)
        self.assertIn("artifact_manifest.json", source)
        self.assertIn("compression=ZIP_STORED", source)
        self.assertIn("PACKAGE VERIFIED", source)
        self.assertNotIn("predictions_test.parquet", source)


if __name__ == "__main__":
    unittest.main()
