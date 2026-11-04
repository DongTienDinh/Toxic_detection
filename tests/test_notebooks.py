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

    def test_proposed_notebook_runs_concat_and_gated_then_tests_each(self):
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
        self.assertIn("variant=variant", source)
        self.assertIn("test_ds = make_dataset(frames['test'])", source)
        self.assertLess(source.index("trainer.train()"), source.index("test_ds ="))
        self.assertIn("for variant in MODEL_VARIANTS:", source)
        self.assertIn("context = train_and_validate(variant)", source)
        self.assertIn("all_results.append(test_and_save(context))", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("'concat': 'ViSoBERT Global-Local Concat CNN v2'", source)
        self.assertIn("'gated': 'ViSoBERT Global-Local Gated CNN v2'", source)


if __name__ == "__main__":
    unittest.main()
