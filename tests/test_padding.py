import tempfile
import unittest

try:
    import torch
    from transformers import XLMRobertaConfig
    from src.models import (
        balanced_class_weights,
        focal_cross_entropy,
        GlobalLocalConfig,
        ViSoBERTGlobalLocalForSequenceClassification,
    )
    AVAILABLE = True
except Exception:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "torch/transformers unavailable")
class PaddingTests(unittest.TestCase):
    def make_model(self, variant, **overrides):
        encoder = XLMRobertaConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=64,
            pad_token_id=1,
        )
        options = dict(
            encoder_config=encoder.to_dict(),
            encoder_model_id="TEST_ONLY",
            encoder_revision="0" * 40,
            variant=variant,
            cnn_kernels=[2, 3, 4],
            cnn_channels=4,
            fusion_dim=8,
            head_dropout=0.0,
            num_labels=3,
        )
        options.update(overrides)
        config = GlobalLocalConfig(**options)
        return ViSoBERTGlobalLocalForSequenceClassification(config)

    def test_all_custom_variants_shape_loss_and_gradients(self):
        ids = torch.tensor([[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]])
        mask = ids.ne(1).long()
        labels = torch.tensor([0, 2])
        for variant in ("mean", "cnn", "concat", "gated"):
            with self.subTest(variant=variant):
                model = self.make_model(variant)
                output = model(input_ids=ids, attention_mask=mask, labels=labels)
                self.assertEqual(tuple(output.logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.encoder.parameters()
                ))

    def test_padding_invariance_eval_fp32(self):
        model = self.make_model(
            "gated",
            cnn_pooling="max_mean",
            normalize_branches=True,
            residual_gate=True,
        ).eval()
        short_ids = torch.tensor([[0, 5, 6, 2]])
        short_mask = torch.ones_like(short_ids)
        batch_ids = torch.tensor([[0, 5, 6, 2, 1, 1], [0, 7, 8, 9, 10, 2]])
        batch_mask = batch_ids.ne(1).long()
        with torch.inference_mode():
            alone = model(input_ids=short_ids, attention_mask=short_mask).logits[0]
            together = model(input_ids=batch_ids, attention_mask=batch_mask).logits[0]
        torch.testing.assert_close(alone, together, atol=1e-4, rtol=1e-4)

    def test_trainer_num_items_kwarg_is_not_forwarded_to_encoder(self):
        model = self.make_model("gated")
        self.assertFalse(model.accepts_loss_kwargs)
        ids = torch.tensor([[0, 5, 6, 2]])
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=torch.tensor([1]),
            num_items_in_batch=torch.tensor(1),
        )
        self.assertEqual(tuple(output.logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(output.loss))

    def test_balanced_focal_loss_upweights_rare_classes(self):
        weights = balanced_class_weights([0] * 16 + [1] * 4 + [2], power=0.5)
        self.assertAlmostEqual(sum(weights), 3.0, places=6)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.5, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = focal_cross_entropy(
            logits, labels, class_weights=torch.tensor(weights), gamma=1.5
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
