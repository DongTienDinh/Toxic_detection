import gc
import tempfile
import unittest

try:
    import torch
    from transformers import XLMRobertaConfig
    from src.models import GlobalLocalConfig, ViSoBERTGlobalLocalForSequenceClassification
    AVAILABLE = True
except Exception:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "torch/transformers unavailable")
class CheckpointRoundtripTests(unittest.TestCase):
    def test_custom_model_save_reload_exact_logits(self):
        encoder = XLMRobertaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=32, pad_token_id=1,
        )
        config = GlobalLocalConfig(
            encoder_config=encoder.to_dict(), variant="gated",
            cnn_kernels=[2, 3], cnn_channels=4, fusion_dim=8,
            head_dropout=0.0, num_labels=3,
            encoder_model_id="TEST_ONLY", encoder_revision="0" * 40,
        )
        model = ViSoBERTGlobalLocalForSequenceClassification(config).eval()
        ids = torch.tensor([[0, 5, 6, 2]])
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            expected = model(input_ids=ids, attention_mask=mask).logits
        with tempfile.TemporaryDirectory() as root:
            model.save_pretrained(root, safe_serialization=True)
            loaded = ViSoBERTGlobalLocalForSequenceClassification.from_pretrained(root).eval()
            with torch.inference_mode():
                actual = loaded(input_ids=ids, attention_mask=mask).logits
            # Transformers 4.57/safetensors can retain a Windows mmap until the
            # loaded model is collected, which otherwise prevents fixture cleanup.
            del loaded
            gc.collect()
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
