import unittest

import torch

from blackcode.config import ModelConfig
from blackcode.model import BlackCode, ConvBlock, AttentionBlock


def tiny_cfg(**kw):
    base = dict(name="test", vocab_size=257, max_seq_len=32, d_model=64,
                n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
                ffn_hidden=128, conv_kernel=3, attn_every=4)
    base.update(kw)
    return ModelConfig(**base)


class TestConfig(unittest.TestCase):
    def test_param_breakdown_sums(self):
        cfg = tiny_cfg()
        parts = cfg.param_breakdown()
        self.assertEqual(parts["total"],
                         parts["embeddings"] + parts["attention"]
                         + parts["convolution"] + parts["norms"])
        self.assertEqual(cfg.n_attention_layers, 1)
        self.assertEqual(cfg.n_conv_layers, 3)

    def test_tied_embeddings_halve_embedding_count(self):
        a = tiny_cfg(tie_embeddings=True).param_breakdown()
        b = tiny_cfg(tie_embeddings=False).param_breakdown()
        self.assertEqual(b["embeddings"], 2 * a["embeddings"])

    def test_block_types(self):
        cfg = tiny_cfg()
        model = BlackCode(cfg)
        types = [type(b).__name__ for b in model.blocks]
        self.assertEqual(types, ["ConvBlock", "ConvBlock", "ConvBlock", "AttentionBlock"])


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg()
        self.model = BlackCode(self.cfg).eval()

    def test_param_count_matches_formula(self):
        self.assertEqual(self.model.num_params(), self.cfg.num_params)

    def test_loss_is_finite_and_backward_reaches_ternary_weights(self):
        x = torch.randint(0, self.cfg.vocab_size, (2, 16))
        y = torch.randint(0, self.cfg.vocab_size, (2, 16))
        loss = self.model.forward_loss(x, y, logit_chunk=8)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad = self.model.blocks[0].in_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_incremental_cache_matches_full_forward(self):
        idx = torch.randint(0, self.cfg.vocab_size, (1, 12))
        with torch.no_grad():
            full, _ = self.model(idx)
            prefill = 7
            logits, caches = self.model(idx[:, :prefill], start_pos=0, use_cache=True)
            for i in range(prefill, 12):
                logits, caches = self.model(idx[:, i:i + 1], caches,
                                            start_pos=i, use_cache=True)
        self.assertTrue(torch.allclose(full, logits, atol=1e-4),
                        (full - logits).abs().max().item())

    def test_generate_shapes(self):
        idx = torch.randint(0, self.cfg.vocab_size, (1, 4))
        out = self.model.generate(idx, max_new_tokens=6, temperature=0.0)
        self.assertEqual(out.shape, (1, 10))

    def test_loop_reuses_parameters(self):
        base = BlackCode(tiny_cfg(loop=1))
        looped = BlackCode(tiny_cfg(loop=3))
        self.assertEqual(base.num_params(), looped.num_params())
        x = torch.randint(0, 257, (1, 8))
        with torch.no_grad():
            self.assertEqual(base(x)[0].shape, looped(x)[0].shape)


if __name__ == "__main__":
    unittest.main()
