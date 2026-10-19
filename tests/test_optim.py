import unittest

import torch

from blackcode.config import ModelConfig
from blackcode.model import BlackCode
from blackcode.optim import (AdEMAMix, Muon, cosine_multiplier,
                             estimate_optimizer_memory_mb, make_optimizer,
                             schedule_multiplier, wsd_multiplier,
                             zeropower_via_newtonschulz5)


def tiny_model():
    cfg = ModelConfig(vocab_size=257, max_seq_len=32, d_model=64, n_layers=4,
                      n_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128,
                      conv_kernel=3, attn_every=4)
    return BlackCode(cfg), cfg


class TestSchedules(unittest.TestCase):
    def test_warmup_then_constant_then_decay(self):
        m = [wsd_multiplier(s, 10, 100, 0.8, 0.02) for s in range(101)]
        self.assertAlmostEqual(m[0], 0.1, places=6)
        self.assertAlmostEqual(m[9], 1.0, places=6)
        self.assertAlmostEqual(m[50], 1.0, places=6)   # phase stable
        self.assertAlmostEqual(m[100], 0.02, places=6)
        for a, b in zip(m[81:], m[82:]):
            self.assertGreaterEqual(a + 1e-9, b)       # décroissance

    def test_stable_phase_allows_resume(self):
        # À 70 % du budget, la LR est encore pleine : on peut brancher.
        self.assertAlmostEqual(wsd_multiplier(70, 10, 100, 0.8), 1.0, places=6)

    def test_cosine_endpoints(self):
        self.assertAlmostEqual(cosine_multiplier(100, 10, 100, 0.1), 0.1, places=6)
        self.assertAlmostEqual(cosine_multiplier(9, 10, 100, 0.1), 1.0, places=6)

    def test_dispatch(self):
        for name in ("cosine", "wsd", "constant"):
            self.assertGreater(schedule_multiplier(50, name, 10, 100), 0.0)
        with self.assertRaises(ValueError):
            schedule_multiplier(0, "inconnu", 10, 100)


class TestMuon(unittest.TestCase):
    def test_newton_schulz_bounds_the_spectrum(self):
        torch.manual_seed(0)
        g = torch.randn(64, 128)
        x = zeropower_via_newtonschulz5(g, steps=8)
        singular = torch.linalg.svdvals(x.float())
        # Les coefficients de Muon donnent un spectre borné, pas exactement 1 :
        # c'est le comportement attendu de l'implémentation de référence.
        self.assertLess(singular.max().item() / singular.min().item(), 2.2)
        self.assertLess(abs(singular.mean().item() - 1.0), 0.2)

    def test_step_moves_parameters_without_nan(self):
        model, _ = tiny_model()
        opt, base = make_optimizer(model, name="muon", lr=6e-4, muon_lr=0.02)
        before = model.blocks[0].in_proj.weight.detach().clone()
        x = torch.randint(0, 257, (2, 16))
        loss = model.forward_loss(x, x, logit_chunk=8)
        loss.backward()
        opt.step()
        moved = (model.blocks[0].in_proj.weight - before).abs().sum().item()
        self.assertGreater(moved, 0.0)
        self.assertFalse(any(torch.isnan(p).any().item() for p in model.parameters()))
        self.assertIn(0.02, base)  # le groupe Muon a sa propre LR

    def test_muon_excludes_embeddings(self):
        model, cfg = tiny_model()
        opt, _ = make_optimizer(model, name="muon")
        muon_opt = opt.optimizers[0]
        self.assertIsInstance(muon_opt, Muon)
        muon_params = {id(p) for g in muon_opt.param_groups for p in g["params"]}
        self.assertNotIn(id(model.tok_emb.weight), muon_params)

    def test_muon_rejects_1d(self):
        opt = Muon([torch.nn.Parameter(torch.zeros(4))])
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones(4)
        with self.assertRaises(ValueError):
            opt.step()


class TestAdEMAMix(unittest.TestCase):
    def test_trains_and_matches_api(self):
        model, _ = tiny_model()
        opt, base = make_optimizer(model, name="ademamix", lr=6e-4)
        self.assertIsInstance(opt, AdEMAMix)
        for _ in range(3):
            x = torch.randint(0, 257, (2, 16))
            loss = model.forward_loss(x, x, logit_chunk=8)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertFalse(any(torch.isnan(p).any().item() for p in model.parameters()))


class TestMemoryEstimates(unittest.TestCase):
    def test_ordering(self):
        n = 1_200_000_000
        adam = estimate_optimizer_memory_mb(n, "adamw")["total_mb"]
        eight = estimate_optimizer_memory_mb(n, "adamw8bit")["total_mb"]
        adaf = estimate_optimizer_memory_mb(n, "adafactor")["total_mb"]
        self.assertGreater(adam, eight)
        self.assertGreater(adam, adaf)
        # AdamW fp32 complet dépasse la VRAM d'un T4 : c'est tout l'enjeu.
        self.assertGreater(adam, 13_000)


if __name__ == "__main__":
    unittest.main()
