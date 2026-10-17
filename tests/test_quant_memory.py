import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blackcode.config import ModelConfig
from blackcode.memory import budget, best_config_under_budget, kv_cache_mb
from blackcode.quant import (dequantize_ternary, pack_ternary, packing_efficiency,
                             ternary_codes, unpack_ternary)


class TestQuant(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 3, size=(7, 13), dtype=np.uint8)
        packed = pack_ternary(codes)
        self.assertEqual(packed.shape, (7, 4))
        out = unpack_ternary(packed, 13)
        np.testing.assert_array_equal(codes, out)

    def test_values_are_ternary(self):
        w = torch.randn(16, 32)
        codes, scales = ternary_codes(w)
        self.assertTrue(set(np.unique(codes)).issubset({0, 1, 2}))
        dq = dequantize_ternary(codes, scales)
        self.assertTrue(np.all(np.isin(np.round(dq / scales[:, None], 3), [-1, 0, 1])))

    def test_packing_efficiency(self):
        bits = packing_efficiency(2048, 2048)
        self.assertGreater(bits, 2.0)
        self.assertLess(bits, 2.1)


class TestMemory(unittest.TestCase):
    def test_budget_positive_and_kv_linear(self):
        cfg = ModelConfig.from_json("configs/t4_small.json")
        b1 = budget(cfg, 1024)
        b2 = budget(cfg, 4096)
        self.assertGreater(b2.total_mb, b1.total_mb)
        self.assertAlmostEqual(b2.kv_cache_mb, 4 * b1.kv_cache_mb, places=3)
        self.assertEqual(kv_cache_mb(cfg, 100), b1.kv_cache_mb * 100 / 1024)

    def test_one_billion_target_fits_under_1gb(self):
        cfg = ModelConfig.from_json("configs/blackcode_1b.json")
        b = budget(cfg, 8192, weight_bits=2.0, embed_bits=8.0)
        self.assertLess(b.total_mb, 1024, b.as_dict())

    def test_scan_returns_a_config(self):
        cfg = ModelConfig.from_json("configs/blackcode_1b.json")
        best = best_config_under_budget(1024, cfg, 8192)
        self.assertLessEqual(best["memory"]["total_mb"], 1024)
        self.assertGreater(best["params"], 0)


class TestExport(unittest.TestCase):
    def test_export_roundtrip_header(self):
        import json
        import struct
        from blackcode.export import export
        from blackcode.model import BlackCode

        cfg = ModelConfig(vocab_size=257, max_seq_len=32, d_model=64, n_layers=4,
                          n_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128,
                          conv_kernel=3, attn_every=4)
        torch.manual_seed(0)
        model = BlackCode(cfg)
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()}, ckpt)
            out = Path(d) / "m.bcpkg"
            export(str(ckpt), str(out), context=64)
            raw = out.read_bytes()
            self.assertTrue(raw.startswith(b"BCPK1\n"))
            hlen = struct.unpack("<I", raw[6:10])[0]
            header = json.loads(raw[10:10 + hlen])
            self.assertEqual(header["format"], "blackcode-packed-v1")
            self.assertEqual(len(header["tensors"]), len(model.state_dict()))
            self.assertLess(header["effective_bits_per_param"], 16)


if __name__ == "__main__":
    unittest.main()
