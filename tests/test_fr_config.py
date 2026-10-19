"""Garde-fous de la cible francophone entraînée de zéro."""
from pathlib import Path
import unittest

from blackcode.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


class FrenchModelConfigTests(unittest.TestCase):
    def test_base_model_is_within_requested_parameter_range(self):
        cfg = ModelConfig.from_json(ROOT / "configs/blackcode_fr_260m.json")
        self.assertGreaterEqual(cfg.num_params, 200_000_000)
        self.assertLessEqual(cfg.num_params, 300_000_000)
        self.assertEqual(cfg.num_params, 258_578_048)
        self.assertFalse(cfg.ternary)
        self.assertTrue(cfg.tie_embeddings)

    def test_long_context_variant_keeps_parameter_count(self):
        cfg = ModelConfig.from_json(ROOT / "configs/blackcode_fr_260m_128k.json")
        self.assertEqual(cfg.num_params, 258_578_048)
        self.assertEqual(cfg.max_seq_len, 131_072)
        self.assertEqual(cfg.rope_scaling, "yarn")
        self.assertEqual(cfg.rope_factor, 8.0)


if __name__ == "__main__":
    unittest.main()
