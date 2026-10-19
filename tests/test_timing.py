import unittest

from blackcode.config import ModelConfig
from blackcode.timing import estimate, flops_per_token


class TestTiming(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig.from_json("configs/t4_small.json")

    def test_components_sum(self):
        d = flops_per_token(self.cfg, 1024)
        self.assertAlmostEqual(d["total"], d["body"] + d["attention"] + d["logits"])

    def test_attention_scales_linearly_with_context(self):
        a = flops_per_token(self.cfg, 1024)["attention"]
        b = flops_per_token(self.cfg, 2048)["attention"]
        self.assertAlmostEqual(b, 2 * a)

    def test_head_is_significant_at_small_scale(self):
        # À 51 M de paramètres avec 32k de vocabulaire, la tête coûte >20 %.
        d = flops_per_token(self.cfg, 1024)
        self.assertGreater(d["logits"] / d["total"], 0.20)

    def test_estimate_scales(self):
        a = estimate(self.cfg, 1e9, 1024, 25.0)
        b = estimate(self.cfg, 2e9, 1024, 25.0)
        c = estimate(self.cfg, 1e9, 1024, 50.0)
        self.assertAlmostEqual(b["hours"], 2 * a["hours"])
        self.assertAlmostEqual(c["hours"], a["hours"] / 2)

    def test_t4_small_under_a_day(self):
        est = estimate(self.cfg, 1e9, 1024, 22.75)
        self.assertLess(est["hours"], 24)


if __name__ == "__main__":
    unittest.main()
