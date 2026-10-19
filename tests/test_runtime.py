"""Test d'intégration : le moteur C doit reproduire PyTorch.

Compile ``runtime/bcrun`` si nécessaire, exporte un petit modèle, puis compare
les logits du moteur C à ceux de PyTorch sur la même séquence.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blackcode.config import ModelConfig
from blackcode.export import export
from blackcode.model import BlackCode

ROOT = Path(__file__).resolve().parent.parent
BCRUN = ROOT / "runtime" / "bcrun"


def ensure_binary() -> bool:
    if BCRUN.exists():
        return True
    if not shutil.which("cc") and not shutil.which("gcc"):
        return False
    try:
        subprocess.run(["make", "-C", str(ROOT / "runtime")], check=True,
                       capture_output=True, timeout=120)
    except Exception:
        return False
    return BCRUN.exists()


@unittest.skipUnless(ensure_binary(), "bcrun non compilé et pas de compilateur C")
class TestRuntimeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        torch.manual_seed(0)
        cls.cfg = ModelConfig(name="rt", vocab_size=257, max_seq_len=32,
                              d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                              head_dim=16, ffn_hidden=128, conv_kernel=3,
                              attn_every=4)
        cls.model = BlackCode(cls.cfg).eval()
        ckpt = cls.dir / "ckpt.pt"
        torch.save({"model": cls.model.state_dict(), "cfg": cls.cfg.to_dict()}, ckpt)
        cls.pkg = cls.dir / "m.bcpkg"
        export(str(ckpt), str(cls.pkg), context=32, embed_bits=16)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, ids, dump):
        subprocess.run([str(BCRUN), "--pkg", str(self.pkg),
                        "--ids", ",".join(map(str, ids)), "--max-new", "0",
                        "--context", "32", "--dump-logits", str(dump), "--quiet"],
                       check=True, timeout=120)

    def test_logits_match_pytorch(self):
        ids = [3, 17, 42, 200, 1]
        with torch.no_grad():
            ref = self.model(torch.tensor([ids]))[0][0].numpy()
        dump = self.dir / "c.f32"
        self._run(ids, dump)
        got = np.fromfile(dump, dtype=np.float32)
        self.assertEqual(got.shape, ref.shape)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        self.assertLess(rel, 1e-3, f"erreur relative {rel:.4%}")
        self.assertEqual(int(np.argmax(got)), int(np.argmax(ref)))

    def test_mem_report_runs(self):
        out = subprocess.run(
            [str(BCRUN), "--pkg", str(self.pkg), "--ids", "1", "--max-new", "1",
             "--context", "32", "--mem-report"],
            check=True, capture_output=True, text=True, timeout=120).stdout
        self.assertIn("pic pendant l'inférence", out)
        self.assertIn("détail cache KV", out)

    def test_runtime_accepts_prompt_over_old_4096_limit(self):
        cfg = ModelConfig(name="rt-long", vocab_size=257, max_seq_len=4100,
                          d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                          head_dim=16, ffn_hidden=128, conv_kernel=3,
                          attn_every=4)
        model = BlackCode(cfg).eval()
        ckpt = self.dir / "long.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()}, ckpt)
        pkg = self.dir / "long.bcpkg"
        export(str(ckpt), str(pkg), context=4100, embed_bits=16)
        ids = ",".join(["1"] * 4100)
        out = subprocess.run(
            [str(BCRUN), "--pkg", str(pkg), "--ids", ids, "--max-new", "0",
             "--context", "4100", "--mem-report"],
            check=True, capture_output=True, text=True, timeout=120).stdout
        self.assertIn("prefill         : 4100 tokens", out)

    def test_yarn_rope_logits_match_pytorch(self):
        cfg = ModelConfig(name="rt-yarn", vocab_size=257, max_seq_len=64,
                          d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                          head_dim=16, ffn_hidden=128, conv_kernel=3,
                          attn_every=4, rope_scaling="yarn",
                          rope_original_max_seq_len=16, rope_factor=4.0)
        torch.manual_seed(17)
        model = BlackCode(cfg).eval()
        ckpt = self.dir / "yarn.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()}, ckpt)
        pkg = self.dir / "yarn.bcpkg"
        export(str(ckpt), str(pkg), context=64, embed_bits=16)
        ids = [3, 17, 42, 200, 1, 8, 12]
        with torch.no_grad():
            ref = model(torch.tensor([ids]))[0][0].numpy()
        dump = self.dir / "yarn.f32"
        subprocess.run([str(BCRUN), "--pkg", str(pkg), "--ids",
                        ",".join(map(str, ids)), "--max-new", "0",
                        "--context", "64", "--dump-logits", str(dump), "--quiet"],
                       check=True, timeout=120)
        got = np.fromfile(dump, dtype=np.float32)
        self.assertEqual(got.shape, ref.shape)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        self.assertLess(rel, 1e-3, f"erreur relative YaRN {rel:.4%}")


if __name__ == "__main__":
    unittest.main()
