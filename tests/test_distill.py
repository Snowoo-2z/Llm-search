"""Tests de la distillation (CPU, modèles minuscules)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blackcode.config import ModelConfig
from blackcode.distill import LocalTeacher, kd_loss, teacher_targets
from blackcode.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(vocab: int = 256) -> ModelConfig:
    return ModelConfig(name="tiny", vocab_size=vocab, max_seq_len=64, d_model=64,
                       n_layers=2, n_heads=2, n_kv_heads=1, head_dim=32,
                       ffn_hidden=128, conv_kernel=4, attn_every=2,
                       tie_embeddings=True, ternary=True)


def make_corpus(path: Path, vocab: int = 256, tokens: int = 4096) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = rng.integers(0, vocab, size=tokens, dtype=np.uint16)
    val = rng.integers(0, vocab, size=256, dtype=np.uint16)
    train.tofile(path / "train.bin")
    val.tofile(path / "val.bin")
    (path / "meta.json").write_text(json.dumps(
        {"vocab_size": vocab, "train_tokens": int(tokens), "val_tokens": 256}))


class DistillUnitTests(unittest.TestCase):
    def test_teacher_targets_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg()
            model = build_model(cfg, "cpu")
            ckpt = Path(tmp) / "t.pt"
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ckpt)
            teacher = LocalTeacher(str(ckpt), "cpu")
            idx = torch.randint(0, cfg.vocab_size, (2, 16))
            values, indices = teacher_targets(teacher, idx, topk=8)
            self.assertEqual(values.shape, (2, 16, 8))
            self.assertEqual(indices.shape, (2, 16, 8))
            self.assertTrue(indices.max().item() < cfg.vocab_size)

    def test_kd_loss_zero_when_identical(self):
        logits = torch.randn(2, 5, 32)
        values = logits.topk(4, dim=-1).values
        indices = logits.topk(4, dim=-1).indices
        loss = kd_loss(logits, values, indices, temperature=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_kd_loss_positive_when_different(self):
        teacher = torch.randn(2, 5, 32)
        student = torch.randn(2, 5, 32)
        top = teacher.topk(4, dim=-1)
        self.assertGreater(float(kd_loss(student, top.values, top.indices, 1.0)), 0.0)


class DistillEndToEndTests(unittest.TestCase):
    def test_two_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tiny_cfg()
            model = build_model(cfg, "cpu")
            teacher_path = tmp / "teacher.pt"
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, teacher_path)
            cfg_path = tmp / "cfg.json"
            cfg_path.write_text(json.dumps(cfg.to_dict()))
            data = tmp / "data"
            make_corpus(data)

            cmd = [sys.executable, "-m", "blackcode.distill",
                   "--teacher", f"local:{teacher_path}",
                   "--config", str(cfg_path), "--data", str(data),
                   "--out", str(tmp / "run"), "--seq-len", "32",
                   "--batch-size", "2", "--grad-accum", "1", "--max-steps", "2",
                   "--optim", "adafactor", "--lr", "1e-3", "--logit-chunk", "16",
                   "--eval-interval", "0", "--save-interval", "1",
                   "--checkpoint-policy", "none", "--log-interval", "1",
                   "--mirror-dir", str(tmp / "drive_backup") ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertTrue((tmp / "run" / "ckpt.pt").exists())
            self.assertTrue((tmp / "drive_backup" / "ckpt.pt").exists())
            backup = torch.load(tmp / "drive_backup" / "ckpt.pt", map_location="cpu",
                                weights_only=False)
            self.assertEqual(backup["step"], 1)
            self.assertFalse(list((tmp / "drive_backup").glob("*.uploading")))
            log = (tmp / "run" / "distill_log.jsonl").read_text().strip().splitlines()
            self.assertGreaterEqual(len(log), 1)
            first = json.loads(log[0])
            self.assertIn("kd", first)
            self.assertIn("ce", first)


if __name__ == "__main__":
    unittest.main()
