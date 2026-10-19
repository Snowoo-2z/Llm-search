import unittest

from blackcode.eval_code import build_test_program, pass_at_k, run_program, truncate_at_stop


class TestEvalHarness(unittest.TestCase):
    def test_run_program_success_and_failure(self):
        ok, _ = run_program("print('hello')", timeout=2)
        self.assertTrue(ok)
        ok, err = run_program("raise ValueError('boom')", timeout=2)
        self.assertFalse(ok)
        self.assertIn("ValueError", err)

    def test_timeout_is_caught(self):
        ok, err = run_program("while True:\n    pass", timeout=0.5)
        self.assertFalse(ok)
        self.assertEqual(err, "timeout")

    def test_correct_completion_passes_tests(self):
        prompt = "def add(a, b):\n"
        completion = "    return a + b\n"
        program = build_test_program(prompt, completion,
                                     "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                                     "add")
        ok, err = run_program(program, timeout=2)
        self.assertTrue(ok, err)

    def test_pass_at_k_extremes(self):
        self.assertAlmostEqual(pass_at_k(10, 10, 1), 1.0)
        self.assertAlmostEqual(pass_at_k(10, 0, 1), 0.0)
        self.assertAlmostEqual(pass_at_k(10, 5, 1), 0.5)
        self.assertGreater(pass_at_k(10, 1, 10), 0.99)

    def test_stop_sequence(self):
        text = "    return x\n\ndef other():\n    pass"
        self.assertEqual(truncate_at_stop(text), "    return x")


if __name__ == "__main__":
    unittest.main()
