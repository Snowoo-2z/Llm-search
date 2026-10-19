"""Tests offline du runtime agentique : aucun modèle ou réseau requis."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blackcode.agent import (
    FrenchAgent,
    NoteStore,
    ProjectReader,
    ToolContext,
    calculate,
)


class FakeClient:
    def __init__(self, messages):
        self.responses = list(messages)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        if not self.responses:
            raise AssertionError("Réponse simulée manquante")
        return self.responses.pop(0)


def tool_call(name, arguments, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate("(26 + 14) * 3"), 120)
        self.assertEqual(calculate("-3 ** 2"), -9)

    def test_refuses_python_execution(self):
        for expression in ("__import__('os').getcwd()", "2 + True", "[1, 2][0]"):
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError)):
                calculate(expression)

    def test_limits_large_exponent(self):
        with self.assertRaisesRegex(ValueError, "Exposant"):
            calculate("2 ** 101")

    def test_division_by_zero_is_safe_error(self):
        with self.assertRaisesRegex(ValueError, "division par zéro"):
            calculate("1 / 0")


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "answer.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        (self.root / ".env").write_text("SECRET=value\n", encoding="utf-8")
        self.context = ToolContext(NoteStore(self.root / "notes"), ProjectReader(self.root))

    def tearDown(self):
        self.temp.cleanup()

    def test_note_round_trip_and_search(self):
        self.context.notes.save("Projet", "Créer un assistant français.")
        self.assertEqual(len(self.context.notes.list("français")), 1)
        self.context.notes.save("Projet", "Mettre à jour la note.")
        self.assertEqual(len(self.context.notes.list()), 1)
        self.assertEqual(self.context.notes.list()[0]["body"], "Mettre à jour la note.")

    def test_project_reader_is_read_only_and_confined(self):
        self.assertEqual(self.context.project.read_file("src/answer.py")["path"], "src/answer.py")
        for path in ("../outside.txt", str(self.root / "src" / "answer.py"), ".env", ".git/config"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.context.project.read_file(path)

    def test_list_project_files(self):
        self.assertEqual(self.context.project.list_files("src/*.py"), ["src/answer.py"])


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.context = ToolContext(NoteStore(root / "notes"), ProjectReader(root))

    def tearDown(self):
        self.temp.cleanup()

    def test_tool_call_then_final_answer(self):
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("calculator", {"expression": "19 * 23"})]},
            {"role": "assistant", "content": "19 fois 23 font 437."},
        ])
        agent = FrenchAgent(client, self.context)
        result = agent.ask("Calcule 19 fois 23.")
        self.assertEqual(result.text, "19 fois 23 font 437.")
        self.assertEqual([entry["name"] for entry in result.tool_trace], ["calculator"])
        self.assertIn('"result": 437', result.tool_trace[0]["result"])
        self.assertEqual(len(client.calls), 2)

    def test_multistep_tool_calls_execute_in_order(self):
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [
                tool_call("calculator", {"expression": "19 * 23"}, "c1"),
                tool_call("save_note", {"title": "Résultat", "body": "437"}, "c2"),
            ]},
            {"role": "assistant", "content": "J'ai calculé 437 et sauvegardé la note."},
        ])
        result = FrenchAgent(client, self.context).ask("Calcule et enregistre le résultat.")
        self.assertEqual([entry["name"] for entry in result.tool_trace], ["calculator", "save_note"])
        self.assertEqual(self.context.notes.list()[0]["body"], "437")

    def test_unknown_tool_is_not_executed(self):
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("run_shell", {"command": "id"})]},
            {"role": "assistant", "content": "Je n'ai pas exécuté de commande."},
        ])
        result = FrenchAgent(client, self.context).ask("Exécute une commande.")
        self.assertIn("non autorisé", result.tool_trace[0]["result"])

    def test_stops_after_step_limit(self):
        response = {"role": "assistant", "content": None, "tool_calls": [tool_call("calculator", {"expression": "1+1"})]}
        client = FakeClient([response, response])
        result = FrenchAgent(client, self.context, max_steps=1).ask("Boucle.")
        self.assertIn("limite d'étapes", result.text)
        self.assertEqual(len(client.calls), 2)

    def test_tool_call_limit_is_enforced(self):
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [
                tool_call("calculator", {"expression": "1+1"}, "c1"),
                tool_call("calculator", {"expression": "2+2"}, "c2"),
            ]},
            {"role": "assistant", "content": "Terminé."},
        ])
        result = FrenchAgent(client, self.context, max_tool_calls=1).ask("Deux calculs.")
        self.assertIn('"result": 2', result.tool_trace[0]["result"])
        self.assertIn("Limite d'appels", result.tool_trace[1]["result"])

    def test_history_keeps_only_complete_recent_turns(self):
        client = FakeClient([{"role": "assistant", "content": f"Réponse {i}"} for i in range(7)])
        agent = FrenchAgent(client, self.context)
        for i in range(7):
            agent.ask(f"Question {i}")
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertEqual(sum(m["role"] == "user" for m in agent.messages), 6)
        self.assertTrue(all(m["role"] in {"system", "user", "assistant", "tool"} for m in agent.messages))

    def test_test_runner_is_opt_in(self):
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("run_project_tests", {})]},
            {"role": "assistant", "content": "Non activé."},
        ])
        result = FrenchAgent(client, self.context).ask("Lance les tests.")
        self.assertIn("non autorisé", result.tool_trace[0]["result"])

    def test_enabled_test_runner_returns_verification_result(self):
        test_dir = Path(self.temp.name) / "tests"
        test_dir.mkdir()
        (test_dir / "test_smoke.py").write_text(
            "import unittest\nclass Smoke(unittest.TestCase):\n    def test_ok(self): self.assertEqual(2 + 2, 4)\n",
            encoding="utf-8",
        )
        enabled = ToolContext(self.context.notes, self.context.project, allow_test_execution=True)
        client = FakeClient([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("run_project_tests", {})]},
            {"role": "assistant", "content": "Les tests passent."},
        ])
        result = FrenchAgent(client, enabled).ask("Lance les tests.")
        report = json.loads(result.tool_trace[0]["result"])
        self.assertTrue(report["passed"], report["output"])
        self.assertEqual(report["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
