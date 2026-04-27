import subprocess
import sys
import unittest


class CliEntrypointTests(unittest.TestCase):
    def test_python_module_entrypoint_propagates_cli_exit_code(self):
        result = subprocess.run(
            [sys.executable, "-m", "uiir", "fidelity", "/definitely/missing/uiir-output"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("uiir fidelity failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
