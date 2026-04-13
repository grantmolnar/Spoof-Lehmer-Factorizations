"""YAFU subprocess backend for factoring large integers (n > 10^30).

Wraps the `yafu` command-line tool. Install yafu separately:
    https://github.com/bbuhrow/yafu

The backend writes n to a temp file, runs `yafu factor(@)`, and parses
the output. Falls back gracefully (raises RuntimeError) if yafu isn't
on PATH or fails on a particular input.

Performance notes:
- For n < 10^30, sympy is faster (no subprocess overhead).
- For 10^30 < n < 10^60, yafu using SIQS is the right choice.
- For n > 10^60 with no small factors, you want CADO-NFS instead.
"""
from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class YafuBackend:
    """Factor integers via the yafu CLI."""
    name = "yafu"
    max_n = 10**60

    def __init__(self, yafu_path: str | None = None, timeout: int = 600):
        resolved = yafu_path or shutil.which("yafu")
        if resolved is None:
            raise RuntimeError(
                "yafu not found on PATH. Install from "
                "https://github.com/bbuhrow/yafu or pass yafu_path explicitly."
            )
        self.yafu_path: str = resolved
        self.timeout = timeout

    def factor(self, n: int) -> dict[int, int]:
        if n <= 1:
            return {}

        # YAFU's output format: lines like "P5 = 12345" or "C20 = ..." for
        # composites, "PRP18 = ..." for probable primes. We parse all "P/PRP"
        # lines as confirmed prime factors.
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.txt"
            input_file.write_text(str(n))

            try:
                result = subprocess.run(
                    [self.yafu_path, "factor(@)", "-batchfile", str(input_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"yafu timed out factoring {n}") from e

            if result.returncode != 0:
                raise RuntimeError(
                    f"yafu failed (exit {result.returncode}): {result.stderr[:500]}"
                )

            return self._parse_output(result.stdout, n)

    def _parse_output(self, output: str, n: int) -> dict[int, int]:
        """Parse yafu output and verify the factorization multiplies to n."""
        # Match lines like "P5 = 12345" or "PRP18 = 123456789012345678"
        pattern = re.compile(r"P(?:RP)?\d+\s*=\s*(\d+)")
        primes = [int(m.group(1)) for m in pattern.finditer(output)]

        if not primes:
            raise RuntimeError(
                f"yafu output had no prime factors:\n{output[:1000]}"
            )

        result: dict[int, int] = {}
        product = 1
        for p in primes:
            result[p] = result.get(p, 0) + 1
            product *= p

        if product != n:
            raise RuntimeError(
                f"yafu factorization mismatch: product = {product}, expected {n}. "
                f"Output:\n{output[:1000]}"
            )
        return result
