"""Easy repository entry point; dependency errors remain readable on a fresh install."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    if sys.version_info < (3, 10):
        print("Vitalis needs Python 3.10 or newer.", file=sys.stderr)
        return 2
    missing = [name for name in ("torch", "transformers", "flask", "numpy")
               if importlib.util.find_spec(name) is None]
    if missing:
        requirements = Path(__file__).resolve().with_name("requirements.txt")
        command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        print("Install once, then run this launcher again:", file=sys.stderr)
        if sys.platform == "win32":
            print(subprocess.list2cmdline(command), file=sys.stderr)
        else:
            import shlex
            print(shlex.join(command), file=sys.stderr)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from compare_server import main as launch
    return launch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
