from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def main() -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed in this Python environment. "
            "Run this file with C:\\Users\\BWY\\anaconda3\\envs\\Pump\\python.exe."
        ) from exc

    arguments = sys.argv[1:]
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_DIR / "app.py"),
        "--global.developmentMode",
        "false",
        *arguments,
    ]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

