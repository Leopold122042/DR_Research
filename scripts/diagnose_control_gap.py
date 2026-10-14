"""Offline, resumable attribution on frozen archived controller observations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr_research.diagnostic_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
