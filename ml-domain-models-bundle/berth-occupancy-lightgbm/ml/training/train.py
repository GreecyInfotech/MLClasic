from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    train_all = ROOT / "ml" / "training" / "train_all.py"
    subprocess.run([sys.executable, str(train_all)], check=True)


if __name__ == "__main__":
    main()
