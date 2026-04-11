from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_DIR = ROOT_DIR / "pipelines" / "models"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["mlp.py", "attention.py", "full_attention.py", "bi_full_attention.py"]

BOOTSTRAP = """from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

"""


def main():
    for filename in FILES:
        src = ROOT_DIR / filename
        dst = TARGET_DIR / filename
        if not src.exists():
            print(f"Skip missing: {src}")
            continue

        content = src.read_text(encoding="utf-8")
        if "ROOT_DIR = Path(__file__).resolve().parents[2]" not in content:
            content = BOOTSTRAP + content
        dst.write_text(content, encoding="utf-8")
        src.unlink()
        print(f"Migrated: {filename} -> {dst}")


if __name__ == "__main__":
    main()
