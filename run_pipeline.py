import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

scripts = [
    BASE_DIR / "collecting" / "collect_data.py",
    BASE_DIR / "collecting" / "build_features.py",
    BASE_DIR / "forecast_update.py",
]

for script in scripts:
    print(f"----> Running {script.name}...")
    subprocess.run([sys.executable, str(script)], check=True)

print("-------------> Pipeline finished")