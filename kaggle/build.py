"""Đóng gói kaggle/pipeline.py + package features/ thành 1 file kernel (Kaggle script chỉ nhận 1 file).

    python kaggle/build.py
    kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4
    kaggle kernels output nguyendangthuc11/camera-health-acdc-rain -p outputs/kaggle
"""
import base64
import io
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "kernel"
PACKAGES = ["features"]

HEADER = '''"""AUTO-GENERATED bởi kaggle/build.py — sửa kaggle/pipeline.py hoặc features/, đừng sửa file này."""
import base64, io, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(_SRC))).extractall("/kaggle/working/src")
sys.path.insert(0, "/kaggle/working/src")
'''


def main():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pkg in PACKAGES:
            for f in (ROOT / pkg).rglob("*.py"):
                z.write(f, f.relative_to(ROOT).as_posix())
    src = base64.b64encode(buf.getvalue()).decode()
    KERNEL.mkdir(exist_ok=True)
    body = (ROOT / "kaggle" / "pipeline.py").read_text(encoding="utf-8")
    (KERNEL / "run.py").write_text(f'_SRC = "{src}"\n' + HEADER + body, encoding="utf-8")
    shutil.copy(ROOT / "kaggle" / "kernel-metadata.json", KERNEL / "kernel-metadata.json")
    print(f"built {KERNEL / 'run.py'} ({len(src) // 1024} KB embedded source)")


if __name__ == "__main__":
    main()
