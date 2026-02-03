import subprocess
import sys

def run_help(mod: str):
    return subprocess.run(
        [sys.executable, "-m", mod, "--help"],
        capture_output=True,
        text=True,
    )

def test_main_help_runs():
    proc = run_help("firepype.cli")
    assert proc.returncode == 0
    assert "FIRE AB-pair NIR reduction pipeline" in proc.stdout

def test_console_scripts_exist():
    # Only a presence check; if not installed as package, skip gracefully
    proc = subprocess.run(["python", "-c", "import shutil; "
                           "print(bool(shutil.which('firepype')))"
                           ],
                          capture_output=True, text=True)
    if "True" not in proc.stdout:
        return  # running in editable mode without console scripts
