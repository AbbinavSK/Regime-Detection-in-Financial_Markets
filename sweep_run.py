import subprocess
import sys
import time

# (script, results_path) pairs run in order -- embedding_geometry_sweep.py runs last since it loads modelling_sweep_raw.py's frozen bundles.
SCRIPTS = [
    ("modelling_sweep_raw.py", "outputs/modelling_sweep_results_raw.csv"),
    ("eigencentrality_sweep_raw.py", "outputs/eigencentrality_sweep_results_raw.csv"),
    ("spectral_sweep_raw.py", "outputs/spectral_sweep_results_raw.csv"),
    ("embedding_geometry_sweep.py", "outputs/embedding_geometry_results.csv"),
]


# Run each sweep script as a subprocess in order, continuing past failures, and print a pass/fail summary.
def main(scripts=SCRIPTS):
    results = []
    for script, results_path in scripts:
        print(f"\n===== running {script} =====")
        start = time.time()
        proc = subprocess.run([sys.executable, script])
        elapsed = time.time() - start
        ok = proc.returncode == 0
        results.append((script, ok, elapsed, results_path))
        status = "OK" if ok else "FAILED"
        print(f"===== {script}: {status} ({elapsed / 60:.1f} min) =====")

    print("\n===== summary =====")
    for script, ok, elapsed, results_path in results:
        status = "OK" if ok else "FAILED"
        print(f"{script:32s} {status:6s} {elapsed / 60:6.1f} min  {results_path}")

    if not all(ok for _, ok, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()