import subprocess
import sys
import os

# Folder containing the scripts
folder = os.path.dirname(os.path.abspath(__file__))

# Run sequence
scripts = ["Milestone1_DataCleaning.py","Milestone3_TrainingLoop.py","Milestone4&5_Optimization_Evaluation.py.py"]

for script in scripts:
    path = os.path.join(folder, script)

    print(f"\nRunning {script}...")

    try:
        subprocess.run(
            [sys.executable, path],
            check=True
        )

        print(f"{script} completed successfully")

    except FileNotFoundError:
        print(f"Error: {script} not found")
        break

    except subprocess.CalledProcessError:
        print(f"Error: {script} failed")
        break

print("\nAIO process finished")
