# src/main.py

import subprocess
import sys


def run_step(step_name, command):
    print("\n" + "=" * 60)
    print(f"RUNNING: {step_name}")
    print("=" * 60)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error while running: {step_name}")
        sys.exit(1)

    print(f"\n✅ Completed: {step_name}")


def main():
    run_step("Preprocessing", "python preprocess.py")
    run_step("Feature Extraction", "python features.py")

    run_step(
        "Train Classifier (Logistic Regression)",
        "python train_classifier_logreg.py"
    )

    run_step(
        "Train Regressor (Linear SVR)",
        "python train_regressor_svr.py"
    )

    run_step("Final Evaluation", "python evaluate.py")


if __name__ == "__main__":
    main()
