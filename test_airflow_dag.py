"""
Local standalone verification script for the Airflow DAG logic.
Verifies all functions, data flow, branching, and deployment triggers.
"""
import os
import sys

# Add dags folder to Python path
sys.path.insert(0, os.path.abspath("dags"))

from sentiment_mlops_dag import (
    download_data,
    preprocess_data,
    train_model,
    evaluate_model,
    check_accuracy_improvement,
    trigger_deployment,
    skip_deployment,
    ACCURACY_DEPLOYMENT_THRESHOLD
)

class MockTaskInstance:
    """Simulates Airflow TaskInstance (XCom push & pull)."""
    def __init__(self):
        self.xcom_data = {}

    def xcom_push(self, key, value):
        self.xcom_data[key] = value

    def xcom_pull(self, task_ids, key):
        return self.xcom_data.get(key)

def main():
    print("=" * 70)
    print("     TESTING APACHE AIRFLOW DAG PIPELINE EXECUTION LOCALLY     ")
    print("=" * 70)

    ti = MockTaskInstance()

    # Step 1: Download
    print("\n[TASK 1/5] Executing: download_data")
    path = download_data()
    assert os.path.exists(path), "download_data failed!"

    # Step 2: Preprocess
    print("\n[TASK 2/5] Executing: preprocess_data")
    preprocess_data()
    assert os.path.exists("data/train.json") and os.path.exists("data/test.json")

    # Step 3: Train
    print("\n[TASK 3/5] Executing: train_model")
    train_model()

    # Step 4: Evaluate
    print("\n[TASK 4/5] Executing: evaluate_model")
    acc = evaluate_model(ti=ti)
    assert acc is not None

    # Step 5: Branching & Conditional Deployment
    print("\n[TASK 5/5] Executing: check_accuracy_improvement (BranchOperator)")
    next_task = check_accuracy_improvement(ti=ti)
    print(f"[*] Airflow Branch Decision: -> '{next_task}'")

    if next_task == 'trigger_deployment':
        trigger_deployment()
    else:
        skip_deployment()

    print("\n" + "=" * 70)
    print("  [SUCCESS] AIRFLOW DAG LOGIC VERIFIED WITH ZERO ERRORS!  ")
    print("  Ready to display in Airflow Web UI Graph View.        ")
    print("=" * 70)

if __name__ == "__main__":
    main()
