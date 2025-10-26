from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import os
import glob

from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    'credit_risk_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for credit risk prediction run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    # end_date=datetime(2023, 2, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 

    # # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py --snapshotdate "{{ ds }}" --modelname "credit_model_2024_09_01.pkl"'
        ),
    )
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    label_store_completed >> model_inference_start >> model_inference >> model_inference_completed


    # # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitor.py --snapshotdate "{{ ds }}" --modelname "credit_model_2024_09_01.pkl"'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    model_inference_completed >> model_monitor_start >> model_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = BashOperator(
        task_id='model_1_automl',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_train.py --snapshotdate "{{ ds }}"'
        ),
    )


    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    label_store_completed >> model_automl_start >> model_1_automl >> model_automl_completed