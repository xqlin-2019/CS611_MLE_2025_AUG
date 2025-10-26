import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python gold_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
    os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate

    print(f"Processing gold label store for snapshot: {date_str}")

    bronze_lms_directory = "datamart/bronze/lms/"
    silver_lms_directory = "datamart/silver/lms/"
    
    # create gold datalake
    gold_label_store_directory = "datamart/gold/"
    
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    # run data processing
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_lms_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

    # folder_path = gold_label_store_directory
    folder_path = 'datamart/gold/label_store/'
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_df = spark.read.option("header", "true").parquet(*files_list)
    print("[gold/label_store] row_count:",label_store_df.count())
    
    label_store_df.show()
    
    
    # folder_path = gold_feature_store_directory
    folder_path = 'datamart/gold/feature_store/'
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    feature_store_df = spark.read.option("header", "true").parquet(*files_list)
    print("[gold/feature_store] row_count:",feature_store_df.count())
    
    feature_store_df.show()
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
