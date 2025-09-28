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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    # csv_file_path = "data/lms_loan_daily.csv"
    datasets = {
        "loan_daily": "data/lms_loan_daily.csv",
        "features_clickstream": "data/feature_clickstream.csv",
        "features_attributes": "data/features_attributes.csv",
        "features_financials": "data/features_financials.csv"
    }

    results = {} 

    for name, path in datasets.items():
        
        # load data - IRL ingest from back end source system
        df = spark.read.csv(path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(name + '_' + snapshot_date_str + ' row count:', df.count())

        dataset_dir = os.path.join(bronze_lms_directory, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        partition_name = f"bronze_{name}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(dataset_dir, partition_name) 
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)
    
        results[name] = df
        
    return results['loan_daily']
