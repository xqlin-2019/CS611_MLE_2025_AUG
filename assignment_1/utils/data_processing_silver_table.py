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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # filepath = os.path.join(bronze_lms_directory, name, filename)
    # partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    # filepath = bronze_lms_directory + partition_name

    datasets = {
        "loan_daily": f"bronze_loan_daily_{snapshot_date_str.replace('-','_')}.csv",
        "features_clickstream": f"bronze_features_clickstream_{snapshot_date_str.replace('-','_')}.csv",
        "features_attributes": f"bronze_features_attributes_{snapshot_date_str.replace('-','_')}.csv",
        "features_financials": f"bronze_features_financials_{snapshot_date_str.replace('-','_')}.csv"
    }

    results = {}
    
    # connect to bronze table
    for name, filename in datasets.items():
        filepath = os.path.join(bronze_lms_directory, name, filename)

        if not os.path.exists(filepath):
            print(f"Skipping {name}, no Bronze file for {snapshot_date}")
            continue
        
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())

        # if df.count() == 0:
        #     print(f"Skipping {name}, empty dataset for {snapshot_date}")
        #     continue
    
        # clean data: enforce schema / data type
        # Dictionary specifying columns and their desired datatypes
        if name == 'loan_daily':
            column_type_map = {
                "loan_id": StringType(),
                "Customer_ID": StringType(),
                "loan_start_date": DateType(),
                "tenure": IntegerType(),
                "installment_num": IntegerType(),
                "loan_amt": FloatType(),
                "due_amt": FloatType(),
                "paid_amt": FloatType(),
                "overdue_amt": FloatType(),
                "balance": FloatType(),
                "snapshot_date": DateType(),
            }
        
            for column, new_type in column_type_map.items():
                df = df.withColumn(column, col(column).cast(new_type))
        
            # augment data: add month on book
            df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
        
            # augment data: add days past due
            # df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
            df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()))
            df = df.fillna({"installments_missed": 0})
            df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
            df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

        
        elif name == "features_clickstream":
            # all fe_1..fe_20 numeric
            for i in range(1, 21):
                df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))
            df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
            df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

        
        elif name == "features_attributes":
            # clean Age, drop SSN (PII)
            df = df.withColumn("Age", col("Age").cast(IntegerType()))
            df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
            df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
            df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
            # drop SSN
            df = df.drop("SSN")

        
        elif name == "features_financials":
            # clean dirty numeric columns and convert types
            df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
            df = df.withColumn("Annual_Income",F.regexp_replace(col("Annual_Income").cast(StringType()), "[^0-9.]", "").cast(FloatType()))
            df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
            df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
            df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
            df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
            df = df.withColumn("Num_of_Loan",F.regexp_replace(col("Num_of_Loan").cast(StringType()), "[^0-9]", "").cast(IntegerType()))
            df = df.withColumn("Type_of_Loan", col("Type_of_Loan").cast(StringType()))
            df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
            df = df.withColumn("Num_of_Delayed_Payment",F.regexp_replace(col("Num_of_Delayed_Payment").cast(StringType()), "[^0-9]", "").cast(IntegerType()))
            df = df.withColumn("Changed_Credit_Limit",F.regexp_replace(col("Changed_Credit_Limit").cast(StringType()), "[^0-9.-]", "").cast(FloatType()))
            df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(FloatType()))
            df = df.withColumn("Credit_Mix", col("Credit_Mix").cast(StringType()))
            df = df.withColumn("Outstanding_Debt",F.regexp_replace(col("Outstanding_Debt").cast(StringType()), "[^0-9.-]", "").cast(FloatType()))
            df = df.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType()))
            df = df.withColumn("Credit_History_Age", col("Credit_History_Age").cast(StringType()))
            df = df.withColumn("Payment_of_Min_Amount", col("Payment_of_Min_Amount").cast(StringType()))
            df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))
            df = df.withColumn("Amount_invested_monthly",F.regexp_replace(col("Amount_invested_monthly").cast(StringType()), "[^0-9.-]", "").cast(FloatType()))
            df = df.withColumn("Payment_Behaviour", col("Payment_Behaviour").cast(StringType()))
            df = df.withColumn("Monthly_Balance",F.regexp_replace(col("Monthly_Balance").cast(StringType()), "[^0-9.-]", "").cast(FloatType()))
            df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
                        
            # parse credit history age (e.g. "10 Years and 9 Months")
            def parse_credit_age(x):
                if x is None: return None
                parts = x.split()
                years = int(parts[0]) if "Year" in x else 0
                months = int(parts[3]) if "Month" in x else 0
                return years * 12 + months

            parse_udf = F.udf(parse_credit_age, IntegerType())
            df = df.withColumn("Credit_History_Months", parse_udf(col("Credit_History_Age")))

    
        # save silver table - IRL connect to database to write
        dataset_dir = os.path.join(silver_lms_directory, name)
        os.makedirs(dataset_dir, exist_ok=True)
        outname = f"silver_{name}_{snapshot_date_str.replace('-', '_')}.parquet"
        outpath = os.path.join(dataset_dir, outname)

        
        df.write.mode("overwrite").parquet(outpath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        print('saved to:', outpath)

        results[name] = df
        
    return results