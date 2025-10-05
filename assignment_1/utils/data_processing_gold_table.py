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
from pyspark.sql.window import Window


def process_labels_gold_table(snapshot_date_str, silver_lms_directory, gold_label_store_directory, spark, dpd, mob):

    def read_silver_snapshots_upto(snapshot_date_str,silver_lms_directory, feature_name, spark):
        """
        Reads all parquet files for a given Silver feature (across all snapshots).
        Returns one unified Spark DataFrame.
        """
        feature_dir = os.path.join(silver_lms_directory, feature_name)
        all_files = glob.glob(os.path.join(feature_dir, "*.parquet"))
        
        if not all_files:
            print(f"No Silver files found for {feature_name}")
            return None

        # Parse snapshot_date from filenames
        snapshot_cutoff = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        eligible_files = []
        for f in all_files:
            try:
                date_str = "_".join(f.split("_")[-3:]).replace(".parquet", "")
                file_date = datetime.strptime(date_str, "%Y_%m_%d")
                if file_date <= snapshot_cutoff:
                    eligible_files.append(f)
            except Exception:
                continue
    
        if not eligible_files:
            print(f"No eligible Silver files for {feature_name} before {snapshot_date_str}")
            return None
    
        df = spark.read.parquet(*eligible_files)
        print(f"Loaded {feature_name}: {len(eligible_files)} snapshots (≤ {snapshot_date_str}), {df.count()} total rows")
        return df

    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    # partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    # filepath = silver_loan_daily_directory + partition_name
    loan_daily_path = os.path.join(silver_lms_directory, "loan_daily", f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet")
    if not os.path.exists(loan_daily_path):
        print(f"Missing Silver dataset: {loan_daily_path}")
    
    # df = spark.read.parquet(filepath)
    # print('loaded from:', filepath, 'row count:', df.count())
    df_loan = spark.read.parquet(loan_daily_path)
    df_click = read_silver_snapshots_upto(snapshot_date_str, silver_lms_directory, "features_clickstream", spark)
    df_attr  = read_silver_snapshots_upto(snapshot_date_str, silver_lms_directory, "features_attributes", spark)
    df_fin   = read_silver_snapshots_upto(snapshot_date_str, silver_lms_directory, "features_financials", spark) 
    
    print(f"Loaded Silver datasets for {snapshot_date_str}:")
    print(f"  Loan Daily: {df_loan.count()} rows")
    print(f"  Clickstream: {df_click.count()} rows")
    print(f"  Attributes: {df_attr.count()} rows")
    print(f"  Financials: {df_fin.count()} rows")

    # ---------------------------------------------------------
    # PART 1: LABEL STORE CREATION
    # ---------------------------------------------------------

    # get customer at mob
    # df = df.filter(col("mob") == mob)
    df_label = df_loan.filter(col("mob") == mob)

    # get label
    # df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    # df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))
    df_label = df_label.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df_label = df_label.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    
    # select columns to save
    # df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    df_label = df_label.select("Customer_ID", "label", "label_def", "snapshot_date")

    # save gold label store - IRL connect to database to write
    label_store_dir = os.path.join(gold_label_store_directory, "label_store")
    os.makedirs(label_store_dir, exist_ok=True)
    label_outpath = os.path.join(label_store_dir, f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet")

    df_label.write.mode("overwrite").parquet(label_outpath)
    print(f"Label Store saved to: {label_outpath}")

    
    # # ---------------------------------------------------------
    # # PART 2: FEATURE STORE CREATION
    # # ---------------------------------------------------------

    # # Join clickstream features by Customer_ID + snapshot_date
    # df_gold = df_label.join(df_click, ["Customer_ID", "snapshot_date"], "left")

    # # Join static features (Attributes, Financials)
    # df_gold = df_gold.join(df_attr.dropDuplicates(["Customer_ID"]), on="Customer_ID", how="left")
    # df_gold = df_gold.join(df_fin.dropDuplicates(["Customer_ID"]), on="Customer_ID", how="left")

    # # save gold table - IRL connect to database to write
    # partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    # filepath = gold_label_store_directory + partition_name
    # df.write.mode("overwrite").parquet(filepath)
    # # df.toPandas().to_parquet(filepath,
    # #           compression='gzip')
    # print('saved to:', filepath)

    df_base = df_loan.filter(col("mob") == 0).select("loan_id", "Customer_ID", "loan_start_date", "snapshot_date")

    # Helper for temporal join (latest snapshot_date ≤ loan_start_date)
    def temporal_join(df_feat, base_df, join_name):
        """
        Temporal join that matches each loan (base_df) with the latest snapshot
        of features (df_feat) where snapshot_date <= loan_start_date.
        """
        if df_feat is None:
            print(f"Skipping {join_name}: no data found")
            return base_df
    
        # Ensure both have proper date types
        df_feat = (
            df_feat
            .withColumnRenamed("snapshot_date", f"{join_name}_snapshot_date")
            .withColumn(f"{join_name}_snapshot_date", F.to_date(f"{join_name}_snapshot_date"))
        )
        base_df = base_df.withColumn("loan_start_date", F.to_date("loan_start_date"))
    
        # Join first by Customer_ID
        joined = base_df.join(df_feat, on="Customer_ID", how="left")
    
        # Filter only records before or on loan_start_date
        joined = joined.filter(F.col(f"{join_name}_snapshot_date") <= F.col("loan_start_date"))
    
        # Rank snapshots by recency per loan
        window_spec = Window.partitionBy("loan_id").orderBy(F.col(f"{join_name}_snapshot_date").desc())
        ranked = joined.withColumn("rank", F.row_number().over(window_spec))
    
        # Step Keep only the most recent valid snapshot per loan
        latest = ranked.filter(F.col("rank") == 1).drop("rank")
    
        print(f"Joined {join_name}: picked latest snapshot ≤ loan_start_date")
        return latest


    # Join all feature sets
    df_feature = df_base
    df_feature = temporal_join(df_attr, df_feature, "attributes")
    df_feature = temporal_join(df_fin, df_feature, "financials")
    df_feature = temporal_join(df_click, df_feature, "clickstream")

    # Save feature store
    feature_store_dir = os.path.join(gold_label_store_directory, "feature_store")
    os.makedirs(feature_store_dir, exist_ok=True)
    feature_outpath = os.path.join(feature_store_dir, f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet")

    df_feature.write.mode("overwrite").parquet(feature_outpath)
    print(f"Feature Store saved to: {feature_outpath}")

    print("\nGold processing completed successfully.\n")

    return {
        "label_store": df_label,
        "feature_store": df_feature
    }