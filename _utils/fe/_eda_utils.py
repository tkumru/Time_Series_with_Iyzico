from pyspark.sql import functions as F
from _utils.logging_utils import setup_logger

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


def seperate_columns(dtypes:list) -> tuple:
    logger = setup_logger("Seperate Columns to Variable Types")
    logger.debug("seperate_columns function executing...")
    
    columns = [_col for _col, _type in dtypes]
    
    categoric_columns = [_col for _col, _type in dtypes 
                             if _type in ['string']]
    numeric_columns = [_col for _col in set(columns) - set(categoric_columns)]
    
    assert len(columns) == len(categoric_columns) + len(numeric_columns), \
        "Total of numerical/categorical column have to equal total columns length"
    
    logger.info(f"\nNumerical Columns: {numeric_columns}")
    logger.info(f"\nCategorical Columns: {categoric_columns}")
    
    return (numeric_columns, categoric_columns)

###############################################################
# Columns Descriptions
###############################################################

def describe_numerical_columns(num_df, plot:bool=0):
    logger = setup_logger("Describing Numerical Columns")
    logger.debug("describe_numerical_columns function is executing...")
    
    num_df = num_df.toPandas()
    columns = num_df.columns
    
    return describe_numeric(num_df, columns, plot)

def describe_numeric(df: pd.DataFrame, numeric_columns: list,
                     plot: bool=0):
    """
    Function describes to numeric columns. It is for pandas dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_columns : list
        Numeric columns list.
    plot : bool, optional
       The default is 0.

    Returns
    -------
    Bar graph for each numerical columns and
    described columns.

    """
    logger = setup_logger("Describe Numeric Columns")
    
    logger.debug("describe_numeric executing...")
    
    if plot:
        for col in numeric_columns:
            df[col].hist(figsize=(15, 10))
            plt.title(col)
            plt.show()
        
    return df[numeric_columns].describe().T

def describe_categoric_columns(cat_df):
    logger = setup_logger("Describing Categorical Columns")
    logger.debug("describe_categoric_columns function is executing...")
    
    columns = cat_df.columns
    
    for col in columns:
        col = [col]
        logger.info(col)
        col_df = cat_df.select(col).toPandas()
        describe_categoric(col_df, col)

def describe_categoric(df: pd.DataFrame, columns: list):
    """
    Function describes to categoric columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Categoric columns list.

    Returns
    -------
    Pie chart.

    """
    logger = setup_logger("Describe Categoric Columns")
    
    logger.debug("describe_categoric executing...")
    
    for col in columns:
        series = df[col].value_counts()
        logger.info(f"\n{series}")
        
        explode = [0.02 for x in range(len(series))]
        series.plot(figsize=(15, 10), kind="pie",
                    autopct='%1.0f%%', shadow=True,
                    explode=explode)
        plt.show()
        
def describe_columns_to_target(df, columns:list, target:str):
    logger = setup_logger("Describe Columns for Target")
    logger.debug("describe_columns_to_target executing...")
    
    for col in columns:
        df.groupBy(col).agg(F.mean(target).alias("Mean"),
                              F.percentile_approx(target, 0.5).alias("Median"),
                              F.stddev(target).alias("Std")).sort('Mean').show(truncate=False)

def describe_for_target(df: pd.DataFrame, columns: list, target: str):
    """
    Function describes to numeric columns for target columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Numeric columns list.
    target : str
        Target column name.

    Returns
    -------
    Bar graph with median, variance and mean.

    """
    logger = setup_logger("Describe Numeric Columns for Target")
    
    logger.debug("describe_for_target executing...")
    
    for col in columns:
        agg = df.groupby(target).agg({col: ["mean", "median", "std"]})
        logger.info(f"\n{agg}")
        
        agg.plot(kind="bar", title=f"{col} Described for Target",
                 figsize=(15, 10), stacked=True)


####################################################
# Missing Values
####################################################

def check_missing_values(df, save:bool=0):
    logger = setup_logger("Missing Values Logging")
    logger.debug("check_missing_values executing...")
    
    def null_count(df, col):
        return df.select(col).filter((F.col(col) == "NA") |
                                (F.col(col) == "") |
                                (F.col(col).isNull()) | 
                                (F.col(col) == "None")).count()
    
    misses = {}
    for index, col in enumerate(df.columns):
        nc = null_count(df, col)
        count = df.count()
        
        if nc > 0:
            logger.info(f"{col} has {nc} - {(nc / count) * 100 : .2f} % null count.") 
            
            misses[index] = {"column_name": col,
                               "null_count": nc,
                               "ratio": (nc / count) * 100}
            
    if save:
        with open('missing_values.json', 'w') as file:
            json.dump(misses, file)
            
    return misses
            

def get_null_summary(df: pd.DataFrame, plot: bool=0):
    logger = setup_logger("Missing Values Logging")
    
    logger.debug("get_null_summary executing...")
    
    count = len(df) 
    null_df = df.isnull().sum()
    
    null_df = pd.concat([null_df, count - null_df], axis=1,
                        keys=["null", "not-null"])
    logger.info(f"\n{null_df}")
    
    if plot:
        for row in null_df.itertuples():
            plt.pie([row[1], row[2]], labels=["not-null", "null"],
                    autopct='%1.0f%%')
            plt.title(row.Index)
            plt.show()
            
    return null_df
    
####################################################
# Outliers
####################################################     

def check_outlier_values(df, columns: list,
                         q1_value: float=0.01,
                         q3_value: float=0.99,
                         save: bool=0):
    logger = setup_logger("Checking Outliers")
    logger.debug("check_outlier_values executing...")
    
    outliers = {}
    for index, col in enumerate(columns):
        array = df.select(col).toPandas().to_numpy()
        length = len(array)
        
        q1 = np.quantile(array, q1_value)
        q3 = np.quantile(array, q3_value)
        
        iqr = np.subtract(q3, q1)
        
        up_limit, low_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
        
        condition = np.where((array > up_limit) | (array < low_limit))
        outliers_length = len(array[condition])
        
        ratio = (outliers_length / length) * 100
        if ratio != 0: 
            logger.info(f"{col} has {outliers_length} - % {ratio : .2f} outliers count.")
            
            outliers[index] = {"column_name": col,
                                "outlier_count": outliers_length,
                                "up_limit": up_limit,
                                "low_limit": low_limit,
                                "ratio": ratio}
        
    if save:
        with open('outlier_values.json', 'w') as file:
            json.dump(outliers, file)
            
    return outliers

def check_outliers(df: pd.DataFrame, columns: list, 
                   plot: bool=0,
                   q1_value: float=0.25,
                   q3_value: float=0.75):
    logger = setup_logger("Get Correlation")
    
    logger.debug("check_outliers executing...")
    
    if plot:
        for col in columns:
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.boxplot(data=df, x=col, ax=ax)
            plt.show()
            
    logger.info("Dataframe Columns Outliers State:")
          
    for col in columns:
        array = df[col].to_numpy()
        
        q1 = np.quantile(array, q1_value)
        q3 = np.quantile(array, q3_value)
        
        iqr = np.subtract(q3, q1)
        
        up_limit, low_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
        
        result = True \
            if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None) \
            else False
            
        logger.info(f"{col} --> {result}")