from pyspark.sql import functions as F
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from _utils.logging_utils import setup_logger

def configurate_missings(df):
    logger = setup_logger()
    logger.debug("configurate_missings function is executing...")
    
    columns = df.columns
    for col in columns:
        logger.info(f"{col} is configurating...")
        df = df.withColumn(col,
                           F.when(F.col(col) == "", None).otherwise(
                           F.when(F.col(col) == "NA", None).otherwise(
                           F.when(F.col(col).isNull(), None).otherwise(
                           F.when(F.col(col) == "None", None).otherwise(
                           F.col(col))))))
    
    return df

########################################################
# Delete
########################################################

def remove_missing_values(df, columns: list):
    logger = setup_logger()
    logger.debug("remove_missing_values function is executing...")
    
    return df.dropna(how="any", subset=columns)

########################################################
# FILLING PART
########################################################

########################################################
# Fill Categorical Columns with Mode
########################################################

def fill_cat_with_mode(df, columns: list):
    logger = setup_logger()
    logger.debug("fill_cat_with_mode function is executing...")
    
    for col in columns:
        mode = df.select(col).toPandas()[col].mode()[0]
        logger.info(f"{col} column's missing values are filled by {mode}")
        
        df = df.na.fill(value=mode, subset=[col])
        
    return df

########################################################
# Fill Numerical Columns with Mode
########################################################

def fill_num_with_mode(df, columns: list):
    logger = setup_logger()
    logger.debug("fill_num_with_mode function is executing...")
    
    for col in columns:
        mode = df.select(F.percentile_approx(col, 0.5)).first()[0]
        logger.info(f"{col} column's missing values are filled by {mode}")
        
        df = df.na.fill(value=mode, subset=[col])
        
    return df

########################################################
# KNN Imputer
########################################################

class KNNImputerSklearn:
    def __init__(self, df: pd.DataFrame, n: int=5):
        self.df = pd.get_dummies(df, drop_first=True)
        self.n = n
        
    def impute(self) -> pd.DataFrame:
        scale = MinMaxScaler()
        impute = KNNImputer(n_neighbors=self.n)
        
        logger = setup_logger()
        logger.debug("impute function is executing...")
        
        logger.info("Dataframe is scaling...")
        scaler = pd.DataFrame(scale.fit_transform(self.df),
                              columns=self.df.columns)
        
        logger.info("Scaled Dataframe is imputing...")
        imputer = pd.DataFrame(impute.fit_transform(scaler),
                               columns=scaler.columns)
        
        logger.info("Imputed Dataframe is inversing...")
        return pd.DataFrame(scale.inverse_transform(imputer),
                            columns=imputer.columns)
        
