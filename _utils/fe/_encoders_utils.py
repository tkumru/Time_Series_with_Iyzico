from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder

from _utils.logging_utils import setup_logger

##################################################
# Rare Encoding
##################################################

def rare_analyser(df, target: str, categoric_cols: list):
    logger = setup_logger()
    logger.debug("rare_analyser function is executing...")
    
    length = df.count()
    for col in categoric_cols:
        df.groupBy(col).agg(F.count(col).alias('Count'),
                          F.round(F.count(col) / length, 2).alias('Ratio'),
                          F.round(F.avg(target), 2).alias('Target Mean')).show()
        
def rare_encoder(df, columns: list, rare_percent: float):
    logger = setup_logger()
    logger.debug("rare_analyser function is executing...")
    
    length = df.count()
    for col in columns:
        logger.info(f"{col} column is encoding...")
        grouped_df = df.groupBy(col) \
            .agg(F.round(F.count(col) / length, 2).alias('Ratio'))
        filtered_df = grouped_df.filter(F.col('Ratio') <= rare_percent)
        
        rare_columns = filtered_df.select(col).rdd \
            .flatMap(lambda column: column).collect()
            
        df = df.withColumn(col,
                           F.when(F.col(col).isin(rare_columns), 'Rare').otherwise(
                           F.col(col)))
        
    return df

##################################################
# Label Encoding
##################################################

def label_encoder(df, binary_cols: list):
    logger = setup_logger()
    logger.debug("label_encoder function is executing...")
    
    encoder = LabelEncoder()
    for col in binary_cols:
        logger.info(f"{col} column is encoding...")
        df[col] = encoder.fit_transform(df[col])
    
    return df
