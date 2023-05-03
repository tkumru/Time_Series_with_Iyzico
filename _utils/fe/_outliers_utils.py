from pyspark.sql import functions as F
from sklearn.neighbors import LocalOutlierFactor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##################################################
# Replace with Thresholds
##################################################

def replace_with_thresholds(df, col: str,
                            up_threshold: float,
                            low_threshold: float):
    df = df.withColumn(col, F.when(F.col(col) < low_threshold, low_threshold)
                       .otherwise(F.col(col)))
    df = df.withColumn(col, F.when(F.col(col) > up_threshold, up_threshold)
                       .otherwise(F.col(col)))
    
    return df

##################################################
# Local Outliers
##################################################

class LocalOutliers:
    def __init__(self, df, numerical_cols: list,
                 df_type: str='spark'):
        self.df = df  
        self.numerical_cols = numerical_cols
        
        if df_type == 'spark': self.df = df.toPandas()
        
    def init_model(self):
        neigh_count = int(self.df.shape[0] * (1 / 100))
        
        clf = LocalOutlierFactor(n_neighbors=neigh_count)
        clf.fit_predict(self.df[self.numerical_cols])
        
        self.scores = clf.negative_outlier_factor_
        
    def plot(self, xlim: list=[0, 50], figsize: tuple=(15, 10)):
        plot_df = pd.DataFrame(np.sort(self.scores))
        plot_df.plot(stacked=True,
                     xlim=xlim,
                     style='.-',
                     figsize=figsize)
        
        plt.show()
        
    def get_local_outliers_df(self, threshold: float):
        threshold = np.sort(self.scores)[threshold]
        
        local_outlier_df = self.df[(self.scores < threshold)]
        non_local_outlier_df = self.df[~(self.scores < threshold)]
        
        return local_outlier_df, non_local_outlier_df
