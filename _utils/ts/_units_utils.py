from _utils.logging_utils import setup_logger
from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class ControlUnits:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def check_stationary(self):
        logger = setup_logger()
        logger.debug("is_stationary function is executing...")
        
        logger.info("H0: Non-Stationary Hypothesis")
        logger.info("H1: Stationary Hypothesis")
        
        p_value = sm.tsa.stattools.adfuller(self.df)[1]
        logger.info(f"Result: Stationary, p-value: {p_value: .4f}") if p_value <= 0.5 \
            else logger.info(f"Result: Non-Stationary, p-value: {p_value: .4f}")
            
    def check_other_terms(self, period: int, plot: bool=1):
        logger = setup_logger()
        logger.debug("check_other_terms function is executing...")
        
        stl = STL(self.df, period=period)
        result = stl.fit()
        
        trend, seasonal, residual = result.trend, result.seasonal, result.resid
        trend_mean, seasonal_mean, residual_mean = (np.mean(trend), 
                                                    np.mean(seasonal),
                                                    np.mean(residual))
        
        if plot:
            fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            
            ax[0].set_title(f"Data Understanding - Period: {period}")
            ax[0].plot(self.df, 'k', label="Original Time Series Data")
            ax[0].legend(loc="upper left")
            
            ax[1].plot(trend, label=f'Trend Component - Mean: {trend_mean: .4f}')
            ax[1].legend(loc='upper left')
            
            ax[2].plot(seasonal, 'g', label=f'Seasonal Component - Mean: {seasonal_mean: .4f}')
            ax[2].legend(loc='upper left')
            
            ax[3].plot(residual, 'r', label=f'Residuals - Mean: {residual_mean: .4f}')
            ax[3].legend(loc='upper left')
            
        
        logger.info(f"Trend Value: {trend_mean: .4f}")
        logger.info(f"Seasonality Value: {seasonal_mean: .4f}")
        logger.info(f"Residual Value: {residual_mean: .4f}")
        
        return (trend, seasonal, residual)
