from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.theta import FourTheta as Theta
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
from _utils.ts._dart._dart_utils import Smoother
from statsmodels.tsa.seasonal import seasonal_decompose
from _utils.logging_utils import setup_logger
from pmdarima.arima import ndiffs, PPTest
from darts.metrics import mae

import numpy as np
import optuna

class KalmanDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8, kf: int=None):
        Smoother.__init__(self, dart_series, percentage)
        
        self.kf = kf
        
    def smooth(self, dim_x: int):
        logger = setup_logger()
        logger.debug("KalmanForecaster is executing...")
        
        model = KalmanForecaster(dim_x=dim_x)
        predictions, accuracy = self.evaluation_model(model)
        
        return model, predictions, accuracy
    
    def optimize_optuna(self, epoch: int=100, x_dim_epoch: int=100):
        logger = setup_logger()
        logger.debug("Kalman is optimizing...")
        
        def objective(trial):
            dim_x = trial.suggest_int('dim_x', 1, x_dim_epoch, log=True)
            
            kalman = KalmanForecaster(dim_x=dim_x)
            
            result = kalman.fit(self.train)
            predictions = kalman.predict(len(self.val))
            
            mae_error = mae(self.val, predictions)
            
            return mae_error
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=epoch, n_jobs=-1, show_progress_bar=True)
        
        return study

class ThetaDart(Smoother):
    def __init__(self, dart_series, theta: float=None, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
        self.theta = theta
        
    def smooth(self, theta: float=2,
               season_mode: SeasonalityMode = SeasonalityMode.MULTIPLICATIVE,
               model_mode: ModelMode = ModelMode.ADDITIVE,
               trend_mode: TrendMode = TrendMode.LINEAR,
               seasonality_period: int=None, plot: bool=0):
        logger = setup_logger()
        logger.debug("Theta is executing...")
        
        model = Theta(theta=theta, trend_mode=trend_mode,
                    season_mode=season_mode,
                    model_mode=model_mode,
                    seasonality_period=seasonality_period)
        predictions, accuracy = self.evaluation_model(model, plot=plot)
        
        return model, predictions, accuracy
    
    def forecasting_single_step(self, **kwargs):
        logger = setup_logger()
        logger.debug("Forecasting one by one...")
        
        model = Theta(**kwargs)
    
    def optimize_optuna(self, periods: list, epoch: int=100):
        logger = setup_logger()
        logger.debug("Theta is optimizing...")
        
        def objective(trial):
            thetas = 2 - np.linspace(-10, 10, 100)
            theta = trial.suggest_categorical("theta", thetas)
            
            trend_mode = trial.suggest_categorical("trend_mode", 
                                                   [TrendMode.EXPONENTIAL, TrendMode.LINEAR])
            season_mode = trial.suggest_categorical("season_mode",
                                                    [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE])
            model_mode = trial.suggest_categorical("model_mode",
                                                   [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE])
            seasonality_period = trial.suggest_categorical("seasonality_period", periods)
            
            theta_model = Theta(theta=theta, trend_mode=trend_mode,
                                season_mode=season_mode,
                                model_mode=model_mode,
                                seasonality_period=seasonality_period)
            
            result = theta_model.fit(self.train)
            predictions = theta_model.predict(len(self.val))
            
            mae_error = mae(self.val, predictions)
            
            return mae_error
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=epoch, n_jobs=-1, show_progress_bar=True)
        
        return study
            

class ProphetDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, add_seasonalities: dict=None,
               growth: str="linear",
               holidays: str=None,
               plot: bool=0):
        """
        Parameters
        ----------
        add_seasonalities : dict, optional
            dict({
            'name': str  # (name of the seasonality component),
            'seasonal_periods': int  # (nr of steps composing a season),
            'fourier_order': int  # (number of Fourier components to use),
            'prior_scale': Optional[float]  # (a prior scale for this component),
            'mode': Optional[str]  # ('additive' or 'multiplicative')
            })
        growth : str, optional
            String 'linear', 'logistic' or 'flat' to specify a linear, 
            logistic or flat trend.
        holidays : str, optional
            Prophet includes holidays for these countries: 
                Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), 
                Vietnam (VN), Thailand (TH), Philippines (PH), Turkey (TU), 
                Pakistan (PK), Bangladesh (BD), Egypt (EG), 
                China (CN), and Russia (RU).
        """
        logger = setup_logger()
        logger.debug("Prophet is executing...")
              
        model = Prophet(add_seasonalities=add_seasonalities,
                        growth=growth,
                        holidays=holidays)
        predictions, accuracy = self.evaluation_model(model, 
                                                      title="Prophet Model",
                                                      plot=plot)
        
        return model, predictions, accuracy
        
    def optimize(self, add_seasonalities: dict=None, holidays: str=None):
        logger = setup_logger()
        logger.debug("Prohet is optimizing...")
        
        params = {
                'seasonality_mode': ('multiplicative','additive'),
                'growth': ('linear', 'flat'),
                'n_changepoints': [25, 50, 100, 150, 200, 250, 500],
                'changepoint_prior_scale': [0.05, 0.1,0.2,0.3,0.4,0.5],
            }
        
        best_model, best_params = Prophet.gridsearch(series=self.dart_series,
                                                     parameters=params,
                                                     forecast_horizon=1,
                                                     verbose=True,
                                                     n_jobs=-1)
        
        self.evaluation_model(best_model, title="Best Prophet Model")
        logger.info(f"Best parameters: \n {best_model}")
        
        return best_model

class ExponentialSmoothingDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, periods: list, 
               is_series: bool=1,
               smoothing_level: float=None,
               smoothing_trend: float=None,
               smoothing_seasonal: float=None,
               method: str="SLSQP",
               plot: bool=0):
        logger = setup_logger()
        logger.debug("TES is executing...")
            
        if is_series:
            for period in periods:
                model = ExponentialSmoothing(seasonal_periods=period,
                                             smoothing_level=smoothing_level,
                                             smoothing_trend=smoothing_trend,
                                             smoothing_seasonal=smoothing_seasonal,
                                             method=method)
                predictions, accuracy = self.evaluation_model(model, plot=plot)
        else:
            model = ExponentialSmoothing(smoothing_level=smoothing_level,
                                         smoothing_trend=smoothing_trend,
                                         smoothing_seasonal=smoothing_seasonal,
                                         method=method)
            predictions, accuracy = self.evaluation_model(model, plot=plot)
            
        return model, predictions, accuracy
    
    def optimize_optuna(self, periods: list, epoch: int=100):
        logger = setup_logger()
        logger.debug("TES is optimizing...")
        
        def objective(trial):
            period = trial.suggest_categorical("period", periods)
            smoothing_level = trial.suggest_float('smoothing_level', 0.05, 1)
            smoothing_trend = trial.suggest_float('smoothing_trend', 0.05, 1)
            smoothing_seasonal = trial.suggest_float('smoothing_seasonal', 0.05, 1)
            
            tes = ExponentialSmoothing(seasonal_periods=period,
                                       smoothing_level=smoothing_level,
                                       smoothing_seasonal=smoothing_seasonal,
                                       smoothing_trend=smoothing_trend)
            
            result = tes.fit(self.train)
            predictions = tes.predict(len(self.val))
            
            mae_error = mae(self.val, predictions)
            
            return mae_error
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=epoch, n_jobs=-1, show_progress_bar=True)
        
        return study

class OCSBTest:
    def __init__(self, data):
        self.data = data
        self.seasonal_diff = None
        self.ocbs_results = None
        self.optimal_order = None
        
    def seasonal_decompose(self, period):
        decomp = seasonal_decompose(self.data, model='additive', period=period)
        self.seasonal_diff = self.data - decomp.seasonal
        
    def autocovariance(self, y, k):
        n = len(y)
        mean_y = np.mean(y)
        gamma_k = sum([(y[i] - mean_y) * (y[i+k] - mean_y) for i in range(n-k)]) / n
        return gamma_k
    
    def ocsb_test(self, y):
        n = len(y)
        gamma_0 = self.autocovariance(y, 0)
        gamma_sum = sum([self.autocovariance(y, k) / gamma_0 for k in range(1, n)])
        ocsb = (1 / np.sqrt(n)) * gamma_sum
        return ocsb
    
    def perform_test(self):
        self.ocbs_results = []
        for i in range(1, 13):
            seasonal_diff_i = self.seasonal_diff.diff(i).dropna()
            ocsb_i = self.ocsb_test(seasonal_diff_i)
            self.ocbs_results.append((i, ocsb_i))
        
        self.optimal_order = min(self.ocbs_results, key=lambda x: abs(x[1]))
        
    def get_optimal_order(self):
        return self.optimal_order

class SARIMADart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, p: int=12,
               d: int=1,
               q: int=0,
               seasonal_order: tuple=(0, 0, 0, 0),
               trend :str=None, 
               plot: bool=0):
        """
        Parameters
        ----------
        p : int, optional
            Order (number of time lags) of the autoregressive model (AR).
        d : int, optional
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted (I).
        q : int, optional
            The size of the moving average window (MA).
        seasonal_order : tuple, optional
            The (P,D,Q,s) order of the seasonal component for the AR parameters,
            differences, MA parameters and periodicity.
        trend : str, optional
            Parameter controlling the deterministic trend. 'n' indicates no trend,
            'c' a constant term, 't' linear trend in time, and 'ct' includes both.
            Default is 'c' for models without integration, and no trend for models 
            with integration.
        plot : bool, optional
            DESCRIPTION. The default is 0.
        """
        
        logger = setup_logger()
        logger.debug("SARIMA is executing...")
        
        train = self.train.pd_series()
        
        period = seasonal_order[3]
        
        alpha = PPTest().should_diff(train)[0]
        first_diff = ndiffs(train, alpha=alpha, test="adf", max_d=3)
        d = first_diff
        
        ocsb = OCSBTest(train)
        ocsb.seasonal_decompose(period=period)
        ocsb.perform_test()
        seasonal_diff = ocsb.get_optimal_order()[0]
        seasonal_order = (
                seasonal_order[0],
                seasonal_diff,
                seasonal_order[2],
                period
            )
        
        model = ARIMA(p=p, d=d, q=q,
                      seasonal_order=seasonal_order,
                      trend=trend)
        
        predictions, accuracy = self.evaluation_model(model, 
                                                      title="SARIMA Model",
                                                      plot=plot)
        
        return model, predictions, accuracy
    
    def optimize_optuna(self, periods: list, epoch: int=100):
        logger = setup_logger()
        logger.debug("SARIMA is optimizing...")
        
        def objective(trial):
            period = trial.suggest_categorical("period", periods)
            p = trial.suggest_int("p", 0, 5)
            q= trial.suggest_int("q", 0, 5)
            d= trial.suggest_int("d", 1, 5)
            P = trial.suggest_int("P", 0, 5)
            Q = trial.suggest_int("Q", 0, 5)
            D = trial.suggest_int("D", 1, 5)
            seasonal_order = (P, D, Q, period)
            
            sarima = ARIMA(p=p, d=d, q=q,
                           seasonal_order=seasonal_order)
            
            result = sarima.fit(self.train)
            predictions = sarima.predict(len(self.val))
            
            mae_error = mae(self.val, predictions)
            
            return mae_error
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=epoch, n_jobs=-1, show_progress_bar=True)
        
        return study
            
