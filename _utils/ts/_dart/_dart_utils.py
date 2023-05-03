from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.utils.statistics import plot_acf, check_seasonality
from darts.metrics import mae, mape, smape, r2_score
from _utils.logging_utils import setup_logger
from darts import TimeSeries

from darts.models import NaiveSeasonal,NaiveDrift

from darts.models.forecasting.croston import Croston
from darts.models.forecasting.dlinear import DLinearModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.tft_model import TFTModel

import matplotlib.pyplot as plt
import optuna
import time

class Smoother:
    def __init__(self, dart_series: TimeSeries,
                 percentage: float=0.8):
        self.dart_series = dart_series
        self.percentage = percentage
        
        self.train, self.val = self.dart_series.split_before(self.percentage)

    def get_season_period(self, season_range: int):
        logger = setup_logger()
        logger.debug("get_season_period is executing...")
        
        periods = []
        
        season_control = lambda func: func[1] \
            if func[0] else None
                
        periods = [
            season_control(check_seasonality(self.train, m=m, alpha=0.05, max_lag=season_range + 1))
            for m in range(2, season_range)
        ]
        periods = [period for period in periods if period is not None]
        
        logger.info(f"Expected Seasonality Periods: \n{periods}")
        
        self.period = min(periods) if periods is not None else None
        self.periods = periods
        
        return periods
            
    def visualize_seasonality(self):
        plot_acf(self.train, m=12, alpha=0.05)
        
    def evaluation_model(self, model, title: str=None, plot: bool=1):
        logger = setup_logger()
        logger.debug(f"{model} is evaluating...")
        
        time_start = time.perf_counter()
        
        result = model.fit(self.train)
        predictions = model.predict(len(self.val))

        mae_error = mae(self.val, predictions)
        mape_error = mape(self.val, predictions)
        smape_error = smape(self.val, predictions)
        r2_error = r2_score(self.val, predictions)
        
        time_taken = time.perf_counter() - time_start
        
        accuracy = {
            "MAE": mae_error,
            "MAPE": mape_error,
            "SMAPE": smape_error,
            "R2": r2_error,
            "Time": round(time_taken, 3)
        }
        
        if title: model = title
        
        logger.info(f"Error metrics: \n{accuracy}")
        logger.info(f"\n{model} evaluated.\nTaken time: {time_taken}")
        
        if plot: self.plot(predictions, mae_error, model)
       
        return [predictions, accuracy]
        
    def plot(self, predictions, error, title):
        train_count = int(len(self.train)) - int(len(self.val) / 2)
        
        plt.figure(figsize=(15, 10))
        
        self.train[train_count:].plot(label="Train")
        self.val.plot(label="Validation")
        predictions.plot(label="Prediction")
        plt.title(f"{title} - MAE: {error: .4f}")
        plt.legend()
        
        plt.show()
        
    def optimize_optuna(self):
        pass
        
    def get_optimized_result(self, study, full_graphs: bool=0):
        logger = setup_logger()
        logger.debug("Result is loading...")
        
        trial = study.best_trial
        logger.info(f"Best Mean Absolute Error: {trial.value: .3f}")
        logger.info(f"Best parameters: \n{trial.params}")
        
        if full_graphs:
            optuna.visualization.plot_optimization_history(study).show(renderer="png")
            optuna.visualization.plot_slice(study).show(renderer="png")
            optuna.visualization.plot_param_importances(study).show(renderer="png")
        else:
            optuna.visualization.plot_param_importances(study).show(renderer="png")
        
class TFTDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("TFTModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=len(self.val),
                                min_delta=0.05,
                                mode="min")
        
        model = TFTModel(input_chunk_length=len(self.val), 
                            output_chunk_length=len(self.val),
                            n_epochs=n_epochs,
                            nr_epochs_val_period=1,
                            add_relative_index=True,
                            pl_trainer_kwargs={"accelerator": "gpu", 
                                               "devices": -1, 
                                               "auto_select_gpus": True,
                                               "callbacks": [stopper]})
        self.evaluation_model(model, title="TFTModel")
        
class TCNDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("TCNModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.05,
                                mode="min")
        
        model = TCNModel(input_chunk_length=len(self.val) * 2, 
                            output_chunk_length=len(self.val),
                            n_epochs=n_epochs,
                            nr_epochs_val_period=1,
                            pl_trainer_kwargs={"accelerator": "gpu", 
                                               "devices": -1, 
                                               "auto_select_gpus": True,
                                               "callbacks": [stopper]})
        self.evaluation_model(model, title="TCNModel")
        
class RNNDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100, model_name: str="RNN"):
        logger = setup_logger()
        logger.debug("BlockRNNModel is executing...")
        
        assert model_name in ["RNN", "LSTM", "GRU"], "Select one of 'RNN', 'LSTM', 'GRU'"
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.01,
                                mode="min")
        
        model = RNNModel(input_chunk_length=len(self.val), 
                             output_chunk_length=len(self.val),
                             model=model_name,
                             n_epochs=n_epochs,
                             nr_epochs_val_period=1,
                             pl_trainer_kwargs={"accelerator": "gpu", 
                                                "devices": -1, 
                                                "auto_select_gpus": True,
                                                "callbacks": [stopper]})
        self.evaluation_model(model, title=f"RNNModel - {model_name}")
        
class NLinearDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("NLinearModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.05,
                                mode="min")
        
        model = NLinearModel(input_chunk_length=len(self.val), 
                            output_chunk_length=len(self.val),
                            n_epochs=n_epochs,
                            nr_epochs_val_period=1,
                            normalize=True,
                            use_static_covariates=False,
                            pl_trainer_kwargs={"accelerator": "gpu", 
                                               "devices": -1, 
                                               "auto_select_gpus": True,
                                               "callbacks": [stopper]})
        self.evaluation_model(model, title="NLinearModel")
        
class NHITSDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("NHiTSModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.05,
                                mode="min")
        
        model = NHiTSModel(input_chunk_length=len(self.val), 
                            output_chunk_length=len(self.val),
                            n_epochs=n_epochs,
                            nr_epochs_val_period=1,
                            pl_trainer_kwargs={"accelerator": "gpu", 
                                               "devices": -1, 
                                               "auto_select_gpus": True,
                                               "callbacks": [stopper]})
        self.evaluation_model(model, title="NHiTSModel")
        
class NBEATSDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("NBEATSModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.05,
                                mode="min")
        
        model = NBEATSModel(input_chunk_length=len(self.val), 
                            output_chunk_length=len(self.val),
                            n_epochs=n_epochs,
                            nr_epochs_val_period=1,
                            pl_trainer_kwargs={"accelerator": "gpu", 
                                               "devices": -1, 
                                               "auto_select_gpus": True,
                                               "callbacks": [stopper]})
        self.evaluation_model(model, title="NBEATSModel")
        
class FFTDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, freq: str="month", trend: str="poly"):
        logger = setup_logger()
        logger.debug("BlockRNNModel is executing...")
        logger.warning("Frequencies may be 'month', 'day' ext.")
        logger.warning("Trend may be 'poly', 'exp' and 'None'.")
        
        if trend == 'poly':
            best_mae, best_degree = float("inf"), 0
            for degree in range(1, 11):
                model = FFT(nr_freqs_to_keep=freq, 
                            trend=trend,
                            trend_poly_degree=degree)
                _, accuracy = self.evaluation_model(model, plot=0)
                
                mae_error = accuracy["MAE"]
                if mae_error < best_mae: best_mae, best_degree = mae_error, degree
                
            model = FFT(nr_freqs_to_keep=freq, 
                        trend=trend,
                        trend_poly_degree=best_degree)
            self.evaluation_model(model)
        else:
            model = FFT(nr_freqs_to_keep=freq, 
                        trend=trend)
            self.evaluation_model(model)
        
class BlockRNNDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100, model_name: str="RNN"):
        logger = setup_logger()
        logger.debug("BlockRNNModel is executing...")
        
        assert model_name in ["RNN", "LSTM", "GRU"], "Select one of 'RNN', 'LSTM', 'GRU'"
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.01,
                                mode="min")
        
        model = BlockRNNModel(input_chunk_length=len(self.val), 
                             output_chunk_length=len(self.val),
                             model=model_name,
                             n_epochs=n_epochs,
                             nr_epochs_val_period=1,
                             pl_trainer_kwargs={"accelerator": "gpu", 
                                                "devices": -1, 
                                                "auto_select_gpus": True,
                                                "callbacks": [stopper]})
        self.evaluation_model(model, title=f"BlockRNNModel - {model_name}")
        
class DLinearDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, n_epochs: int=100):
        logger = setup_logger()
        logger.debug("DLinearModel is executing...")
        
        stopper = EarlyStopping(monitor="train_loss",
                                patience=int(len(self.val)),
                                min_delta=0.01,
                                mode="min")
        
        model = DLinearModel(input_chunk_length=len(self.val), 
                             output_chunk_length=len(self.val), 
                             use_static_covariates=True,
                             n_epochs=n_epochs,
                             nr_epochs_val_period=1,
                             pl_trainer_kwargs={"accelerator": "gpu", 
                                                "devices": -1, 
                                                "auto_select_gpus": True,
                                                "callbacks": [stopper]})
        self.evaluation_model(model, title="DLinearModel")
        
class CrostonDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, optimized: bool=1):
        logger = setup_logger()
        logger.debug("Croston is executing...")
        
        if optimized:
            model = Croston(version="optimized")
            self.evaluation_model(model)
        else:
            model = Croston()
            self.evaluation_model(model)
        
class NaiveDart(Smoother):
    def __init__(self, dart_series, percentage: float=0.8):
        Smoother.__init__(self, dart_series, percentage)
        
    def smooth(self, periods: list):
        logger = setup_logger()
        logger.debug("Naive Algorithm is executing...")
        
        best_model, best_period, best_mae = None, None, float("inf")
        for period in periods:          
            model = NaiveDrift()
            _, accuracy = self.evaluation_model(model, period, plot=0)
            
            mae_error = accuracy["MAE"]
            if mae_error < best_mae: best_model, best_period, best_mae = model, period, mae_error
            logger.info(f"Naive Models - MAE: {mae_error}")
            
        logger.info(f"Best Model: \n{best_model}")
        logger.info(f"Best Period: {best_period}, Best MAE: {best_mae: .3f}")
        self.evaluation_model(best_model, best_period)
            
    def evaluation_model(self, model, period, plot=1):
        logger = setup_logger()
        logger.debug(f"{model} is evaluating...")
        
        time_start = time.perf_counter()
        
        result = model.fit(self.train)
        predictions = model.predict(len(self.val))
        
        model_seasonal = NaiveSeasonal(K=period)
        model_seasonal.fit(self.train)
        
        forecast_drift = predictions
        forecast_seasonal = model_seasonal.predict(len(self.val))
        predictions = forecast_drift + forecast_seasonal - self.train.last_value()
        
        mae_error = mae(self.val, predictions)
        mape_error = mape(self.val, predictions)
        smape_error = smape(self.val, predictions)
        r2_error = r2_score(self.val, predictions)
        
        time_taken = time.perf_counter() - time_start
        
        accuracy = {
            "MAE": mae_error,
            "MAPE": mape_error,
            "SMAPE": smape_error,
            "R2": r2_error,
            "Time": round(time_taken, 3)
        }
        
        logger.info(f"Error metrics: \n{accuracy}")
        logger.info(f"\n{model} evaluated.\nTaken time: {time_taken}")
        
        if plot: self.plot(predictions, mae_error, f"{model} - Period: {period}")
       
        return [predictions, accuracy]
