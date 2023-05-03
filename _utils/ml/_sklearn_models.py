from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from xgboost import XGBRegressor
from _utils.logging_utils import setup_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def prepare_dataframe(df: pd.DataFrame,
                      target_column: str) -> tuple:
    temp_df = df.copy()
    temp_df = pd.get_dummies(temp_df, drop_first=True)
    
    X, y = temp_df.drop(target_column, axis=1), df[target_column]
    
    return (X, y)

def holdout_dataframe(df: pd.DataFrame, target_column: str) -> tuple:
    X, y = prepare_dataframe(df, target_column)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, 
                                                        shuffle=False)
    
    return (X_train, X_test, y_train, y_test)

class Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def init_model(self):
        pass
    
    def get_scores(self, cv: int=5,
                   scoring: list=["roc_auc", "f1", "precision", 
                                  "recall", "accuracy",
                                  "neg_mean_squared_error",
                                  "neg_root_mean_squared_error", 
                                  'neg_mean_absolute_percentage_error', 
                                  "r2"]) -> pd.DataFrame:
        cross_result = cross_validate(self.model, self.X, self.y,
                                      cv=cv, scoring=scoring)
        results = {metric: abs(cross_result["test_" + metric].mean())
                   for metric in scoring}
        result_df = pd.DataFrame(results.items()).dropna()
        result_df.set_index(0, inplace=True)
        
        result_df[1] = result_df[1].apply(lambda x: np.format_float_positional(x, trim='-'))
        
        return result_df
    
    def visualize_feature_importance(self, num: int=20, save: bool=0):
        logger = setup_logger()
        logger.debug("visualize_feature_importance function is executing...")
        
        feature_importance = pd.DataFrame({"Value": self.model.feature_importances_,
                                           "Feature": self.X.columns})
        
        total_value = feature_importance["Value"].sum()
        feature_importance["Value"] = (feature_importance["Value"] / total_value) * 100
        
        logger.info(f"Feature importances for the machine learning model:\n{feature_importance}")
        
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", 
                    data=feature_importance.sort_values(by="Value", 
                                                        ascending=False)[0: num])
        plt.title("Features")
        plt.tight_layout()
        
        if save: plt.savefig("feature_importances.png")
        
        plt.show(block=True)
    
class RandomForestSklearn(Model):
    def __init__(self, X, y):
        Model.__init__(self, X, y)
        
    def init_model(self, _type: bool=1):
        self.model = RandomForestClassifier().fit(self.X, self.y) \
            if _type else RandomForestRegressor().fit(self.X, self.y)

class LogisticRegressionSklearn(Model):
    def __init__(self, X, y):
        Model.__init__(self, X, y)
        
    def init_model(self):
        self.model = LogisticRegression().fit(self.X, self.y)
    
    
class LinearRegressionSklearn(Model):
    def __init__(self, X, y):
        Model.__init__(self, X, y)
        
    def init_model(self):
        self.model = LinearRegression().fit(self.X, self.y)
        
class XGBRegressorModel(Model):
    def __init__(self, X, y):
        Model.__init__(self, X, y)
        
    def init_model(self):
        self.model = XGBRegressor().fit(self.X, self.y)
