from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

class Transformer(ABC, BaseEstimator, TransformerMixin):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame):
        pass

class Transformer(ABC, BaseEstimator, TransformerMixin):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame):
        pass

# Binary encode gender

class OrdinalEncoderGender(Transformer):
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        pass

    def fit(self, X, y=None):
        self.encoder.fit(X[["Gender"]])
        return self

    def transform(self, X):
        X_t = X.copy()
        X_t["Gender"] = self.encoder.transform(X[["Gender"]])
        return X_t
    
class OrdinalEncoderOther(Transformer):
    def __init__(self, category, column):
        self.cat = category
        self.col = column
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[self.cat])
        pass

    def fit(self, X, y=None):
        self.encoder.fit(X[[self.col]])
        return self

    def transform(self, X):
        X_t = X.copy()
        X_t[self.col] = self.encoder.transform(X[[self.col]])
        return X

class Classifier(BaseEstimator):
    def __init__(self):
        self.transformer = Pipeline(
            steps=[
                ("Gender", OrdinalEncoderGender()),
                ("Physical Activity", OrdinalEncoderOther(["Low", "Moderate", "High"], "Physical_Activity")),
                ("Sleep Quality", OrdinalEncoderOther(["Poor", "Moderate", "Good"], "Sleep_Quality")),
                ("Mood", OrdinalEncoderOther(["Stressed", "Neutral", "Happy"], "Mood")),
                ("Health Risk Level", OrdinalEncoderOther(["Low", "Moderate", "High"], "Health_Risk_Level")),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.model = LogisticRegression(max_iter=500)
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        # X = X.drop(["groups"], axis=1)
        self.pipe.fit(X, y)

    def predict(self, X):
        # X = X.drop(["groups"], axis=1)
        return self.pipe.predict(X)

    def predict_proba(self, X):
        # X = X.drop(["groups"], axis=1)
        return self.pipe.predict_proba(X)
