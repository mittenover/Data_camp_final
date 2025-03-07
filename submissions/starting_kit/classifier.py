from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class Classifier(BaseEstimator):
    """
    Retourne un pipeline de base pour la classification du stress.
    Le pipeline inclut :
      - Une imputation avec la stratégie 'median'
      - Une standardisation des données
      - Un XGBClassifier (objectif multi:softprob) avec des paramètres par défaut
    """
    def __init__(self):
        self.transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.model = XGBClassifier(
            objective='multi:softprob', 
            eval_metric='mlogloss', 
            use_label_encoder=False,
            random_state=42
        )
        self.pipeline = Pipeline(steps=[
            ('transformer', self.transformer),
            ('model', self.model)
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)