from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def get_estimator():
    """
    Retourne un pipeline de base pour la classification du stress.
    Le pipeline inclut :
      - Une imputation avec la stratégie 'median'
      - Une standardisation des données
      - Un XGBClassifier (objectif multi:softprob) avec des paramètres par défaut
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            objective='multi:softprob', 
            eval_metric='mlogloss', 
            use_label_encoder=False,
            random_state=42
        ))
    ])
    return pipeline