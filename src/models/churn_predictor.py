import joblib
import pandas as pd
from sklearn.base import BaseEstimator

class ChurnPredictor:
    """Modelo de predição de churn."""
    
    def __init__(self, model_path: str):
        self.model: BaseEstimator = joblib.load(model_path)
        self.feature_names = [
            'age', 'gender_encoded', 'city_encoded', 'premium_member',
            'total_spent', 'avg_order_value', 'total_orders',
            'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]
    
    def predict(self, customer_data: pd.DataFrame) -> pd.Series:
        """Faz predições para novos clientes."""
        return self.model.predict_proba(customer_data[self.feature_names])[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
