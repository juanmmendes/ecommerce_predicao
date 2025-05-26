import pytest
import sqlite3
from src.etl.pipeline import EcommerceETL

@pytest.fixture
def etl():
    """Fixture para criar instância de teste do ETL com conexão em memória compartilhada."""
    conn = sqlite3.connect(":memory:")
    return EcommerceETL(":memory:", conn=conn)

def test_generate_sample_data(etl):
    """Testa geração de dados de exemplo."""
    etl.generate_sample_data()
    
    # Verificar se as tabelas foram criadas
    cursor = etl.conn.cursor()
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [table[0] for table in tables]
    
    assert "customers" in table_names
    assert "products" in table_names
    assert "orders" in table_names

def test_transform_data(etl):
    """Testa transformação de dados."""
    etl.generate_sample_data()
    etl.extract_data()
    etl.transform_data()
    
    df = etl.data['transformed']
    
    # Verificar colunas essenciais
    required_columns = [
        'customer_id', 'age', 'gender', 'city',
        'total_spent', 'is_churn'
    ]
    for col in required_columns:
        assert col in df.columns
    
    # Verificar tipos de dados
    assert df['is_churn'].dtype == 'int'
    assert df['total_spent'].dtype == 'float'
    
    # Verificar regras de negócio
    assert df['age'].min() >= 18
    assert df['age'].max() <= 80

def test_churn_prediction(etl):
    """Testa modelo de predição de churn."""
    etl.generate_sample_data()
    etl.extract_data()
    etl.transform_data()
    etl.train_predictive_model()
    
    # Verificar se o modelo foi treinado
    assert etl.model is not None
    
    # Usar todas as features do modelo
    features = [
        'age', 'gender_encoded', 'city_encoded', 'premium_member',
        'total_spent', 'avg_order_value', 'total_orders',
        'categories_bought', 'discount_usage', 'avg_rating',
        'days_since_last_order', 'customer_lifetime_days'
    ]
    df = etl.data['transformed']
    X = df[features]
    predictions = etl.model.predict(etl.scaler.transform(X))
    
    assert len(predictions) == len(df)
