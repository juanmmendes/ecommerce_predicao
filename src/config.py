import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configurações globais do projeto."""
    
    # Caminhos
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
    
    # Configurações do banco
    DB_PATH: str = os.path.join(PROJECT_ROOT, "data", "ecommerce.db")
    
    # Parâmetros do modelo
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    
    # Definições de negócio
    CHURN_DAYS: int = 90  # Dias sem compra para considerar churn
    VIP_PERCENTILE: float = 0.75  # Percentil para cliente VIP
    MIN_LOYAL_ORDERS: int = 10  # Mínimo de pedidos para cliente fiel
    
    @classmethod
    def setup(cls) -> None:
        """Cria diretórios necessários."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.DATA_DIR, "raw"), exist_ok=True)
        os.makedirs(os.path.join(cls.DATA_DIR, "processed"), exist_ok=True)
