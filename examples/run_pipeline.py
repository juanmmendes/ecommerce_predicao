"""
Exemplo de uso do sistema de análise de e-commerce.
Este script demonstra as principais funcionalidades do projeto.
"""

import logging
from pathlib import Path

from src.config import Config
from src.etl import EcommerceETL
from src.utils import setup_logging

def main():
    # Configurar logging
    logger = setup_logging(
        log_file=Path("logs") / "ecommerce_analytics.log"
    )
    
    # Configurar ambiente
    Config.setup()
    
    try:
        # Inicializar pipeline
        logger.info("Iniciando pipeline ETL")
        etl = EcommerceETL(Config.DB_PATH)
        
        # Executar pipeline completo
        etl.run_complete_pipeline()
        
        logger.info("Pipeline concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}", exc_info=True)
        raise
    finally:
        if 'etl' in locals():
            etl.close_connection()

if __name__ == "__main__":
    main()
