import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura o logging para o projeto."""
    
    # Criar logger
    logger = logging.getLogger("ecommerce_analytics")
    logger.setLevel(level)
    
    # Formato do log
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo se especificado
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
