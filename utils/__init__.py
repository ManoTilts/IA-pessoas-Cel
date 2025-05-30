# -*- coding: utf-8 -*-
"""
Utilitários para o projeto de detecção de pessoas com celular usando YOLO.

Este pacote contém classes e funções auxiliares para:
- Detecção de objetos com YOLO
- Processamento de dados
- Visualização de resultados
"""

from .detector import PersonPhoneDetector
from .data_utils import DataProcessor

__version__ = "1.0.0"
__author__ = "Adriana Fujita, Daniel Henrique"

__all__ = [
    "PersonPhoneDetector",
    "DataProcessor"
] 