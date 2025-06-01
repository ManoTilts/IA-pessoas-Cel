# -*- coding: utf-8 -*-
"""
Módulo de Utilitários de Dados - DataProcessor
Compatível com o app.py
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    """Processador de dados para o projeto YOLO."""
    
    def __init__(self):
        """Inicializa o processador."""
        pass
    
    def validate_dataset_structure(self, dataset_path):
        """Valida a estrutura do dataset."""
        try:
            dataset_path = Path(dataset_path)
            
            required_dirs = [
                'images/train',
                'images/val', 
                'images/test',
                'labels/train',
                'labels/val',
                'labels/test'
            ]
            
            problemas = []
            estatisticas = {}
            
            # Verificar diretórios
            for dir_name in required_dirs:
                dir_path = dataset_path / dir_name
                if not dir_path.exists():
                    problemas.append(f"Diretório faltando: {dir_name}")
                else:
                    # Contar arquivos
                    if 'images' in dir_name:
                        files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpeg'))
                    else:
                        files = list(dir_path.glob('*.txt'))
                    
                    estatisticas[dir_name] = len(files)
            
            # Verificar arquivo de configuração
            yaml_file = dataset_path / 'dataset.yaml'
            if not yaml_file.exists():
                problemas.append("Arquivo dataset.yaml não encontrado")
            
            valido = len(problemas) == 0
            
            return {
                "valido": valido,
                "problemas": problemas,
                "estatisticas": estatisticas
            }
            
        except Exception as e:
            return {
                "valido": False,
                "problemas": [f"Erro na validação: {e}"],
                "estatisticas": {}
            }
    
    def get_dataset_statistics(self, dataset_path):
        """Retorna estatísticas do dataset."""
        try:
            stats = {}
            dataset_path = Path(dataset_path)
            
            splits = ['train', 'val', 'test']
            
            for split in splits:
                images_dir = dataset_path / 'images' / split
                labels_dir = dataset_path / 'labels' / split
                
                if images_dir.exists():
                    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpeg'))
                    stats[f'{split}_images'] = len(image_files)
                
                if labels_dir.exists():
                    label_files = list(labels_dir.glob('*.txt'))
                    stats[f'{split}_labels'] = len(label_files)
            
            return stats
            
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return {}
    
    def create_dataset_report(self, dataset_path, output_path):
        """Cria relatório do dataset."""
        try:
            validation = self.validate_dataset_structure(dataset_path)
            statistics = self.get_dataset_statistics(dataset_path)
            
            report = {
                "dataset_path": str(dataset_path),
                "validation": validation,
                "statistics": statistics,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Relatório salvo em: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar relatório: {e}")
            return False

def create_sample_data(num_frames=50):
    """Cria dados de exemplo para demonstração."""
    np.random.seed(42)  # Para resultados consistentes
    
    frames = list(range(num_frames))
    
    # Simular dados de detecção realistas
    pessoas = np.random.poisson(3, num_frames)  # Média de 3 pessoas por frame
    celulares = np.random.poisson(2, num_frames)  # Média de 2 celulares por frame
    
    # Pessoas com celular não pode ser maior que min(pessoas, celulares)
    pessoas_com_celular = [min(p, c) for p, c in zip(pessoas, celulares)]
    pessoas_com_celular = [max(0, pc - np.random.randint(0, 2)) for pc in pessoas_com_celular]  # Adicionar variação
    
    total_deteccoes = pessoas + celulares
    
    data = []
    for i in range(num_frames):
        data.append({
            'frame': frames[i],
            'pessoas': int(pessoas[i]),
            'celulares': int(celulares[i]),
            'pessoas_com_celular': int(pessoas_com_celular[i]),
            'total_deteccoes': int(total_deteccoes[i]),
            'timestamp': i * 0.033  # ~30 FPS
        })
    
    return data

def load_detection_results(file_path):
    """Carrega resultados de detecção de um arquivo."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Erro ao carregar resultados: {e}")
        return None

def save_detection_results(results, file_path):
    """Salva resultados de detecção em arquivo."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Resultados salvos em: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        return False

def analyze_detection_trends(results_data):
    """Analisa tendências nos dados de detecção."""
    try:
        if not results_data:
            return {}
        
        df = pd.DataFrame(results_data)
        
        analysis = {
            "tendencias": {
                "pessoas_media": df['pessoas'].mean(),
                "celulares_media": df['celulares'].mean(),
                "total_frames": len(df),
                "frame_com_mais_deteccoes": df.loc[df['total_deteccoes'].idxmax()]['frame'] if not df.empty else 0
            },
            "estatisticas": {
                "pessoas": {
                    "min": df['pessoas'].min(),
                    "max": df['pessoas'].max(),
                    "std": df['pessoas'].std()
                },
                "celulares": {
                    "min": df['celulares'].min(),
                    "max": df['celulares'].max(),
                    "std": df['celulares'].std()
                }
            }
        }
        
        return analysis
        
    except Exception as e:
        print(f"❌ Erro na análise de tendências: {e}")
        return {}

# Função auxiliar para compatibilidade
def get_data_processor():
    """Retorna uma instância do DataProcessor."""
    return DataProcessor() 