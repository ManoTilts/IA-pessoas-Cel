# -*- coding: utf-8 -*-
"""
Utilitários para processamento de dados e visualização.

Este módulo contém funções auxiliares para:
- Preparação de datasets
- Visualização de resultados
- Análise estatística
- Manipulação de arquivos
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple
from pathlib import Path
import yaml
import json


class DataProcessor:
    """
    Classe para processamento e análise de dados do projeto.
    """
    
    def __init__(self):
        """Inicializa o processador de dados."""
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def create_dataset_config(self, dataset_path: str, output_path: str = "data/dataset.yaml"):
        """
        Cria arquivo de configuração do dataset para YOLO.
        
        Args:
            dataset_path (str): Caminho para o dataset
            output_path (str): Caminho para salvar o arquivo YAML
        """
        config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'names': {
                0: 'pessoa',
                1: 'celular',
                2: 'pessoa_com_celular'
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Configuração do dataset salva em: {output_path}")
        return config
    
    def analyze_detection_results(self, results: List[Dict]) -> Dict:
        """
        Analisa estatisticamente os resultados de detecção.
        
        Args:
            results (List[Dict]): Lista de resultados de detecção
            
        Returns:
            Dict: Estatísticas dos resultados
        """
        if not results:
            return {"erro": "Nenhum resultado para analisar"}
        
        df = pd.DataFrame(results)
        
        stats = {
            "total_frames": len(df),
            "estatisticas_pessoas": {
                "media": df['pessoas'].mean(),
                "mediana": df['pessoas'].median(),
                "maximo": df['pessoas'].max(),
                "minimo": df['pessoas'].min(),
                "desvio_padrao": df['pessoas'].std()
            },
            "estatisticas_celulares": {
                "media": df['celulares'].mean(),
                "mediana": df['celulares'].median(),
                "maximo": df['celulares'].max(),
                "minimo": df['celulares'].min(),
                "desvio_padrao": df['celulares'].std()
            },
            "total_deteccoes": df['total_deteccoes'].sum(),
            "frame_com_mais_deteccoes": df.loc[df['total_deteccoes'].idxmax()].to_dict()
        }
        
        return stats
    
    def plot_detection_timeline(self, results: List[Dict], save_path: str = None):
        """
        Cria gráfico temporal das detecções.
        
        Args:
            results (List[Dict]): Resultados de detecção
            save_path (str): Caminho para salvar o gráfico (opcional)
        """
        if not results:
            print("❌ Nenhum resultado para plotar")
            return
        
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Detecções ao longo do tempo
        plt.subplot(2, 2, 1)
        plt.plot(df['frame'], df['pessoas'], label='Pessoas', marker='o', alpha=0.7)
        plt.plot(df['frame'], df['celulares'], label='Celulares', marker='s', alpha=0.7)
        if 'pessoas_com_celular' in df.columns:
            plt.plot(df['frame'], df['pessoas_com_celular'], label='Pessoas c/ Celular', marker='^', alpha=0.7)
        plt.xlabel('Frame')
        plt.ylabel('Número de Detecções')
        plt.title('Detecções ao Longo do Tempo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Distribuição de pessoas
        plt.subplot(2, 2, 2)
        plt.hist(df['pessoas'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Número de Pessoas')
        plt.ylabel('Frequência')
        plt.title('Distribuição do Número de Pessoas')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Distribuição de celulares
        plt.subplot(2, 2, 3)
        plt.hist(df['celulares'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Número de Celulares')
        plt.ylabel('Frequência')
        plt.title('Distribuição do Número de Celulares')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Total de detecções
        plt.subplot(2, 2, 4)
        plt.plot(df['frame'], df['total_deteccoes'], color='red', linewidth=2)
        plt.fill_between(df['frame'], df['total_deteccoes'], alpha=0.3, color='red')
        plt.xlabel('Frame')
        plt.ylabel('Total de Detecções')
        plt.title('Total de Detecções por Frame')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def create_detection_heatmap(self, detections: List[Dict], image_shape: Tuple[int, int]):
        """
        Cria mapa de calor das detecções.
        
        Args:
            detections (List[Dict]): Lista de detecções com coordenadas
            image_shape (Tuple[int, int]): Dimensões da imagem (altura, largura)
        """
        height, width = image_shape
        heatmap = np.zeros((height, width))
        
        for detection in detections:
            coords = detection['coordenadas']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Garantir que as coordenadas estão dentro dos limites
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            
            heatmap[int(y1):int(y2), int(x1):int(x2)] += 1
        
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Densidade de Detecções')
        plt.title('Mapa de Calor das Detecções')
        plt.xlabel('Largura (pixels)')
        plt.ylabel('Altura (pixels)')
        plt.show()
        
        return heatmap
    
    def export_results_to_csv(self, results: List[Dict], output_path: str):
        """
        Exporta resultados para arquivo CSV.
        
        Args:
            results (List[Dict]): Resultados a exportar
            output_path (str): Caminho do arquivo CSV
        """
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Resultados exportados para: {output_path}")
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {e}")
    
    def load_sample_images(self, images_dir: str, max_images: int = 5) -> List[str]:
        """
        Carrega caminhos de imagens de exemplo.
        
        Args:
            images_dir (str): Diretório com imagens
            max_images (int): Número máximo de imagens
            
        Returns:
            List[str]: Lista de caminhos das imagens
        """
        image_paths = []
        
        if not os.path.exists(images_dir):
            print(f"⚠️ Diretório não encontrado: {images_dir}")
            return image_paths
        
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in self.supported_image_formats):
                image_paths.append(os.path.join(images_dir, file))
                if len(image_paths) >= max_images:
                    break
        
        print(f"📷 Encontradas {len(image_paths)} imagens em {images_dir}")
        return image_paths
    
    def create_detection_summary(self, results: List[Dict]) -> str:
        """
        Cria resumo textual dos resultados.
        
        Args:
            results (List[Dict]): Resultados de detecção
            
        Returns:
            str: Resumo formatado
        """
        if not results:
            return "Nenhum resultado disponível para resumir."
        
        stats = self.analyze_detection_results(results)
        
        summary = f"""
📊 RESUMO DOS RESULTADOS DE DETECÇÃO

🎯 Estatísticas Gerais:
   • Total de frames processados: {stats['total_frames']}
   • Total de detecções: {stats['total_deteccoes']}

👥 Pessoas Detectadas:
   • Média por frame: {stats['estatisticas_pessoas']['media']:.1f}
   • Máximo em um frame: {stats['estatisticas_pessoas']['maximo']}
   • Desvio padrão: {stats['estatisticas_pessoas']['desvio_padrao']:.1f}

📱 Celulares Detectados:
   • Média por frame: {stats['estatisticas_celulares']['media']:.1f}
   • Máximo em um frame: {stats['estatisticas_celulares']['maximo']}
   • Desvio padrão: {stats['estatisticas_celulares']['desvio_padrao']:.1f}

🏆 Frame com Mais Detecções:
   • Frame: {stats['frame_com_mais_deteccoes']['frame']}
   • Total de detecções: {stats['frame_com_mais_deteccoes']['total_deteccoes']}
        """
        
        return summary
    
    def validate_dataset_structure(self, dataset_path: str) -> Dict:
        """
        Valida a estrutura do dataset.
        
        Args:
            dataset_path (str): Caminho do dataset
            
        Returns:
            Dict: Resultado da validação
        """
        validation = {
            "valido": True,
            "problemas": [],
            "estatisticas": {}
        }
        
        required_dirs = ['images', 'labels']
        
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path):
                validation["valido"] = False
                validation["problemas"].append(f"Diretório ausente: {dir_name}")
        
        # Contar arquivos
        if os.path.exists(os.path.join(dataset_path, 'images')):
            images = len([f for f in os.listdir(os.path.join(dataset_path, 'images')) 
                         if any(f.lower().endswith(ext) for ext in self.supported_image_formats)])
            validation["estatisticas"]["total_imagens"] = images
        
        if os.path.exists(os.path.join(dataset_path, 'labels')):
            labels = len([f for f in os.listdir(os.path.join(dataset_path, 'labels')) 
                         if f.endswith('.txt')])
            validation["estatisticas"]["total_labels"] = labels
        
        return validation


def create_sample_data():
    """
    Cria dados de exemplo para demonstração.
    """
    sample_data = []
    
    for i in range(50):
        frame_data = {
            'frame': i,
            'timestamp': i * 0.033,  # ~30 FPS
            'pessoas': np.random.randint(0, 5),
            'celulares': np.random.randint(0, 3),
            'pessoas_com_celular': np.random.randint(0, 2),
        }
        frame_data['total_deteccoes'] = (frame_data['pessoas'] + 
                                       frame_data['celulares'] + 
                                       frame_data['pessoas_com_celular'])
        sample_data.append(frame_data)
    
    return sample_data


def setup_project_structure(base_path: str = "."):
    """
    Configura a estrutura completa do projeto.
    
    Args:
        base_path (str): Caminho base do projeto
    """
    directories = [
        "models",
        "data/images/train",
        "data/images/val", 
        "data/images/test",
        "data/videos",
        "data/labels/train",
        "data/labels/val",
        "data/labels/test",
        "utils",
        "docs",
        "results"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Criado: {dir_path}")
    
    print("✅ Estrutura do projeto configurada com sucesso!") 