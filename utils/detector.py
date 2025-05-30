# -*- coding: utf-8 -*-
"""
Classe PersonPhoneDetector para detecção de pessoas com celulares usando YOLO.

Esta classe encapsula toda a funcionalidade de detecção, incluindo:
- Carregamento de modelos YOLO
- Processamento de imagens e vídeos
- Contagem e análise de detecções
- Anotação de resultados
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import os


class PersonPhoneDetector:
    """
    Detector de pessoas com celulares usando modelos YOLO.
    
    Esta classe fornece uma interface simples para detectar pessoas e celulares
    em imagens e vídeos, com suporte para modelos customizados e pré-treinados.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Inicializa o detector.
        
        Args:
            model_path (str, optional): Caminho para modelo customizado. 
                                      Se None, usa modelo pré-treinado.
            confidence_threshold (float): Limiar de confiança para detecções.
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_path)
        
        # Classes do COCO que nos interessam
        self.target_classes = {
            0: 'person',        # pessoa
            67: 'cell phone'    # celular
        }
        
        # Mapeamento para português
        self.class_names_pt = {
            0: 'Pessoa',
            67: 'Celular',
            999: 'Pessoa com Celular'  # classe customizada se disponível
        }
        
        print(f"✅ Detector inicializado no dispositivo: {self.device}")
        print(f"🎯 Limiar de confiança: {self.confidence_threshold}")
    
    def _load_model(self, model_path):
        """Carrega o modelo YOLO."""
        try:
            if model_path and os.path.exists(model_path):
                print(f"📦 Carregando modelo customizado: {model_path}")
                model = YOLO(model_path)
            else:
                print("📦 Carregando modelo pré-treinado YOLOv8n...")
                model = YOLO('yolov8n.pt')  # Modelo pré-treinado
                
            return model
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            print("🔄 Tentando carregar modelo YOLOv8n padrão...")
            return YOLO('yolov8n.pt')
    
    def set_confidence_threshold(self, threshold):
        """Atualiza o limiar de confiança."""
        self.confidence_threshold = threshold
        print(f"🎯 Limiar de confiança atualizado para: {threshold}")
    
    def detect(self, image):
        """
        Realiza detecção na imagem.
        
        Args:
            image: Imagem como array numpy ou caminho do arquivo
            
        Returns:
            Resultados da detecção do YOLO
        """
        try:
            # Realizar detecção
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            return results
            
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return None
    
    def count_detections(self, results):
        """
        Conta as detecções por categoria.
        
        Args:
            results: Resultados da detecção do YOLO
            
        Returns:
            tuple: (pessoas, celulares, pessoas_com_celulares)
        """
        if not results:
            return 0, 0, 0
            
        people_count = 0
        phones_count = 0
        people_with_phones = 0
        
        try:
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                    
                    for cls, conf in zip(classes, confidences):
                        if conf >= self.confidence_threshold:
                            cls_int = int(cls)
                            if cls_int == 0:  # pessoa
                                people_count += 1
                            elif cls_int == 67:  # celular
                                phones_count += 1
                            elif cls_int == 999:  # pessoa com celular (customizado)
                                people_with_phones += 1
            
            return people_count, phones_count, people_with_phones
            
        except Exception as e:
            print(f"❌ Erro ao contar detecções: {e}")
            return 0, 0, 0
    
    def get_detection_details(self, results):
        """
        Extrai detalhes das detecções.
        
        Args:
            results: Resultados da detecção do YOLO
            
        Returns:
            list: Lista de dicionários com detalhes das detecções
        """
        if not results:
            return []
            
        detections = []
        
        try:
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                    coordinates = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                    
                    for i, (cls, conf, coord) in enumerate(zip(classes, confidences, coordinates)):
                        if conf >= self.confidence_threshold:
                            cls_int = int(cls)
                            class_name = self.class_names_pt.get(cls_int, "Desconhecido")
                            
                            detection = {
                                "id": i + 1,
                                "classe": class_name,
                                "confianca": f"{conf:.2f}",
                                "coordenadas": f"({coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f}, {coord[3]:.0f})",
                                "largura": coord[2] - coord[0],
                                "altura": coord[3] - coord[1]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Erro ao extrair detalhes: {e}")
            return []
    
    def annotate_image(self, image, results):
        """
        Anota a imagem com as detecções.
        
        Args:
            image: Imagem original
            results: Resultados da detecção
            
        Returns:
            Imagem anotada
        """
        if not results or not results[0].boxes:
            return image
            
        try:
            # Usar o método plot do YOLO para anotação
            annotated = results[0].plot(
                conf=True,
                labels=True,
                boxes=True,
                line_width=2
            )
            return annotated
            
        except Exception as e:
            print(f"❌ Erro na anotação: {e}")
            return image
    
    def process_video_frame(self, frame):
        """
        Processa um frame de vídeo.
        
        Args:
            frame: Frame do vídeo
            
        Returns:
            tuple: (frame_anotado, contagens)
        """
        try:
            results = self.detect(frame)
            annotated_frame = self.annotate_image(frame, results)
            counts = self.count_detections(results)
            return annotated_frame, counts
            
        except Exception as e:
            print(f"❌ Erro no processamento do frame: {e}")
            return frame, (0, 0, 0)
    
    def analyze_detection_confidence(self, results):
        """
        Analisa a distribuição de confiança das detecções.
        
        Args:
            results: Resultados da detecção
            
        Returns:
            dict: Estatísticas de confiança
        """
        if not results:
            return {}
            
        confidences = []
        
        try:
            for r in results:
                if r.boxes is not None:
                    conf_values = r.boxes.conf.cpu().numpy()
                    confidences.extend(conf_values)
            
            if confidences:
                return {
                    "media": np.mean(confidences),
                    "mediana": np.median(confidences),
                    "min": np.min(confidences),
                    "max": np.max(confidences),
                    "desvio_padrao": np.std(confidences),
                    "total_deteccoes": len(confidences)
                }
            else:
                return {"total_deteccoes": 0}
                
        except Exception as e:
            print(f"❌ Erro na análise de confiança: {e}")
            return {}
    
    def save_results(self, results, output_path):
        """
        Salva os resultados em arquivo.
        
        Args:
            results: Resultados da detecção
            output_path: Caminho do arquivo de saída
        """
        try:
            import json
            
            detection_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_info": {
                    "confidence_threshold": self.confidence_threshold,
                    "device": self.device
                },
                "detections": self.get_detection_details(results),
                "summary": {
                    "people": self.count_detections(results)[0],
                    "phones": self.count_detections(results)[1],
                    "people_with_phones": self.count_detections(results)[2]
                },
                "confidence_stats": self.analyze_detection_confidence(results)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Resultados salvos em: {output_path}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")


def create_detector(model_path: str = None) -> PersonPhoneDetector:
    """
    Função auxiliar para criar uma instância do detector.
    
    Args:
        model_path (str): Caminho para o modelo (opcional)
        
    Returns:
        PersonPhoneDetector: Instância do detector
    """
    return PersonPhoneDetector(model_path) 