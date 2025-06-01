# -*- coding: utf-8 -*-
"""
Módulo de Detecção - PersonPhoneDetector
Compatível com o app.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class PersonPhoneDetector:
    """Detector de pessoas e celulares usando YOLO."""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Inicializa o detector."""
        self.confidence_threshold = confidence_threshold
        
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"  Modelo customizado carregado: {model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("  Modelo pré-treinado carregado")
        except Exception as e:
            print(f"  Erro ao carregar modelo: {e}")
            self.model = None
    
    def detect(self, image):
        """Detecta objetos na imagem."""
        if not self.model:
            return None
        
        try:
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            return results
        except Exception as e:
            print(f"  Erro na detecção: {e}")
            return None
    
    def count_detections(self, results):
        """Conta diferentes tipos de detecções."""
        people_count = 0
        phones_count = 0
        people_with_phones = 0
        
        if not results:
            return people_count, phones_count, people_with_phones
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.confidence_threshold:
                        if cls == 0:  # person
                            people_count += 1
                        elif cls == 67:  # cell phone
                            phones_count += 1
                        # Lógica para pessoas com celular (aproximação)
                        # Se pessoa e celular estão próximos, conta como pessoa_com_celular
        
        # Estimativa simples: pessoas com celular = min(pessoas, celulares)
        people_with_phones = min(people_count, phones_count)
        
        return people_count, phones_count, people_with_phones
    
    def annotate_image(self, image, results):
        """Anota a imagem com as detecções."""
        if not results:
            return image
        
        try:
            # Usar a função plot do YOLO
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"  Erro ao anotar imagem: {e}")
            return image
    
    def get_detection_details(self, results):
        """Retorna detalhes das detecções em formato de lista."""
        details = []
        
        if not results:
            return details
        
        class_names = {0: 'person', 67: 'cell phone'}
        
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.confidence_threshold:
                        # Coordenadas da bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        details.append({
                            'ID': i + 1,
                            'Classe': class_names.get(cls, f'classe_{cls}'),
                            'Confiança': round(conf, 3),
                            'X1': round(x1),
                            'Y1': round(y1),
                            'X2': round(x2),
                            'Y2': round(y2),
                            'Largura': round(x2 - x1),
                            'Altura': round(y2 - y1)
                        })
        
        return details
    
    def set_confidence_threshold(self, confidence):
        """Atualiza o limiar de confiança."""
        self.confidence_threshold = confidence
    
    def get_model_info(self):
        """Retorna informações do modelo."""
        if not self.model:
            return {"erro": "Modelo não carregado"}
        
        try:
            return {
                "modelo": "YOLOv8",
                "classes_total": len(self.model.names),
                "classes_principais": ["person", "cell phone"],
                "confianca_atual": self.confidence_threshold,
                "dispositivo": "CPU" if not hasattr(self.model, 'device') else str(self.model.device)
            }
        except Exception as e:
            return {"erro": str(e)}
    
    def process_video(self, video_path, max_frames=100):
        """Processa vídeo e retorna estatísticas por frame."""
        if not self.model:
            return []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_results = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                # Detectar no frame
                results = self.detect(frame)
                people, phones, people_with_phones = self.count_detections(results)
                
                frame_results.append({
                    'frame': frame_count,
                    'pessoas': people,
                    'celulares': phones,
                    'pessoas_com_celular': people_with_phones,
                    'total_deteccoes': people + phones,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                })
                
                frame_count += 1
            
            cap.release()
            return frame_results
            
        except Exception as e:
            print(f" Erro ao processar vídeo: {e}")
            return [] 