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
    """Detector de pessoas e celulares usando YOLO com configurações otimizadas."""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Inicializa o detector."""
        self.confidence_threshold = confidence_threshold
        
        # MELHORIAS: Configurações específicas por classe otimizadas
        self.class_configs = {
            0: {  # person
                'name': 'person',
                'conf_threshold': 0.5,
                'iou_threshold': 0.5,
                'color': (255, 0, 0),  # Vermelho
                'min_area': 1000,
                'max_area': 500000
            },
            67: {  # cell phone
                'name': 'cell phone',
                'conf_threshold': 0.2,  # REDUZIDO de 0.5 para 0.2
                'iou_threshold': 0.4,
                'color': (0, 255, 0),  # Verde
                'min_area': 50,         # Área mínima pequena
                'max_area': 20000,      # Área máxima para celulares
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 4.0
            }
        }
        
        # MELHORIAS: Escalas para detecção multi-escala
        self.scales = [640, 832, 1024]
        self.scale_confidences = [0.3, 0.25, 0.2]
        
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
    
    def detect(self, image, multi_scale=True):
        """Detecta objetos na imagem com configurações otimizadas."""
        if not self.model:
            return None
        
        try:
            if multi_scale:
                # Detecção multi-escala para melhorar detecção de celulares
                return self._detect_multi_scale(image)
            else:
                # Detecção padrão
                results = self.model(image, conf=0.25, iou=0.4, verbose=False)
                return results
        except Exception as e:
            print(f"  Erro na detecção: {e}")
            return None
    
    def _detect_multi_scale(self, image):
        """Realiza detecção em múltiplas escalas."""
        original_height, original_width = image.shape[:2] if len(image.shape) == 3 else (image.shape[0], image.shape[1])
        
        all_detections = []
        
        for scale, base_conf in zip(self.scales, self.scale_confidences):
            try:
                # Redimensionar imagem mantendo proporção
                if scale != max(original_width, original_height):
                    # Calcular nova dimensão mantendo aspect ratio
                    if original_width > original_height:
                        new_width = scale
                        new_height = int(scale * original_height / original_width)
                    else:
                        new_height = scale
                        new_width = int(scale * original_width / original_height)
                    
                    resized_img = cv2.resize(image, (new_width, new_height))
                else:
                    resized_img = image
                    new_width, new_height = original_width, original_height
                
                # Fazer detecção com configuração específica
                results = self.model(resized_img, conf=base_conf, iou=0.4, verbose=False)
                
                # Processar resultados com validações melhoradas
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # MELHORIA: Aplicar limiar específico da classe
                            class_config = self.class_configs.get(cls, {})
                            min_conf = class_config.get('conf_threshold', 0.3)
                            
                            if confidence >= min_conf:
                                # Converter coordenadas de volta para escala original
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                if scale != max(original_width, original_height):
                                    scale_factor_x = original_width / new_width
                                    scale_factor_y = original_height / new_height
                                    
                                    x1 = x1 * scale_factor_x
                                    y1 = y1 * scale_factor_y
                                    x2 = x2 * scale_factor_x
                                    y2 = y2 * scale_factor_y
                                
                                # MELHORIA: Validação específica para celulares
                                if cls == 67:  # cell phone
                                    width = x2 - x1
                                    height = y2 - y1
                                    area = width * height
                                    aspect_ratio = height / width if width > 0 else 0
                                    
                                    # Filtros melhorados para celulares
                                    min_area = class_config.get('min_area', 50)
                                    max_area = class_config.get('max_area', 20000)
                                    min_ratio = class_config.get('min_aspect_ratio', 0.3)
                                    max_ratio = class_config.get('max_aspect_ratio', 4.0)
                                    
                                    if area < min_area or area > max_area:
                                        continue
                                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                                        continue
                                
                                detection = {
                                    'class': cls,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],
                                    'scale': scale
                                }
                                all_detections.append(detection)
            
            except Exception as e:
                print(f"⚠️ Erro na escala {scale}: {e}")
        
        # MELHORIA: Aplicar NMS customizado para remover duplicatas
        unique_detections = self._apply_nms_improved(all_detections)
        
        # Converter de volta para formato YOLO results
        return self._convert_to_yolo_results(unique_detections, image.shape)
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Aplica Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Separar por classe
        detections_by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        final_detections = []
        
        for cls, class_detections in detections_by_class.items():
            # Usar IoU threshold específico da classe
            class_config = self.class_configs.get(cls, {})
            iou_thresh = class_config.get('iou_threshold', iou_threshold)
            
            # Ordenar por confiança (maior primeiro)
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            selected = []
            for current in class_detections:
                overlap = False
                for selected_det in selected:
                    iou = self._calculate_iou(current['bbox'], selected_det['bbox'])
                    if iou > iou_thresh:
                        overlap = True
                        break
                
                if not overlap:
                    selected.append(current)
            
            final_detections.extend(selected)
        
        return final_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Interseção
            x1_int = max(x1_1, x1_2)
            y1_int = max(y1_1, y1_2)
            x2_int = min(x2_1, x2_2)
            y2_int = min(y2_1, y2_2)
            
            if x2_int <= x1_int or y2_int <= y1_int:
                return 0.0
            
            intersection = (x2_int - x1_int) * (y2_int - y1_int)
            
            # União
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _convert_to_yolo_results(self, detections, image_shape):
        """Converte detecções de volta para formato YOLO results."""
        # Esta é uma implementação simplificada
        # Na prática, você pode manter as detecções no formato customizado
        return detections
    def count_detections(self, results):
        """
        MELHORIA: Conta diferentes tipos de detecções com lógica melhorada.
        
        Implementa lógica de proximidade para detectar pessoas com celulares.
        """
        people_count = 0
        phones_count = 0
        people_with_phones = 0
        
        if not results:
            return people_count, phones_count, people_with_phones
        
        people_boxes = []
        phone_boxes = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.confidence_threshold:
                        bbox = box.xyxy[0].tolist()
                        
                        if cls == 0:  # person
                            people_count += 1
                            people_boxes.append(bbox)
                        elif cls == 67:  # cell phone
                            phones_count += 1
                            phone_boxes.append(bbox)
        
        # MELHORIA: Lógica melhorada para detectar pessoas com celulares
        # Verifica proximidade entre pessoas e celulares
        for person_box in people_boxes:
            for phone_box in phone_boxes:
                if self._is_phone_near_person(person_box, phone_box):
                    people_with_phones += 1
                    break  # Uma pessoa pode ter apenas um celular contado
        
        return people_count, phones_count, people_with_phones
    
    def _is_phone_near_person(self, person_box, phone_box, proximity_factor=0.3):
        """
        MELHORIA: Verifica se um celular está próximo de uma pessoa.
        
        Args:
            person_box: [x1, y1, x2, y2] da pessoa
            phone_box: [x1, y1, x2, y2] do celular
            proximity_factor: Fator de proximidade (0.3 = 30% da área da pessoa)
        """
        px1, py1, px2, py2 = person_box
        fx1, fy1, fx2, fy2 = phone_box
        
        # Calcular centro do celular
        phone_center_x = (fx1 + fx2) / 2
        phone_center_y = (fy1 + fy2) / 2
        
        # Expandir área da pessoa para verificar proximidade
        person_width = px2 - px1
        person_height = py2 - py1
        
        expansion_x = person_width * proximity_factor
        expansion_y = person_height * proximity_factor
        
        expanded_px1 = px1 - expansion_x
        expanded_py1 = py1 - expansion_y
        expanded_px2 = px2 + expansion_x
        expanded_py2 = py2 + expansion_y
        
        # Verificar se o centro do celular está na área expandida da pessoa
        return (expanded_px1 <= phone_center_x <= expanded_px2 and
                expanded_py1 <= phone_center_y <= expanded_py2)
    
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