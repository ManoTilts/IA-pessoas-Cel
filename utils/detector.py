# -*- coding: utf-8 -*-
"""
Módulo de Detecção - PersonPhoneDetector
Compatível com o app.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import warnings

# Suprimir todos os avisos relacionados ao torch e ultralytics
warnings.filterwarnings("ignore")
os.environ['ULTRALYTICS_QUIET'] = 'true'

class PersonPhoneDetector:
    """Detector de pessoas e celulares usando YOLO com configurações otimizadas."""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Inicializa o detector."""
        print(f"🔧 Inicializando detector (confidence={confidence_threshold})")
        
        self.confidence_threshold = confidence_threshold
        self.model = None
        
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
                'conf_threshold': 0.15,  # REDUZIDO ainda mais para detectar melhor
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
        self.scale_confidences = [0.2, 0.15, 0.1]  # Reduzir ainda mais os thresholds
        
        # Tentar carregar modelo com tratamento de erro melhorado
        try:
            print("📥 Carregando modelo YOLO...")
            
            if model_path and os.path.exists(model_path):
                print(f"📁 Tentando carregar modelo customizado: {model_path}")
                self.model = YOLO(model_path, verbose=False)
                print(f"✅ Modelo customizado carregado: {model_path}")
            else:
                print("📥 Baixando/carregando modelo pré-treinado YOLOv8n...")
                self.model = YOLO('yolov8n.pt', verbose=False)
                print("✅ Modelo pré-treinado carregado")
                
            # Verificar se o modelo foi carregado corretamente
            if self.model is None:
                raise Exception("Modelo retornado é None")
                
            # Teste rápido do modelo
            print("🧪 Testando modelo...")
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(test_img, conf=0.5, verbose=False)
            print(f"✅ Teste do modelo bem-sucedido: {len(test_results)} resultados")
            
        except Exception as e:
            error_msg = f"❌ Erro ao carregar modelo: {str(e)}"
            print(error_msg)
            
            # Verificar se é erro de dependências
            if "No module named" in str(e):
                print("📦 Erro de dependências detectado")
                print("💡 Execute: pip install ultralytics torch torchvision")
            elif "torch" in str(e).lower():
                print("🔥 Erro relacionado ao PyTorch detectado")
                print("💡 Execute: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            elif "Permission denied" in str(e) or "WinError" in str(e):
                print("🔒 Erro de permissão/Windows detectado")
                print("💡 Execute como administrador ou verifique antivírus")
            
            self.model = None
            # Não levantar exceção, apenas retornar com model=None
    
    def detect(self, image, multi_scale=True):
        """Detecta objetos na imagem com configurações otimizadas."""
        if not self.model:
            print("❌ Modelo não carregado")
            return []
        
        try:
            print(f"🔍 Iniciando detecção (multi_scale={multi_scale})")
            print(f"📷 Imagem shape: {image.shape}")
            
            # Usar detecção simples com parâmetros otimizados para detectar celulares
            # Usar confiança muito baixa para detectar celulares e depois filtrar
            results = self.model(image, conf=0.1, iou=0.4, verbose=False)
            print(f"✅ Detecção retornou: {type(results)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return []
    
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
        unique_detections = self._apply_nms(all_detections)
        
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
    
    def _apply_nms_improved(self, detections, iou_threshold=0.5):
        """Versão melhorada do NMS que mantém detecções de diferentes classes."""
        return self._apply_nms(detections, iou_threshold)
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union entre duas bounding boxes."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calcular área de intersecção
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calcular área das duas bounding boxes
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # Calcular união
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _convert_to_yolo_results(self, detections, image_shape):
        """Converte detecções de volta para formato YOLO results."""
        # Se não há detecções, retorna None para indicar resultado vazio
        if not detections:
            return None
        
        # Para multi-escala, simplifique retornando detecção direta do modelo
        # A conversão completa seria complexa, então vamos usar detecção simples
        return None  # Isso força o uso da detecção simples
    
    def count_detections(self, results):
        """
        MELHORIA: Conta diferentes tipos de detecções com lógica melhorada.
        
        Implementa lógica de proximidade para detectar pessoas com celulares.
        """
        people_count = 0
        phones_count = 0
        people_with_phones = 0
        
        try:
            if not results:
                return people_count, phones_count, people_with_phones
            
            people_boxes = []
            phone_boxes = []
            
            # Verificar se results é lista de detecções customizadas ou objeto YOLO
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                # Formato customizado (multi-escala)
                for detection in results:
                    cls = detection.get('class', -1)
                    conf = detection.get('confidence', 0.0)
                    bbox = detection.get('bbox', [])
                    
                    # Usar threshold específico por classe
                    class_config = self.class_configs.get(cls, {})
                    min_conf = class_config.get('conf_threshold', self.confidence_threshold)
                    
                    if conf >= min_conf and len(bbox) == 4:
                        if cls == 0:  # person
                            people_count += 1
                            people_boxes.append(bbox)
                            print(f"👤 Detecção adicionada - Classe: {cls}, Conf: {conf:.3f}")
                        elif cls == 67:  # cell phone
                            phones_count += 1
                            phone_boxes.append(bbox)
                            print(f"📱 Detecção adicionada - Classe: {cls}, Conf: {conf:.3f}")
            else:
                # Formato YOLO padrão
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Usar threshold específico por classe
                            class_config = self.class_configs.get(cls, {})
                            min_conf = class_config.get('conf_threshold', self.confidence_threshold)
                            
                            if conf >= min_conf:
                                bbox = box.xyxy[0].tolist()
                                
                                if cls == 0:  # person
                                    people_count += 1
                                    people_boxes.append(bbox)
                                    print(f"👤 Detecção adicionada - Classe: {cls}, Conf: {conf:.3f}")
                                elif cls == 67:  # cell phone
                                    phones_count += 1
                                    phone_boxes.append(bbox)
                                    print(f"📱 Detecção adicionada - Classe: {cls}, Conf: {conf:.3f}")
                                else:
                                    # Log para debug de outras classes próximas
                                    if 60 <= cls <= 70:
                                        print(f"🔍 Classe {cls} detectada com conf {conf:.3f} (não é celular)")
            
            # MELHORIA: Lógica melhorada para detectar pessoas com celulares
            # Verifica proximidade entre pessoas e celulares
            for person_box in people_boxes:
                for phone_box in phone_boxes:
                    if self._is_phone_near_person(person_box, phone_box):
                        people_with_phones += 1
                        break  # Uma pessoa pode ter apenas um celular contado
            
            print(f"📊 Detecções: {people_count} pessoas, {phones_count} celulares, {people_with_phones} pessoas c/ celular")
            return people_count, phones_count, people_with_phones
            
        except Exception as e:
            print(f"❌ Erro em count_detections: {e}")
            return 0, 0, 0
    
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
            # Se results é uma lista de detecções customizadas, desenhar manualmente
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                annotated = image.copy()
                
                for detection in results:
                    cls = detection['class']
                    conf = detection['confidence']
                    bbox = detection['bbox']
                    
                    # Usar threshold específico por classe
                    class_config = self.class_configs.get(cls, {})
                    min_conf = class_config.get('conf_threshold', self.confidence_threshold)
                    
                    if conf >= min_conf:
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Cor baseada na classe
                        if cls == 0:  # person
                            color = (255, 0, 0)  # Vermelho
                            label = f"Person {conf:.2f}"
                        elif cls == 67:  # cell phone
                            color = (0, 255, 0)  # Verde
                            label = f"Phone {conf:.2f}"
                        else:
                            color = (0, 0, 255)  # Azul
                            label = f"Class_{cls} {conf:.2f}"
                        
                        # Desenhar retângulo
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Desenhar label
                        cv2.putText(annotated, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                return annotated
            else:
                # Usar a função plot do YOLO para resultados padrão
                annotated = results[0].plot()
                return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            print(f"❌ Erro ao anotar imagem: {e}")
            return image
    
    def get_detection_details(self, results):
        """Retorna detalhes das detecções em formato de lista."""
        details = []
        
        if not results:
            return details
        
        class_names = {0: 'person', 67: 'cell phone'}
        
        # Verificar se results é lista de detecções customizadas ou objeto YOLO
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            # Formato customizado (multi-escala)
            for i, detection in enumerate(results):
                cls = detection['class']
                conf = detection['confidence']
                bbox = detection['bbox']
                
                # Usar threshold específico por classe
                class_config = self.class_configs.get(cls, {})
                min_conf = class_config.get('conf_threshold', self.confidence_threshold)
                
                if conf >= min_conf:
                    x1, y1, x2, y2 = bbox
                    
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
        else:
            # Formato YOLO padrão
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for i, box in enumerate(r.boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Usar threshold específico por classe
                        class_config = self.class_configs.get(cls, {})
                        min_conf = class_config.get('conf_threshold', self.confidence_threshold)
                        
                        if conf >= min_conf:
                            # Coordenadas da bounding box
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            details.append({
                                'ID': len(details) + 1,
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
        """Define o threshold de confiança."""
        self.confidence_threshold = confidence
    
    def get_model_info(self):
        """Retorna informações sobre o modelo."""
        if not self.model:
            return {"status": "Modelo não carregado"}
        
        try:
            info = {
                "Tipo": str(type(self.model).__name__),
                "Device": str(self.model.device),
                "Confiança": self.confidence_threshold,
                "Classes_suportadas": "Person, Cell Phone",
                "Status": "Carregado"
            }
            return info
        except Exception as e:
            return {"Status": f"Erro: {e}"}
    
    def process_video(self, video_path, max_frames=100):
        """Processa vídeo frame por frame."""
        if not self.model:
            return None
        
        # Esta funcionalidade seria implementada para processamento de vídeo
        # Por enquanto, retorna placeholder
        return {"message": "Processamento de vídeo não implementado nesta versão"} 