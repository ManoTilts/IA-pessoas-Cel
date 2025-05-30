# -*- coding: utf-8 -*-
"""
Demo script para testar o Detector de Pessoas com Celular.

Este script demonstra como usar a classe PersonPhoneDetector
para detectar pessoas e celulares em imagens.
"""

import cv2
import numpy as np
from utils.detector import PersonPhoneDetector
from pathlib import Path
import matplotlib.pyplot as plt

def demo_detector():
    """
    Demonstração básica do detector.
    """
    print("🚀 Iniciando demonstração do Detector de Pessoas com Celular")
    print("=" * 60)
    
    # Inicializar detector
    print("\n📦 Carregando detector...")
    detector = PersonPhoneDetector(confidence_threshold=0.5)
    
    # Verificar se há imagens no diretório data/images
    images_dir = Path("data/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if image_files:
            print(f"\n📊 Encontradas {len(image_files)} imagens para teste")
            
            # Testar com a primeira imagem
            test_image = image_files[0]
            print(f"\n🔍 Testando com: {test_image.name}")
            
            # Carregar imagem
            image = cv2.imread(str(test_image))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Realizar detecção
            print("🔄 Realizando detecção...")
            results = detector.detect(image)
            
            if results:
                # Contar detecções
                people, phones, people_with_phones = detector.count_detections(results)
                
                print(f"\n📊 Resultados da Detecção:")
                print(f"   👥 Pessoas: {people}")
                print(f"   📱 Celulares: {phones}")
                print(f"   📱👥 Pessoas com Celular: {people_with_phones}")
                
                # Obter detalhes
                details = detector.get_detection_details(results)
                print(f"\n🔍 Detalhes das Detecções ({len(details)} total):")
                for i, detail in enumerate(details[:5]):  # Mostrar apenas as primeiras 5
                    print(f"   {i+1}. {detail['classe']} - Confiança: {detail['confianca']}")
                
                # Criar imagem anotada
                annotated = detector.annotate_image(image, results)
                
                # Salvar resultado
                output_path = Path("data/resultado_demo.jpg")
                cv2.imwrite(str(output_path), annotated)
                print(f"\n💾 Resultado salvo em: {output_path}")
                
                # Analisar confiança
                confidence_stats = detector.analyze_detection_confidence(results)
                if confidence_stats:
                    print(f"\n📈 Estatísticas de Confiança:")
                    print(f"   Média: {confidence_stats.get('media', 0):.3f}")
                    print(f"   Mínima: {confidence_stats.get('min', 0):.3f}")
                    print(f"   Máxima: {confidence_stats.get('max', 0):.3f}")
                
            else:
                print("❌ Nenhuma detecção encontrada")
                
        else:
            print("\n⚠️ Nenhuma imagem encontrada no diretório data/images/")
            print("   Você pode adicionar imagens (.jpg ou .png) para testar o detector")
    else:
        print("\n⚠️ Diretório data/images/ não encontrado")
        print("   Criando diretório para você adicionar imagens de teste...")
        images_dir.mkdir(parents=True, exist_ok=True)
    
    # Teste com imagem sintética
    print("\n🎨 Testando com imagem sintética...")
    synthetic_image = create_synthetic_test_image()
    
    results = detector.detect(synthetic_image)
    if results:
        people, phones, people_with_phones = detector.count_detections(results)
        print(f"   Detecções na imagem sintética: {people} pessoas, {phones} celulares")
    
    print("\n✅ Demonstração concluída!")
    print("\n📝 Para usar o detector em suas próprias imagens:")
    print("   1. Adicione imagens ao diretório data/images/")
    print("   2. Execute este script novamente")
    print("   3. Ou use a aplicação Streamlit: streamlit run app.py")

def create_synthetic_test_image():
    """
    Cria uma imagem sintética simples para teste.
    """
    # Criar uma imagem simples com forma de pessoa
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.fill(100)  # Fundo cinza
    
    # Desenhar formas que possam ser detectadas como pessoa
    cv2.rectangle(image, (250, 150), (350, 400), (255, 255, 255), -1)  # Corpo
    cv2.circle(image, (300, 120), 30, (255, 255, 255), -1)  # Cabeça
    
    return image

def test_video_processing():
    """
    Teste básico de processamento de vídeo.
    """
    print("\n🎥 Teste de processamento de vídeo...")
    
    videos_dir = Path("data/videos")
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
        
        if video_files:
            print(f"📹 Encontrados {len(video_files)} vídeos para teste")
            # Aqui poderia adicionar processamento de vídeo
        else:
            print("⚠️ Nenhum vídeo encontrado no diretório data/videos/")
    else:
        print("⚠️ Diretório data/videos/ não encontrado")
        videos_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    try:
        demo_detector()
        test_video_processing()
    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        print("   Verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt") 