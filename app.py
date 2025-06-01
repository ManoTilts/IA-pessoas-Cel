# -*- coding: utf-8 -*-
"""
Aplicação Streamlit para Detecção de Pessoas com Celular usando YOLO.

Esta aplicação web permite:
- Upload e análise de imagens
- Upload e análise de vídeos
- Visualização de resultados em tempo real
- Download de relatórios
"""

import warnings
import os
import sys
import logging

# Configurar supressão completa de warnings ANTES de importar qualquer coisa
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*")
warnings.filterwarnings("ignore", message=".*running event loop.*")

# Configurar variáveis de ambiente para supressão
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ULTRALYTICS_QUIET'] = 'true'

# Configurar logging para suprimir mensagens do PyTorch/Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# Suprimir stderr temporariamente para imports
from io import StringIO
import contextlib

@contextlib.contextmanager
def suppress_output():
    """Context manager para suprimir output temporariamente."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = devnull
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

# Imports principais com supressão
with suppress_output():
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    import tempfile
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from utils.detector import PersonPhoneDetector
    from utils.data_utils import DataProcessor
    import json

# Configurar avisos do Streamlit
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('client.showErrorDetails', False)
except Exception:
    pass

# Configuração da página
st.set_page_config(
    page_title=" Detector de Pessoas com Celular",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📱 Detector de Pessoas com Celular</h1>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("**Sistema inteligente para detecção de pessoas usando celulares em imagens e vídeos**")

# Sidebar para configurações
st.sidebar.header(" Configurações")
st.sidebar.markdown("---")

# Configurações do modelo
confidence = st.sidebar.slider(
    " Confiança Mínima", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.1,
    help="Ajuste o limiar de confiança para as detecções"
)

model_option = st.sidebar.selectbox(
    "🤖 Modelo YOLO",
    ["Modelo Pré-treinado (COCO)", "Modelo Customizado"],
    help="Escolha entre o modelo pré-treinado ou um modelo customizado"
)

# Configurações de visualização
show_confidence = st.sidebar.checkbox(" Mostrar Confiança", value=True)
show_labels = st.sidebar.checkbox(" Mostrar Rótulos", value=True)
save_results = st.sidebar.checkbox(" Salvar Resultados", value=False)

# Configurações avançadas
st.sidebar.markdown("---")
st.sidebar.subheader(" Configurações Avançadas")

# Opção de detecção multi-escala
multi_scale = st.sidebar.checkbox(
    " Detecção Multi-escala", 
    value=True,
    help="Melhora detecção de objetos pequenos (celulares)"
)

# Configurações específicas por classe
with st.sidebar.expander(" Configurações por Classe"):
    conf_pessoas = st.sidebar.slider(
        "Confiança para Pessoas", 
        0.1, 0.9, 0.5, 0.05,
        help="Limiar de confiança para detectar pessoas"
    )
    
    conf_celulares = st.sidebar.slider(
        "Confiança para Celulares", 
        0.1, 0.9, 0.25, 0.05,
        help="Limiar menor para detectar celulares"
    )

# Filtros de qualidade
with st.sidebar.expander(" Filtros de Qualidade"):
    aplicar_filtros = st.sidebar.checkbox(
        "Aplicar Filtros de Celular", 
        value=True,
        help="Remove detecções implausíveis de celulares"
    )
    
    area_min_celular = st.sidebar.number_input(
        "Área Mínima Celular (px)", 
        50, 1000, 100,
        help="Área mínima para considerar um celular válido"
    )
    
    area_max_celular = st.sidebar.number_input(
        "Área Máxima Celular (px)", 
        1000, 100000, 50000,
        help="Área máxima para considerar um celular válido"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("###  Suporte")
st.sidebar.info("""
**Desenvolvedores:**
- Andre Akio Morita Osakawa (10340045)
- André Franco Ranieri (10390470)
- Felipe Mazzeo Barbosa (10402808)
- Fernando Pegoraro Bilia (10402097)
- Francesco Zangrandi Coppola (10403340)

**Universidade Presbiteriana Mackenzie**
""")

# Inicializar detector
@st.cache_resource
def load_detector():
    """Carrega o detector YOLO com configurações melhoradas."""
    try:
        
        # Verificar se existe modelo customizado
        if model_option == "Modelo Customizado":
            model_path = "models/best_model.pt"
            if os.path.exists(model_path):
                print(f" Modelo customizado encontrado: {model_path}")
                detector = PersonPhoneDetector(model_path, confidence)
                if detector.model is not None:
                    st.sidebar.success(" Modelo customizado carregado!")
                    
                    # Mostrar informações do modelo
                    with st.sidebar.expander("ℹ Informações do Modelo"):
                        model_info = detector.get_model_info()
                        for key, value in model_info.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    return detector
                else:
                    print(" Falha ao carregar modelo customizado")
                    st.sidebar.error(" Falha ao carregar modelo customizado")
                    return None
            else:
                print(" Modelo customizado não encontrado, tentando pré-treinado")
                st.sidebar.warning(" Modelo customizado não encontrado, usando pré-treinado")
        
        print(" Carregando modelo pré-treinado...")
        detector = PersonPhoneDetector(None, confidence)
        
        if detector.model is not None:
            print(" Modelo pré-treinado carregado com sucesso!")
            st.sidebar.success(" Modelo pré-treinado carregado!")
            
            # Mostrar informações do modelo
            with st.sidebar.expander("ℹ Informações do Modelo"):
                model_info = detector.get_model_info()
                for key, value in model_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            return detector
        else:
            print(" Falha ao carregar modelo pré-treinado")
            st.sidebar.error(" Falha ao carregar modelo pré-treinado")
            return None
        
    except Exception as e:
        error_msg = f" Erro ao carregar modelo: {str(e)}"
        print(error_msg)
        st.sidebar.error(error_msg)
        
        # Mostrar detalhes do erro para debug
        import traceback
        full_error = traceback.format_exc()
        print(f" Traceback completo:\n{full_error}")
        
        # Mostrar erro na interface apenas se for diferente de torch.classes
        if "torch.classes" not in str(e):
            with st.sidebar.expander(" Detalhes do Erro", expanded=False):
                st.code(full_error, language="python")
        
        return None

# Carregar detector
detector = load_detector()

if detector:
    # Atualizar confiança do detector
    detector.set_confidence_threshold(confidence)
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([" Análise de Imagens", " Análise de Vídeos", " Estatísticas", "ℹ Sobre"])
    
    # ==================== TAB 1: ANÁLISE DE IMAGENS ====================
    with tab1:
        st.header(" Detecção em Imagens")
        st.markdown("Faça upload de uma imagem para detectar pessoas e celulares.")
        
        # Upload de imagem
        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Formatos suportados: JPG, JPEG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Carregar e exibir imagem original
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Imagem Original")
                st.image(image, use_container_width=True)
                
                # Informações da imagem
                st.info(f"""
                **Informações da Imagem:**
                - Dimensões: {image.size[0]} x {image.size[1]} pixels
                - Formato: {image.format}
                - Modo: {image.mode}
                """)
            
            with col2:
                st.subheader(" Resultados da Detecção")
                  # Processar imagem
                with st.spinner(" Processando imagem..."):
                    try:
                        # Atualizar configurações do detector
                        detector.set_confidence_threshold(confidence)
                        
                        # Atualizar configurações específicas por classe se disponível
                        if hasattr(detector, 'class_configs'):
                            detector.class_configs[0]['conf_threshold'] = conf_pessoas  # pessoa
                            detector.class_configs[67]['conf_threshold'] = conf_celulares  # celular
                        
                        # Fazer detecção com configurações avançadas
                        results = detector.detect(img_array, multi_scale=multi_scale)
                        
                        if results:
                            # Contar detecções
                            people, phones, people_with_phones = detector.count_detections(results)
                            
                            # Criar imagem anotada
                            annotated_img = detector.annotate_image(img_array, results)
                            
                            # Mostrar resultado
                            st.image(annotated_img, use_container_width=True)
                            
                            # Obter detalhes das detecções
                            detection_details = detector.get_detection_details(results)
                            
                        else:
                            st.warning(" Nenhuma detecção encontrada.")
                            people = phones = people_with_phones = 0
                            detection_details = []
                            
                    except Exception as e:
                        st.error(f" Erro no processamento: {e}")
                        people = phones = people_with_phones = 0
                        detection_details = []
            
            # Métricas dos resultados
            st.markdown("---")
            st.subheader("📊 Resumo dos Resultados")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=" Pessoas",
                    value=people,
                    help="Número total de pessoas detectadas"
                )
            
            with col2:
                st.metric(
                    label="📱 Celulares", 
                    value=phones,
                    help="Número total de celulares detectados"
                )
            
            with col3:
                st.metric(
                    label="📱👥 Pessoas c/ Celular",
                    value=people_with_phones,
                    help="Número de pessoas usando celular"
                )
            
            with col4:
                total_detections = people + phones  
                st.metric(
                    label="🔍 Total de Detecções",
                    value=total_detections,
                    help="Soma de todas as detecções (pessoas + celulares)"
                )
            
            # Detalhes das detecções
            if detection_details:
                with st.expander(" Detalhes das Detecções", expanded=False):
                    st.write(f"**{len(detection_details)} detecções encontradas:**")
                    
                    # Criar DataFrame para exibição
                    df_detections = pd.DataFrame(detection_details)
                    st.dataframe(df_detections, use_container_width=True)
                    
                    # Opção de download
                    if save_results:
                        csv = df_detections.to_csv(index=False)
                        st.download_button(
                            label=" Baixar Detecções (CSV)",
                            data=csv,
                            file_name=f"deteccoes_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
    
    # ==================== TAB 2: ANÁLISE DE VÍDEOS ====================
    with tab2:
        st.header(" Detecção em Vídeos")
        st.markdown("Faça upload de um vídeo para análise frame por frame.")
        
        # Upload de vídeo
        uploaded_video = st.file_uploader(
            "Escolha um vídeo",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formatos suportados: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            # Salvar vídeo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Exibir vídeo
            st.video(uploaded_video)
            
            # Configurações de processamento
            col1, col2 = st.columns(2)
            
            with col1:
                max_frames = st.number_input(
                    "🎬 Máximo de Frames",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Número máximo de frames a processar (para demonstração)"
                )
            
            with col2:
                process_interval = st.number_input(
                    "⏭ Intervalo de Processamento",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Processar a cada N frames"
                )
            
            # Botão para processar vídeo
            if st.button("🎬 Processar Vídeo", type="primary"):
                # Placeholder para progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Processar vídeo
                    with st.spinner(" Processando vídeo..."):
                        frame_results = detector.process_video(video_path, max_frames)
                    
                    if frame_results:
                        st.success(f" Processamento concluído! {len(frame_results)} frames analisados.")
                        
                        # Criar DataFrame dos resultados
                        df_video = pd.DataFrame(frame_results)
                        
                        # Gráficos de análise temporal
                        st.subheader(" Análise Temporal")
                        
                        # Gráfico de linha temporal
                        fig_timeline = px.line(
                            df_video, 
                            x='frame', 
                            y=['pessoas', 'celulares', 'pessoas_com_celular'],
                            title='Detecções ao Longo do Tempo',
                            labels={'value': 'Número de Detecções', 'frame': 'Frame'},
                            color_discrete_map={
                                'pessoas': '#1f77b4',
                                'celulares': '#ff7f0e', 
                                'pessoas_com_celular': '#2ca02c'
                            }
                        )
                        fig_timeline.update_layout(height=400)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Estatísticas do vídeo
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                " Média de Pessoas",
                                f"{df_video['pessoas'].mean():.1f}",
                                f"Max: {df_video['pessoas'].max()}"
                            )
                        
                        with col2:
                            st.metric(
                                " Média de Celulares", 
                                f"{df_video['celulares'].mean():.1f}",
                                f"Max: {df_video['celulares'].max()}"
                            )
                        
                        with col3:
                            peak_detections = df_video['total_deteccoes'].max()
                            st.metric(
                                " Pico de Detecções",
                                peak_detections,
                                f"Frame: {df_video.loc[df_video['total_deteccoes'].idxmax(), 'frame']}"
                            )
                        
                        # Histogramas
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_hist_people = px.histogram(
                                df_video, 
                                x='pessoas',
                                title='Distribuição - Pessoas por Frame',
                                nbins=20
                            )
                            st.plotly_chart(fig_hist_people, use_container_width=True)
                        
                        with col2:
                            fig_hist_phones = px.histogram(
                                df_video,
                                x='celulares', 
                                title='Distribuição - Celulares por Frame',
                                nbins=20
                            )
                            st.plotly_chart(fig_hist_phones, use_container_width=True)
                        
                        # Opção de download dos resultados
                        if save_results:
                            csv_video = df_video.to_csv(index=False)
                            st.download_button(
                                label="📥 Baixar Resultados do Vídeo (CSV)",
                                data=csv_video,
                                file_name=f"analise_video_{uploaded_video.name}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.error(" Erro no processamento do vídeo.")
                        
                except Exception as e:
                    st.error(f" Erro durante o processamento: {e}")
                
                finally:
                    # Limpar arquivo temporário
                    if os.path.exists(video_path):
                        os.unlink(video_path)
    
    # ==================== TAB 3: ESTATÍSTICAS ====================
    with tab3:
        st.header(" Estatísticas e Análises")
        
        # Informações do modelo
        model_info = detector.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Informações do Modelo")
            st.json(model_info)
        
        with col2:
            st.subheader(" Dados de Exemplo")
            
            # Criar dados de exemplo para demonstração
            from utils.data_utils import create_sample_data
            sample_data = create_sample_data()
            
            if st.button(" Gerar Análise de Exemplo"):
                df_sample = pd.DataFrame(sample_data)
                
                # Gráfico de exemplo
                fig_sample = px.area(
                    df_sample,
                    x='frame',
                    y='total_deteccoes',
                    title='Exemplo: Total de Detecções por Frame',
                    color_discrete_sequence=['#ff6b6b']
                )
                st.plotly_chart(fig_sample, use_container_width=True)
                
                # Estatísticas de exemplo
                st.write("**Estatísticas dos Dados de Exemplo:**")
                st.write(df_sample.describe())
        
        # Processador de dados
        st.subheader(" Ferramentas de Análise")
        
        processor = DataProcessor()
        
        if st.button(" Validar Estrutura do Dataset"):
            validation = processor.validate_dataset_structure("./data")
            
            if validation["valido"]:
                st.success(" Estrutura do dataset válida!")
            else:
                st.error(" Problemas encontrados na estrutura:")
                for problema in validation["problemas"]:
                    st.write(f"- {problema}")
            
            if validation["estatisticas"]:
                st.write("**Estatísticas do Dataset:**")
                st.json(validation["estatisticas"])
    
    # ==================== TAB 4: SOBRE ====================
    with tab4:
        st.header("ℹ Sobre o Projeto")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Objetivo")
            st.write("""
            Este projeto desenvolve um sistema de detecção automática de pessoas 
            utilizando celulares em ambientes públicos ou privados, utilizando 
            técnicas de visão computacional com YOLO (You Only Look Once).
            """)
            
            st.subheader(" Tecnologias")
            st.write("""
            - **YOLOv8** (Ultralytics) - Modelo de detecção
            - **OpenCV** - Processamento de imagens
            - **Streamlit** - Interface web
            - **Python** - Linguagem principal
            """)
            
            st.subheader(" Performance Esperada")
            st.write("""
            - **Precisão**: ~85%
            - **Tempo de inferência**: <50ms por frame
            - **Classes detectadas**: 3 (pessoa, celular, pessoa_com_celular)
            """)
        
        with col2:
            st.subheader(" Equipe")
            st.write("""
            **Integrantes:**
            - Andre Akio Morita Osakawa (10340045)
            - André Franco Ranieri (10390470)
            - Felipe Mazzeo Barbosa (10402808)
            - Fernando Pegoraro Bilia (10402097)
            - Francesco Zangrandi Coppola (10403340)
            
            **Instituição:**
            Universidade Presbiteriana Mackenzie
            Faculdade de Computação e Informática
            
            **Disciplina:**
            Inteligência Artificial - 2025
            """)
            
            st.subheader(" Como Usar")
            st.write("""
            1. **Upload**: Carregue uma imagem ou vídeo
            2. **Configuração**: Ajuste a confiança na barra lateral
            3. **Processamento**: Aguarde a análise automática
            4. **Resultados**: Visualize as detecções e métricas
            """)
            
            st.subheader(" Suporte")
            st.write("""
            Para dúvidas ou problemas:
            - 📧 Email: suporte@projeto.com
            - 📖 GitHub: [Repositório do Projeto](#)
            - 🐛 Issues: [Reportar Problemas](#)
            """)

else:
    st.error(" Não foi possível carregar o detector. Verifique a instalação das dependências.")
    st.info("""
    **Para resolver este problema:**
    1. Certifique-se de que todas as dependências estão instaladas
    2. Execute: `pip install -r requirements.txt`
    3. Verifique se o modelo está disponível
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📱 Detector de Pessoas com Celular | Desenvolvido com usando Streamlit<br>
        Universidade Presbiteriana Mackenzie - 2025
    </div>
    """, 
    unsafe_allow_html=True
)