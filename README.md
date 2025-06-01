Detector de Pessoas com Celular - YOLO
Projeto Semestral de Inteligência Artificial
Universidade Presbiteriana Mackenzie
Integrantes do Grupo

10340045 - Andre Akio Morita Osakawa
10390470 - André Franco Ranieri
10402808 - Felipe Mazzeo Barbosa
10402097 - Fernando Pegoraro Bilia
10403340 - Francesco Zangrandi Coppola

# Sobre o Projeto
Este projeto teve como objetivo desenvolver um sistema inteligente capaz de detectar automaticamente pessoas que estão utilizando celulares em diferentes ambientes. Utilizamos técnicas avançadas de visão computacional, especificamente o modelo YOLO (You Only Look Once), para criar uma solução que identifica tanto pessoas quanto dispositivos móveis simultaneamente.
A motivação por trás do projeto surgiu da necessidade de monitorar o uso de dispositivos móveis em locais específicos, como escolas, bibliotecas, locais de trabalho ou eventos, onde pode ser necessário controlar ou quantificar o uso de celulares.

# O que Desenvolvemos
Nosso sistema é composto por várias partes que trabalham em conjunto:

1. Modelo de Detecção Inteligente
Criamos um detector baseado no YOLOv8 que consegue identificar três elementos principais: pessoas, celulares e especificamente pessoas usando celulares. O modelo foi treinado com milhares de imagens para garantir alta precisão.

2. Interface Web Interativa
Desenvolvemos uma aplicação web usando Streamlit que permite qualquer pessoa usar nosso sistema de forma simples e intuitiva. Você pode carregar uma imagem ou vídeo e ver os resultados imediatamente.

3. Análise Completa de Resultados
O sistema não apenas detecta os objetos, mas também fornece estatísticas detalhadas, contagens precisas e níveis de confiança para cada detecção.

4. Artigo: 
Um artigo que apresenta o desenvolvimento, funcionamento e resultados do código. Esse pode ser acessado na pasta Docs--> Artigo

Tecnologias que Utilizamos
Para tornar este projeto realidade, trabalhamos com uma combinação de tecnologias modernas:

YOLOv8 da Ultralytics: A base do nosso sistema de detecção, conhecida por sua velocidade e precisão
OpenCV: Para processamento avançado de imagens e vídeos
PyTorch: O framework de inteligência artificial que alimenta todo o aprendizado do modelo
Streamlit: Para criar uma interface web bonita e funcional
Python: A linguagem que conecta tudo isso

# Como Usar Nosso Sistema
Preparando o Ambiente
Primeiro, você precisa configurar o ambiente em seu computador:
bash# Baixe o projeto do GitHub
git clone [https://github.com/ManoTilts/IA-pessoas-Cel.git](https://github.com/ManoTilts/IA-pessoas-Cel.git)
cd IA-pessoas-Cel

Crie um ambiente isolado para o projeto
python -m venv .venv

Ative o ambiente (no Windows)
.venv\Scripts\activate

Ou no Linux/Mac
source .venv/bin/activate

Instale todas as dependências necessárias
pip install -r requirements.txt
Executando a Aplicação Web
Para usar nossa interface web, simplesmente execute:
bashstreamlit run app.py
Isso abrirá uma página no seu navegador onde você pode:

Carregar suas próprias imagens ou vídeos
Ver os resultados da detecção em tempo real
Baixar as análises processadas
Ajustar configurações como sensibilidade da detecção

# Explorando o Código
Se você quiser entender como tudo funciona por baixo dos panos, abra nosso notebook principal:
bashjupyter notebook Projeto_semestral_Yolo.ipynb
Como Organizamos o Projeto
IA-pessoas-Cel/
├── Projeto_semestral_Yolo.ipynb    # Todo o código principal explicado passo a passo

├── app.py                          # Nossa aplicação web

├── requirements.txt                # Lista de todas as bibliotecas necessárias

├── README.md                       # Este arquivo que você está lendo

├── models/                         # Onde guardamos nossos modelos treinados

├── data/                           # Imagens e vídeos para teste

├── utils/                          # Código auxiliar organizado

└── docs/                           # Documentação técnica detalhada

# O que Nosso Sistema Consegue Fazer
Detecções Principais
Nosso sistema identifica com precisão pessoas, celulares e especificamente pessoas que estão usando celulares. Cada detecção vem com um nível de confiança, para você saber o quão certo o sistema está sobre cada identificação.
Interface Amigável
A aplicação web que criamos permite que qualquer pessoa, mesmo sem conhecimento técnico, use nosso sistema. Você pode carregar imagens nos formatos mais comuns (JPG, PNG, etc.) ou vídeos (MP4, AVI, etc.) e ver os resultados instantaneamente.
Análises Detalhadas
Além de apenas detectar objetos, nosso sistema fornece estatísticas completas: quantas pessoas foram encontradas, quantos celulares, gráficos de confiança das detecções, e muito mais.
Os Dados que Utilizamos
Para treinar nosso modelo, trabalhamos com diferentes fontes de dados:

Dataset COCO: 
Uma base de dados mundialmente reconhecida que já contém milhares de imagens de pessoas e celulares
Dataset Personalizado: Coletamos e anotamos nossas próprias imagens para casos específicos
Aproximadamente 5000 imagens: Cada uma cuidadosamente marcada com as localizações exatas de pessoas e celulares

Utilizamos o Roboflow, uma ferramenta profissional, para marcar precisamente onde cada objeto aparece nas imagens, permitindo que nosso modelo aprendesse efetivamente.
Nossa Metodologia
Preparação Cuidadosa dos Dados
Não bastava apenas ter muitas imagens. Precisamos organizá-las, marcá-las corretamente e criar variações para que o modelo pudesse aprender diferentes situações: pessoas em ambientes claros e escuros, celulares de diferentes tipos, pessoas em diferentes posições.

Treinamento Inteligente
Começamos com um modelo YOLO que já sabia identificar objetos gerais, e então o ensinamos especificamente sobre nossa tarefa. Foi como pegar um estudante que já conhece o básico e especializá-lo em nossa área específica.

Testes Rigorosos:
Testamos nosso sistema de várias formas: com imagens que ele nunca viu antes, com vídeos em tempo real, e medimos não apenas se ele acerta, mas quão confiante ele está em suas respostas.
Resultados que Conseguimos
Nossos testes mostraram resultados muito promissores:

Modelo Pré-treinado:
Detecta pessoas com cerca de 90% de precisão
Identifica celulares com cerca de 85% de precisão
Processa mais de 30 imagens por segundo em um computador comum

Com Nosso Modelo Personalizado:
Identifica pessoas usando celulares com cerca de 85% de precisão
Funciona em tempo real para aplicações práticas

Como Desenvolvemos Tudo
Código Bem Organizado
Estruturamos nosso código seguindo as melhores práticas de programação. Cada parte tem uma função específica, está bem documentada e pode ser facilmente entendida e modificada.
Tratamento de Problemas
Preparamos o sistema para lidar com situações inesperadas: imagens corrompidas, formatos não suportados, problemas de conexão, etc. O usuário sempre recebe feedback claro sobre o que está acontecendo.
Interface Pensada no Usuário
Nossa aplicação web foi desenvolvida pensando em pessoas que podem não ter conhecimento técnico. As instruções são claras, os resultados são apresentados de forma visual e intuitiva.
Aplicações Práticas
Este sistema pode ser utilizado em diversos cenários reais:
Educação: Monitorar o uso de celulares em salas de aula ou bibliotecas
Segurança: Identificar uso inadequado de dispositivos em áreas restritas
Pesquisa: Estudar padrões de comportamento e uso de tecnologia
Eventos: Controlar o uso de celulares durante apresentações ou cerimônias
Aprendizados e Desafios
Durante o desenvolvimento, enfrentamos vários desafios interessantes:

Qualidade dos Dados: Descobrimos que a qualidade das imagens de treinamento é mais importante que a quantidade
Balanceamento: Precisamos garantir que o modelo visse exemplos suficientes de todas as situações possíveis
Performance: Equilibrar precisão com velocidade de processamento foi um desafio constante
Interface de Usuário: Fazer com que uma tecnologia complexa fosse acessível para qualquer pessoa

# Trabalhos Futuros
Temos várias ideias para melhorar ainda mais o sistema:

Detecção de diferentes tipos de dispositivos (tablets, laptops)
Análise de comportamento (pessoa apenas segurando vs. usando ativamente)
Integração com câmeras de segurança em tempo real
Versão mobile do aplicativo
Análise de múltiplas pessoas interagindo com dispositivos

# Contato e Suporte
Este projeto foi desenvolvido como parte de nossa formação acadêmica, mas estamos orgulhosos do resultado e dispostos a compartilhar conhecimento.
Para dúvidas ou sugestões:

Use o sistema de issues do GitHub para problemas técnicos
Consulte nossa documentação detalhada na pasta docs/
Entre em contato através dos emails institucionais para questões acadêmicas

Agradecimentos
Gostaríamos de agradecer à Universidade Presbiteriana Mackenzie pela oportunidade de desenvolver este projeto, aos professores pelo direcionamento técnico, e à comunidade open source pelas ferramentas incríveis que tornaram este trabalho possível.
Este projeto representa não apenas um requisito acadêmico, mas uma exploração genuína de como a inteligência artificial pode resolver problemas do mundo real de forma acessível e prática.

Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática
Projeto Semestral - Inteligência Artificial - 2025
