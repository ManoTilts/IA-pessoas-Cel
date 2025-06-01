# 🎥 Roteiro de Apresentação - Detector de Pessoas com Celular

**Duração:** 3 minutos  
**Apresentadores:** Andre Akio, André Franco, Felipe Mazzeo, Fernando Pegoraro e Francesco Zangrandi  
**Projeto:** Detecção de Pessoas com Celular usando YOLO

---

## 📋 Estrutura da Apresentação

### ⏱️ Cronograma (3 minutos)

| Tempo | Seção | Responsável | Conteúdo |
|-------|-------|-------------|----------|
| 0:00-0:45 | Introdução | Andre Akio | Apresentação do projeto, problema e objetivos |
| 0:45-1:30 | Tecnologias | André Franco | Stack tecnológica e arquitetura |
| 1:30-2:15 | Demonstração | Felipe | Interface Streamlit e funcionalidades |
| 2:15-2:45 | Aplicações | Fernando | Casos de uso e potencial prático |
| 2:45-3:00 | Conclusão | Francesco | Aprendizados e trabalhos futuros |

---

## 🎬 Roteiro Detalhado

### 📍 **INTRODUÇÃO (0:00 - 0:45) - Andre Akio**

**[SLIDE: Título do Projeto]**

> "Olá! Somos Andre Akio, André Franco, Felipe Mazzeo, Fernando Pegoraro e Francesco Zangrandi, estudantes da Universidade Presbiteriana Mackenzie. Apresentamos nosso projeto de IA: **Detector de Pessoas com Celular usando YOLO**."

**[SLIDE: Problema e Motivação]**

> "Com smartphones presentes em todos os lugares, surge a necessidade de monitorar automaticamente seu uso em diversos contextos - escolas, empresas, eventos e pesquisas comportamentais."

**[SLIDE: Objetivos]**

> "Desenvolvemos um sistema inteligente que detecta pessoas, celulares e especificamente pessoas usando celulares, utilizando visão computacional com YOLOv8. O foco foi criar uma solução prática e acessível."

---

### 🛠️ **TECNOLOGIAS (0:45 - 1:30) - André Franco**

**[SLIDE: Stack Tecnológica]**

> "Nossa solução combina tecnologias modernas de IA. Utilizamos **YOLOv8** da Ultralytics como base, conhecido por sua velocidade e eficiência em detecção de objetos em tempo real."

**[SLIDE: Arquitetura do Sistema]**

> "A arquitetura é modular: entrada de dados, processamento com YOLO, e interface web. Usamos **PyTorch** para deep learning, **OpenCV** para processamento de imagens, e **Streamlit** para a interface."

**[SLIDE: Pipeline de Desenvolvimento]**

> "Começamos com um modelo YOLOv8 pré-treinado no dataset COCO, adaptamos com transfer learning para nosso domínio específico, e criamos um dataset personalizado com anotações precisas usando Roboflow."

---

### 🖥️ **DEMONSTRAÇÃO (1:30 - 2:15) - Felipe**

**[TELA: Aplicação Streamlit]**

> "Criamos uma aplicação web completa e intuitiva. Vou mostrar as principais funcionalidades do nosso sistema:"

**[DEMO: Interface Principal]**

> "A interface permite upload simples de imagens nos formatos mais comuns. O processamento é automático e os resultados aparecem instantaneamente com bounding boxes coloridos."

**[DEMO: Configurações Avançadas]**

> "Os usuários podem ajustar configurações como limiar de confiança, escolher entre modelo base ou customizado, e personalizar a visualização das detecções."

**[DEMO: Análise de Resultados]**

> "O sistema fornece estatísticas detalhadas: contagem de pessoas, celulares detectados, pessoas usando celulares, e informações de confiança para cada detecção. Para vídeos, geramos gráficos temporais mostrando a evolução das detecções."

---

### 🎯 **APLICAÇÕES (2:15 - 2:45) - Fernando**

**[SLIDE: Casos de Uso Práticos]**

> "Nosso sistema tem aplicações em múltiplas áreas. Na **educação**, pode monitorar uso de celulares em salas de aula. Em **segurança**, controla áreas onde dispositivos são restritos."

**[SLIDE: Pesquisa e Análise]**

> "Para **pesquisa comportamental**, oferece dados objetivos sobre padrões de uso de dispositivos. Em **ambientes corporativos**, pode analisar distração ou engajamento em reuniões."

**[SLIDE: Características Diferenciais]**

> "Nosso diferencial é a **facilidade de uso** - qualquer pessoa pode usar sem conhecimento técnico. É **open source**, completamente documentado, e funciona tanto com imagens quanto vídeos."

---

### 🔮 **CONCLUSÃO (2:45 - 3:00) - Francesco**

**[SLIDE: Aprendizados]**

> "Este projeto nos ensinou muito sobre visão computacional prática. Aprendemos que transfer learning é eficaz, a qualidade dos dados é fundamental, e interfaces amigáveis são cruciais para adoção."

**[SLIDE: Próximos Passos]**

> "Como evolução, planejamos implementar tracking de objetos, expansão para detecção de outros dispositivos, e integração com sistemas IoT para monitoramento contínuo."

**[SLIDE: Agradecimentos]**

> "Agradecemos à Universidade Mackenzie e nossos professores. O projeto completo está no GitHub para a comunidade usar e contribuir. Demonstramos que IA pode resolver problemas reais de forma acessível!"

---

## 🎯 **Pontos-Chave a Destacar**

### ✅ **Aspectos Técnicos**
- Transfer learning com YOLOv8
- Interface web responsiva e intuitiva
- Processamento de múltiplos formatos
- Arquitetura modular e escalável

### ✅ **Diferenciais do Projeto**
- Sistema completo end-to-end
- Interface acessível para não-técnicos
- Documentação completa e detalhada
- Projeto open source disponível

### ✅ **Valor Prático**
- Aplicações em múltiplos domínios
- Ferramenta pronta para uso
- Base sólida para pesquisas futuras
- Demonstração de IA aplicada



## 🎯 **Distribuição de Responsabilidades**

### **Andre Akio (Introdução):**
- Contextualização do problema
- Apresentação dos objetivos
- Motivação do projeto

### **André Franco (Tecnologias):**
- Stack tecnológica utilizada
- Arquitetura do sistema
- Pipeline de desenvolvimento

### **Felipe (Demonstração):**
- Interface Streamlit
- Funcionalidades principais
- Experiência do usuário

### **Fernando (Aplicações):**
- Casos de uso práticos
- Valor do projeto
- Diferencial competitivo

### **Francesco (Conclusão):**
- Aprendizados obtidos
- Trabalhos futuros
- Agradecimentos

---

## 🎯 **Mensagem Principal**

> "Demonstramos como tecnologias modernas de IA podem ser aplicadas para resolver problemas reais de forma prática e acessível. Nosso sistema combina eficiência técnica com usabilidade, criando uma ferramenta valiosa para diversos contextos profissionais e acadêmicos."

---

## 📱 **Call to Action**

> "Convidamos todos a explorar nosso projeto no GitHub, testar a aplicação web e contribuir com melhorias. Acreditamos que projetos open source impulsionam a inovação e democratizam o acesso à tecnologia."

---

**🎬 Sucesso na apresentação! 🎬** 