# E-commerce Analytics & Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![SQLite](https://img.shields.io/badge/SQLite-3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green)
![Testes](https://img.shields.io/badge/tests-passing-brightgreen)

## Visão Geral

Este projeto é um pipeline completo de análise de dados para e-commerce, com geração de dados sintéticos, ETL, análise exploratória, visualização, predição de churn e dashboard interativo. O objetivo é fornecer uma solução profissional, modular e pronta para portfólio/GitHub, demonstrando habilidades de Engenharia de Dados, Data Science e Machine Learning.

- **Geração de dados sintéticos**: simula clientes, produtos e pedidos realistas.
- **Pipeline ETL**: extração, transformação, limpeza, feature engineering e carga em banco SQLite.
- **Análise exploratória**: métricas, segmentação, churn, gráficos e insights de negócio.
- **Modelo preditivo de churn**: Random Forest treinado com features relevantes, pronto para responder perguntas.
- **Dashboard web (Streamlit)**: interface moderna, visualizações interativas, upload de CSV e perguntas ao modelo.
- **Testes unitários**: cobertura do pipeline e modelo.
- **Documentação e exemplos**: instruções, exemplos de uso, estrutura profissional.

## Estrutura do Projeto

```
ecommerce-analytics/
│
├── app.py                 # Dashboard Streamlit (principal)
├── requirements.txt       # Dependências do projeto
├── pyproject.toml         # Configuração de formatação e build
│
├── src/                  # Código fonte
│   ├── etl/              # Pipeline ETL (pipeline.py)
│   ├── models/           # Modelos de ML (churn_predictor.py)
│   └── utils.py          # Utilitários
│
├── data/                 # Dados
│   ├── raw/              # Dados brutos (mantém .gitkeep)
│   └── processed/        # Dados processados (mantém .gitkeep)
│
├── models/               # Modelos treinados (opcional)
│   └── saved_models/     # Modelos salvos (mantém .gitkeep)
│
├── tests/                # Testes unitários (test_etl.py)
│
├── docs/                 # Documentação e imagens
│   └── images/           # Screenshots e ilustrações
│
├── examples/             # Exemplos de uso (run_pipeline.py)
└── README.md             # Documentação principal
```

## Como Executar

1. **Clone o repositório:**
```bash
git clone https://github.com/juanmmendes/ecommerce_predicao.git
cd ecommerce_predicao
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Como Usar

1. Execute o pipeline ETL e treine o modelo:
```bash
python src/etl/pipeline.py
```

2. Inicie o dashboard:
```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

```
ecommerce_portfolio/
│
├── app.py                 # Dashboard Streamlit (principal)
├── ecommerce_etl.py      # Arquivo ETL legado
├── requirements.txt       # Dependências do projeto
├── pyproject.toml        # Configuração do projeto
│
├── src/                  # Código fonte
│   ├── __init__.py
│   ├── config.py        # Configurações do sistema
│   ├── utils.py         # Funções utilitárias
│   ├── etl/             # Pipeline ETL
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── models/          # Modelos ML
│       └── churn_predictor.py
│
├── data/                # Dados
│   ├── raw/            # Dados brutos
│   └── processed/      # Dados processados
│
├── models/             # Modelos treinados
│   └── saved_models/   # Modelos salvos
│
├── tests/             # Testes unitários
│   └── test_etl.py   # Testes do pipeline
│
├── docs/              # Documentação
│   └── images/       # Screenshots e imagens
│       ├── ex1.png
│       └── ex2.png
│
└── examples/          # Exemplos de uso
    └── run_pipeline.py
```

## 📊 Funcionalidades do Dashboard

- **Análise de Churn**: Visualização detalhada das taxas de churn
- **Segmentação de Clientes**: Análise por segmentos (VIP, Regular, Em Risco)
- **Previsões em Tempo Real**: Modelo de ML para prever churn
- **Insights Automatizados**: Recomendações baseadas em dados

## 🤖 Modelo de Machine Learning

- Algoritmo: Random Forest Classifier
- Features principais:
  - Tempo desde última compra
  - Total gasto
  - Frequência de compras
  - Categorias compradas
  - Rating médio

## 📈 Performance do Modelo

- AUC-ROC Score: 1.000
- Precisão: 100%
- Recall: 100%
- F1-Score: 100%

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia as [diretrizes de contribuição](CONTRIBUTING.md) primeiro.

## 📝 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## 📧 Contato

Juan Mendes - [juan.zx016@gmail.com](mailto:juan.zx016@gmail.com)

Project Link: [https://github.com/juanmmendes/ecommerce_predicao](https://github.com/juanmmendes/ecommerce_predicao)
