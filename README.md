# E-commerce Analytics Dashboard

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
git clone https://github.com/seu-usuario/ecommerce-analytics.git
cd ecommerce-analytics
```

2. **Crie o ambiente virtual e instale as dependências:**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. **Execute o dashboard:**
```bash
streamlit run app.py
```

O pipeline ETL será executado automaticamente ao abrir o dashboard, gerando dados sintéticos, processando, treinando o modelo e exibindo as análises.

## Funcionalidades do Dashboard

- **Resumo Executivo**: Métricas gerais do e-commerce (clientes, receita, churn, ticket médio, etc).
- **Análise de Churn**: Visualização da taxa de churn, principais cidades, faixas etárias e distribuição de gastos.
- **Segmentação de Clientes**: Classificação automática em VIP, Fiel, Regular e Em Risco.
- **Recomendações Estratégicas**: Sugestões de ações para retenção e recuperação de receita.
- **Impacto Financeiro**: Cálculo da receita em risco devido ao churn.
- **Visualizações Interativas**: Gráficos dinâmicos com Plotly e Streamlit.
- **Modelo Preditivo**: Upload de CSV ou preenchimento manual para prever churn de novos clientes.
- **Pergunte ao Modelo**: Interface para perguntas ao modelo preditivo, com resultados instantâneos.

## Exemplo de CSV para Upload

O arquivo CSV para previsão de churn deve conter as seguintes colunas:

```csv
age,gender,city,premium_member,total_spent,avg_order_value,total_orders,categories_bought,discount_usage,avg_rating,days_since_last_order,customer_lifetime_days
30,F,São Paulo,1,500.0,100.0,5,3,0.2,4.5,20,400
```

Você pode baixar um exemplo em [`examples/exemplo_churn.csv`](examples/exemplo_churn.csv) (crie este arquivo se desejar).

## Testes

Execute os testes unitários para garantir a integridade do pipeline:
```bash
pytest tests/
```

## Principais Tecnologias

- **Python 3.8+**
- **Streamlit** (dashboard web)
- **Pandas, Numpy** (manipulação de dados)
- **scikit-learn** (machine learning)
- **Plotly** (visualizações interativas)
- **SQLite** (armazenamento local)
- **Pytest** (testes unitários)

## Screenshots

Adicione imagens reais do dashboard em `docs/images/` e inclua exemplos aqui:

![Dashboard - Resumo](docs/images/dashboard_resumo.png)
![Dashboard - Churn](docs/images/dashboard_churn.png)

## Licença

Este projeto está sob a licença MIT.

## Como Funciona

1. **Geração de Dados Sintéticos**: O pipeline cria automaticamente dados fictícios de clientes, produtos e pedidos, simulando um cenário realista de e-commerce.
2. **ETL (Extract, Transform, Load)**:
   - **Extract**: Extrai os dados do banco SQLite.
   - **Transform**: Realiza limpeza, feature engineering e cálculo de métricas por cliente.
   - **Load**: Salva os dados processados em uma tabela analítica.
3. **Análise Exploratória**: O dashboard apresenta estatísticas, segmentação de clientes, análise de churn e visualizações interativas (gráficos de pizza, barras, boxplot, etc).
4. **Geração de Insights**: Segmentação automática dos clientes, recomendações estratégicas baseadas em regras de negócio e cálculo do impacto financeiro do churn.
5. **Modelo Preditivo de Churn**: Um modelo Random Forest é treinado com as principais features do cliente. O dashboard permite fazer perguntas ao modelo, seja por upload de CSV ou preenchimento manual dos dados, para prever a probabilidade de churn de novos clientes.
6. **Dashboard Web Interativo**: Tudo é apresentado em uma interface moderna e responsiva via Streamlit, facilitando a exploração dos dados, métricas, recomendações e previsões.

---

Desenvolvido para portfólio de Data Science e Engenharia de Dados. Sinta-se à vontade para contribuir, adaptar ou usar como referência!

---

**Destaques:**
- Estrutura profissional e modular
- Código limpo, testado e documentado
- Pronto para portfólio, entrevistas e demonstrações
- Fácil de expandir para dados reais ou outros modelos
