import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.etl import EcommerceETL
from src.models.churn_predictor import ChurnPredictor
from src.config import Config
from sklearn.preprocessing import LabelEncoder

# Configuração da página
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Título principal
st.title("🛍️ E-commerce Analytics Dashboard")

# Remover cache da conexão e garantir conexão local em cada acesso
import sqlite3

def get_connection():
    return sqlite3.connect('ecommerce_portfolio.db', check_same_thread=False)

# Inicializar ETL
@st.cache_resource
def get_etl():
    etl = EcommerceETL('ecommerce_portfolio.db')
    etl.run_complete_pipeline()  # Isso garante que o scaler e o modelo estejam treinados
    return etl

etl = get_etl()
df = etl.data['transformed']

# Sidebar com métricas gerais
st.sidebar.header("📊 Métricas Gerais")
st.sidebar.metric("Total de Clientes", f"{len(df):,}")
st.sidebar.metric("Taxa de Churn", f"{df['is_churn'].mean():.1%}")
st.sidebar.metric("Receita Total", f"R$ {df['total_spent'].sum():,.2f}")
st.sidebar.metric("Ticket Médio", f"R$ {df['avg_order_value'].mean():.2f}")
st.sidebar.metric("% Clientes Premium", f"{df['premium_member'].mean():.1%}")

# Layout principal com tabs
tab1, tab2, tab3 = st.tabs(["📈 Análise de Churn", "👥 Segmentação", "💰 Impacto Financeiro"])

with tab1:
    st.header("Análise de Churn")
    
    # Layout em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de pizza do churn
        fig_pie = px.pie(
            names=['Ativo', 'Churn'],
            values=df['is_churn'].value_counts().values,
            title='Distribuição de Churn',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # Churn por cidade
        churn_by_city = df.groupby('city')['is_churn'].mean().sort_values(ascending=False)
        fig_city = px.bar(
            x=churn_by_city.index,
            y=churn_by_city.values,
            title='Taxa de Churn por Cidade',
            labels={'x': 'Cidade', 'y': 'Taxa de Churn'}
        )
        st.plotly_chart(fig_city, use_container_width=True)
    
    # Churn por idade
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    churn_by_age = df.groupby('age_group')['is_churn'].mean()
    fig_age = px.bar(
        x=churn_by_age.index,
        y=churn_by_age.values,
        title='Taxa de Churn por Faixa Etária',
        labels={'x': 'Faixa Etária', 'y': 'Taxa de Churn'}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with tab2:
    st.header("Segmentação de Clientes")
    
    # Criar segmentação
    df['segment'] = 'Regular'
    df.loc[(df['premium_member'] == 1) & (df['total_spent'] > df['total_spent'].quantile(0.75)), 'segment'] = 'VIP'
    df.loc[(df['total_orders'] > 10) & (df['is_churn'] == 0), 'segment'] = 'Fiel'
    df.loc[df['is_churn'] == 1, 'segment'] = 'Em Risco'
    
    # Análise por segmento
    segment_analysis = df.groupby('segment').agg({
        'customer_id': 'count',
        'total_spent': 'mean',
        'is_churn': 'mean'
    }).round(2)
    
    # Gráfico de barras empilhadas
    fig_segment = px.bar(
        segment_analysis,
        y=segment_analysis.index,
        x='customer_id',
        title='Distribuição de Clientes por Segmento',
        labels={'customer_id': 'Número de Clientes', 'segment': 'Segmento'}
    )
    st.plotly_chart(fig_segment, use_container_width=True)
    
    # Métricas por segmento
    col1, col2, col3, col4 = st.columns(4)
    for segment, metrics in segment_analysis.iterrows():
        with locals()[f"col{list(segment_analysis.index).index(segment) + 1}"]:
            st.metric(
                label=segment,
                value=f"{metrics['customer_id']} clientes",
                delta=f"R$ {metrics['total_spent']:.2f} média"
            )

with tab3:
    st.header("Impacto Financeiro")
    
    # Receita mensal
    with get_connection() as conn:
        df_orders = pd.read_sql_query(
            """
            SELECT 
                strftime('%Y-%m', order_date) as month,
                SUM(quantity * price * (1 - discount_applied * 0.1)) as revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE order_date >= '2023-01-01'
            GROUP BY month
            ORDER BY month
            """,
            conn
        )
    
    fig_revenue = px.line(
        df_orders,
        x='month',
        y='revenue',
        title='Receita Mensal',
        labels={'month': 'Mês', 'revenue': 'Receita (R$)'}
    )
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Impacto do churn
    col1, col2 = st.columns(2)
    
    with col1:
        at_risk_customers = df[df['is_churn'] == 1]
        potential_revenue = at_risk_customers['total_spent'].sum()
        
        st.metric(
            "Receita em Risco",
            f"R$ {potential_revenue:,.2f}",
            f"{len(at_risk_customers)} clientes"
        )
    
    with col2:
        avg_revenue_loss = potential_revenue / len(at_risk_customers) if len(at_risk_customers) > 0 else 0
        st.metric(
            "Perda Média por Cliente",
            f"R$ {avg_revenue_loss:.2f}",
            "valor a recuperar"
        )

# Modelo preditivo
st.header("🤖 Modelo Preditivo")

# Feature importance
with get_connection() as conn:
    feature_importance = pd.read_sql_query(
        "SELECT * FROM feature_importance ORDER BY importance DESC LIMIT 5",
        conn
    )

fig_importance = px.bar(
    feature_importance,
    x='feature',
    y='importance',
    title='Importância das Features',
    labels={'feature': 'Feature', 'importance': 'Importância'}
)
st.plotly_chart(fig_importance, use_container_width=True)

# Recomendações
st.header("💡 Recomendações Estratégicas")

def gerar_recomendacoes(df):
    recs = []
    churn_rate = df['is_churn'].mean()
    premium_churn = df[df['premium_member'] == 1]['is_churn'].mean()
    regular_churn = df[df['premium_member'] == 0]['is_churn'].mean()
    ticket_medio = df['avg_order_value'].mean()
    clientes_inativos = (df['days_since_last_order'] > 90).sum()
    if churn_rate > 0.2:
        recs.append("Campanhas de reativação para clientes inativos (churn elevado)")
    if premium_churn > 0.15:
        recs.append("Ofereça benefícios exclusivos para clientes premium (churn premium alto)")
    if ticket_medio < 80:
        recs.append("Aposte em cross-sell e up-sell para aumentar o ticket médio")
    if regular_churn > 0.25:
        recs.append("Crie programa de fidelidade para clientes regulares")
    if clientes_inativos > 0:
        recs.append(f"Existem {clientes_inativos} clientes inativos há mais de 90 dias. Realize campanhas de reengajamento.")
    if not recs:
        recs.append("Continue monitorando os segmentos e invista em personalização de ofertas.")
    return recs

for i, rec in enumerate(gerar_recomendacoes(df), 1):
    st.info(f"{i}. {rec}")

# Modelo preditivo - Pergunte ao modelo
st.header("🤖 Faça perguntas ao modelo preditivo de churn")

with st.expander("Prever churn para novos clientes"):
    st.write("Você pode fazer upload de um arquivo CSV com dados de clientes ou preencher manualmente os dados abaixo para prever a probabilidade de churn.")
    
    uploaded_file = st.file_uploader("Faça upload de um arquivo CSV com as colunas corretas", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        # Preencher campos que faltam com valores padrão
        for col in [
            'age', 'gender', 'city', 'premium_member', 'total_spent', 'avg_order_value',
            'total_orders', 'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]:
            if col not in input_df.columns:
                input_df[col] = 0
        # Encoding manual para gender/city
        le_gender = LabelEncoder().fit(df['gender'])
        le_city = LabelEncoder().fit(df['city'])
        input_df['gender_encoded'] = le_gender.transform(input_df['gender'])
        input_df['city_encoded'] = le_city.transform(input_df['city'])
        features = [
            'age', 'gender_encoded', 'city_encoded', 'premium_member',
            'total_spent', 'avg_order_value', 'total_orders',
            'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]
        X_input = input_df[features]
        scaler = etl.scaler
        X_input_scaled = scaler.transform(X_input)
        model = etl.model
        churn_proba = model.predict_proba(X_input_scaled)[:, 1]
        input_df['prob_churn'] = churn_proba
        st.write("Resultados da previsão:")
        st.dataframe(input_df[['age', 'gender', 'city', 'prob_churn']])
    else:
        st.write("Ou preencha os dados manualmente:")
        manual_data = {}
        manual_data['age'] = st.number_input('Idade', min_value=18, max_value=80, value=30)
        manual_data['gender'] = st.selectbox('Gênero', options=df['gender'].unique())
        manual_data['city'] = st.selectbox('Cidade', options=df['city'].unique())
        manual_data['premium_member'] = st.selectbox('Premium?', options=[0, 1])
        manual_data['total_spent'] = st.number_input('Total gasto', min_value=0.0, value=100.0)
        manual_data['avg_order_value'] = st.number_input('Ticket médio', min_value=0.0, value=50.0)
        manual_data['total_orders'] = st.number_input('Total de pedidos', min_value=0, value=2)
        manual_data['categories_bought'] = st.number_input('Categorias compradas', min_value=0, value=1)
        manual_data['discount_usage'] = st.number_input('Uso de desconto (%)', min_value=0.0, max_value=1.0, value=0.2)
        manual_data['avg_rating'] = st.number_input('Rating médio', min_value=0.0, max_value=5.0, value=4.0)
        manual_data['days_since_last_order'] = st.number_input('Dias desde última compra', min_value=0, value=30)
        manual_data['customer_lifetime_days'] = st.number_input('Dias como cliente', min_value=0, value=365)
        # Encoding
        le_gender = LabelEncoder().fit(df['gender'])
        le_city = LabelEncoder().fit(df['city'])
        manual_data['gender_encoded'] = le_gender.transform([manual_data['gender']])[0]
        manual_data['city_encoded'] = le_city.transform([manual_data['city']])[0]
        features = [
            'age', 'gender_encoded', 'city_encoded', 'premium_member',
            'total_spent', 'avg_order_value', 'total_orders',
            'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]
        X_manual = pd.DataFrame([manual_data])[features]
        scaler = etl.scaler
        X_manual_scaled = scaler.transform(X_manual)
        model = etl.model
        churn_proba = model.predict_proba(X_manual_scaled)[0, 1]
        st.success(f"Probabilidade de churn: {churn_proba:.1%}")

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Python, Streamlit e Machine Learning")
