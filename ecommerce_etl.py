# ETL Pipeline com Análise Preditiva - E-commerce Analytics
# Projeto completo de análise de dados com predição de churn de clientes

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class EcommerceETL:
    """
    Pipeline completo de ETL para análise de e-commerce
    Inclui extração, transformação, análise e predição de churn
    """
    
    def __init__(self, db_name='ecommerce_analytics.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.data = {}
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_sample_data(self):
        """Gera dados de exemplo simulando um e-commerce real"""
        print("🔄 Gerando dados de exemplo...")
        
        # Configurações
        n_customers = 5000
        n_products = 200
        n_orders = 15000
        
        # Dados de clientes
        np.random.seed(42)
        customers = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(35, 12, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'city': np.random.choice([
                'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 
                'Salvador', 'Fortaleza', 'Brasília', 'Curitiba'
            ], n_customers),
            'registration_date': pd.date_range(
                start='2022-01-01', end='2024-01-01', periods=n_customers
            ),
            'premium_member': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
        })
        
        # Produtos
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
        products = pd.DataFrame({
            'product_id': range(1, n_products + 1),
            'category': np.random.choice(categories, n_products),
            'price': np.random.lognormal(3, 1, n_products).round(2),
            'rating': np.random.normal(4, 0.5, n_products).clip(1, 5).round(1)
        })
        
        # Pedidos (com lógica de negócio)
        orders_data = []
        for _ in range(n_orders):
            customer_id = np.random.randint(1, n_customers + 1)
            customer_age = customers[customers['customer_id'] == customer_id]['age'].iloc[0]
            is_premium = customers[customers['customer_id'] == customer_id]['premium_member'].iloc[0]
            
            # Lógica: clientes mais velhos e premium compram mais
            base_prob = 0.3
            if customer_age > 40:
                base_prob += 0.2
            if is_premium:
                base_prob += 0.3
                
            if np.random.random() < base_prob:
                product_id = np.random.randint(1, n_products + 1)
                quantity = np.random.poisson(2) + 1
                
                orders_data.append({
                    'order_id': len(orders_data) + 1,
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'quantity': quantity,
                    'order_date': np.random.choice(
                        pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
                    ),
                    'discount_applied': np.random.choice([0, 1], p=[0.6, 0.4])
                })
        
        orders = pd.DataFrame(orders_data)
        
        # Salvar no banco
        customers.to_sql('customers', self.conn, if_exists='replace', index=False)
        products.to_sql('products', self.conn, if_exists='replace', index=False)
        orders.to_sql('orders', self.conn, if_exists='replace', index=False)
        
        print(f"✅ Dados gerados: {len(customers)} clientes, {len(products)} produtos, {len(orders)} pedidos")
        
    def extract_data(self):
        """Extração: Carrega dados do banco SQL"""
        print("📥 Extraindo dados do banco...")
        
        # Query complexa joining as tabelas
        query = """
        SELECT 
            c.customer_id,
            c.age,
            c.gender,
            c.city,
            c.registration_date,
            c.premium_member,
            o.order_id,
            o.product_id,
            o.quantity,
            o.order_date,
            o.discount_applied,
            p.category,
            p.price,
            p.rating,
            (o.quantity * p.price * (1 - o.discount_applied * 0.1)) as total_value
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN products p ON o.product_id = p.product_id
        """
        
        self.data['raw'] = pd.read_sql_query(query, self.conn)
        print(f"✅ Extraídos {len(self.data['raw'])} registros")
        
    def transform_data(self):
        """Transformação: Limpeza e feature engineering"""
        print("🔄 Transformando dados...")
        
        df = self.data['raw'].copy()
          # Converter datas (ignorando erros e usando formato ISO)
        df['registration_date'] = pd.to_datetime(df['registration_date'], format='mixed')
        df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')
        
        # Remover idades inválidas
        df = df[(df['age'] >= 18) & (df['age'] <= 80)]
        
        # Feature Engineering - Métricas por cliente
        customer_metrics = df.groupby('customer_id').agg({
            'total_value': ['sum', 'mean', 'count'],
            'order_date': ['min', 'max'],
            'category': lambda x: x.nunique(),
            'discount_applied': 'mean',
            'rating': 'mean'
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = [
            'total_spent', 'avg_order_value', 'total_orders',
            'first_order', 'last_order', 'categories_bought',
            'discount_usage', 'avg_rating'
        ]
        
        # Calcular dias desde última compra
        reference_date = pd.to_datetime('2024-01-31')
        customer_metrics['days_since_last_order'] = (
            reference_date - customer_metrics['last_order']
        ).dt.days
        
        # Merge com dados do cliente
        customer_info = df[['customer_id', 'age', 'gender', 'city', 
                          'registration_date', 'premium_member']].drop_duplicates()
        
        final_df = customer_info.merge(customer_metrics, on='customer_id', how='left')
        
        # Definir churn (cliente inativo há mais de 90 dias)
        final_df['is_churn'] = (final_df['days_since_last_order'] > 90).astype(int)
        
        # Tratar valores nulos (clientes sem pedidos)
        final_df = final_df.fillna({
            'total_spent': 0,
            'avg_order_value': 0,
            'total_orders': 0,
            'categories_bought': 0,
            'discount_usage': 0,
            'avg_rating': 0,
            'days_since_last_order': 365,
            'is_churn': 1
        })
        
        # Calcular tempo como cliente
        final_df['customer_lifetime_days'] = (
            reference_date - final_df['registration_date']
        ).dt.days
        
        self.data['transformed'] = final_df
        print(f"✅ Transformação concluída: {len(final_df)} clientes processados")
        
    def load_data(self):
        """Load: Salva dados transformados"""
        print("💾 Carregando dados transformados...")
        
        self.data['transformed'].to_sql(
            'customer_analytics', self.conn, if_exists='replace', index=False
        )
        print("✅ Dados carregados na tabela customer_analytics")
        
    def analyze_data(self):
        """Análise exploratória dos dados"""
        print("📊 Iniciando análise exploratória...")
        
        df = self.data['transformed']
        
        # Estatísticas básicas
        print("\n=== RESUMO EXECUTIVO ===")
        print(f"Total de clientes: {len(df):,}")
        print(f"Taxa de churn: {df['is_churn'].mean():.1%}")
        print(f"Receita total: R$ {df['total_spent'].sum():,.2f}")
        print(f"Ticket médio: R$ {df['avg_order_value'].mean():.2f}")
        print(f"Clientes premium: {df['premium_member'].mean():.1%}")
        
        # Análise por segmento
        print("\n=== ANÁLISE POR SEGMENTO ===")
        churn_by_premium = df.groupby('premium_member')['is_churn'].mean()
        print(f"Churn - Clientes regulares: {churn_by_premium[0]:.1%}")
        print(f"Churn - Clientes premium: {churn_by_premium[1]:.1%}")
        
        churn_by_city = df.groupby('city')['is_churn'].mean().sort_values(ascending=False)
        print(f"\nCidades com maior churn:")
        for city, rate in churn_by_city.head(3).items():
            print(f"  {city}: {rate:.1%}")
            
        # Criar visualizações
        self._create_visualizations(df)
        
    def _create_visualizations(self, df):
        """Cria visualizações dos dados"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dashboard E-commerce Analytics', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de Churn
        churn_counts = df['is_churn'].value_counts()
        axes[0,0].pie(churn_counts.values, 
                     labels=['Ativo', 'Churn'], 
                     autopct='%1.1f%%',
                     colors=['#2ecc71', '#e74c3c'])
        axes[0,0].set_title('Distribuição de Churn')
        
        # 2. Receita por mês
        df_orders = pd.read_sql_query("""
            SELECT 
                strftime('%Y-%m', order_date) as month,
                SUM(quantity * price * (1 - discount_applied * 0.1)) as revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE order_date >= '2023-01-01'
            GROUP BY month
            ORDER BY month
        """, self.conn)
        
        axes[0,1].plot(range(len(df_orders)), df_orders['revenue'])
        axes[0,1].set_title('Receita Mensal')
        axes[0,1].set_ylabel('Receita (R$)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Churn por idade
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        churn_by_age = df.groupby('age_group')['is_churn'].mean()
        axes[1,0].bar(range(len(churn_by_age)), churn_by_age.values)
        axes[1,0].set_title('Taxa de Churn por Faixa Etária')
        axes[1,0].set_xticks(range(len(churn_by_age)))
        axes[1,0].set_xticklabels(churn_by_age.index)
        axes[1,0].set_ylabel('Taxa de Churn')
        
        # 4. Valor gasto vs Churn
        axes[1,1].boxplot([df[df['is_churn']==0]['total_spent'], 
                          df[df['is_churn']==1]['total_spent']])
        axes[1,1].set_title('Distribuição de Gastos por Status')
        axes[1,1].set_xticklabels(['Ativo', 'Churn'])
        axes[1,1].set_ylabel('Total Gasto (R$)')
        
        plt.tight_layout()
        plt.show()
        
    def train_predictive_model(self):
        """Treina modelo de predição de churn"""
        print("🤖 Treinando modelo preditivo...")
        
        df = self.data['transformed'].copy()
        
        # Preparar features
        # Encoding categóricas
        le_gender = LabelEncoder()
        le_city = LabelEncoder()
        
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['city_encoded'] = le_city.fit_transform(df['city'])
        
        # Selecionar features
        features = [
            'age', 'gender_encoded', 'city_encoded', 'premium_member',
            'total_spent', 'avg_order_value', 'total_orders',
            'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]
        
        X = df[features]
        y = df['is_churn']
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelo
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=20
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n=== PERFORMANCE DO MODELO ===")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== IMPORTÂNCIA DAS FEATURES ===")
        for _, row in feature_importance.head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
            
        # Salvar modelo e resultados
        self._save_model_results(feature_importance, y_test, y_pred, y_pred_proba)
        
    def _save_model_results(self, feature_importance, y_test, y_pred, y_pred_proba):
        """Salva resultados do modelo"""
        # Salvar feature importance
        feature_importance.to_sql('feature_importance', self.conn, 
                                if_exists='replace', index=False)
        
        # Salvar predições
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba
        })
        results_df.to_sql('model_predictions', self.conn, 
                         if_exists='replace', index=False)
        
    def generate_insights(self):
        """Gera insights de negócio"""
        print("\n" + "="*50)
        print("🎯 INSIGHTS DE NEGÓCIO")
        print("="*50)
        
        df = self.data['transformed']
        
        # Segmentação de clientes
        df['segment'] = 'Regular'
        df.loc[(df['premium_member'] == 1) & (df['total_spent'] > df['total_spent'].quantile(0.75)), 'segment'] = 'VIP'
        df.loc[(df['total_orders'] > 10) & (df['is_churn'] == 0), 'segment'] = 'Fiel'
        df.loc[df['is_churn'] == 1, 'segment'] = 'Em Risco'
        
        segment_analysis = df.groupby('segment').agg({
            'customer_id': 'count',
            'total_spent': 'mean',
            'is_churn': 'mean'
        }).round(2)
        
        print("\n1️⃣ SEGMENTAÇÃO DE CLIENTES:")
        for segment, data in segment_analysis.iterrows():
            print(f"   {segment}: {data['customer_id']} clientes | "
                  f"Gasto médio: R$ {data['total_spent']:.2f} | "
                  f"Taxa churn: {data['is_churn']:.1%}")
        
        # Recomendações
        print("\n2️⃣ RECOMENDAÇÕES ESTRATÉGICAS:")
        print("   💡 Foco em retenção: Clientes premium têm menor churn")
        print("   💡 Programa de fidelidade para clientes regulares")
        print("   💡 Campanhas de reativação para inativos há 60+ dias")
        print("   💡 Monitorar clientes com baixo rating médio")
        
        # Potencial de receita
        at_risk_customers = df[df['is_churn'] == 1]
        potential_revenue = at_risk_customers['total_spent'].sum()
        
        print(f"\n3️⃣ IMPACTO FINANCEIRO:")
        print(f"   💰 Receita em risco: R$ {potential_revenue:,.2f}")
        print(f"   📈 Oportunidade de recuperação: {len(at_risk_customers)} clientes")
        
    def run_complete_pipeline(self):
        """Executa pipeline completo"""
        print("🚀 Iniciando Pipeline ETL E-commerce Analytics\n")
        
        # ETL Process
        self.generate_sample_data()
        self.extract_data()
        self.transform_data()
        self.load_data()
        
        # Analytics
        self.analyze_data()
        self.train_predictive_model()
        self.generate_insights()
        
        print(f"\n✅ Pipeline concluído! Banco de dados salvo como '{self.db_name}'")
        print("📁 Tabelas criadas: customers, products, orders, customer_analytics, feature_importance, model_predictions")
        
    def close_connection(self):
        """Fecha conexão com banco"""
        self.conn.close()

# Exemplo de uso
if __name__ == "__main__":
    # Inicializar e executar pipeline
    etl = EcommerceETL('ecommerce_portfolio.db')
    
    try:
        etl.run_complete_pipeline()
    finally:
        etl.close_connection()
    
    print("\n" + "="*50)
    print("🎉 PROJETO FINALIZADO!")
    print("="*50)
    print("Este projeto demonstra:")
    print("✓ ETL completo com Python e SQL")
    print("✓ Análise exploratória de dados")
    print("✓ Machine Learning (predição de churn)")
    print("✓ Insights de negócio acionáveis")
    print("✓ Visualizações profissionais")
    print("✓ Código limpo e documentado")
    print("\nPronto para adicionar ao seu portfólio! 🚀")