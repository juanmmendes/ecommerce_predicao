# pipeline.py - Implementação da classe EcommerceETL para o pipeline de ETL e análise de dados de e-commerce.

import pandas as pd
import numpy as np
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
    """Pipeline completo de ETL para análise de e-commerce."""
    
    def __init__(self, db_name='ecommerce_analytics.db', conn=None):
        self.db_name = db_name
        self.conn = conn  # Para testes: conexão compartilhada
        self.data = {}
        self.model = None
        self.scaler = StandardScaler()
    
    def generate_sample_data(self):
        print("🔄 Gerando dados de exemplo...")
        n_customers = 5000
        n_products = 200
        n_orders = 15000
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
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
        products = pd.DataFrame({
            'product_id': range(1, n_products + 1),
            'category': np.random.choice(categories, n_products),
            'price': np.random.lognormal(3, 1, n_products).round(2),
            'rating': np.random.normal(4, 0.5, n_products).clip(1, 5).round(1)
        })
        orders_data = []
        for _ in range(n_orders):
            customer_id = np.random.randint(1, n_customers + 1)
            customer_age = customers[customers['customer_id'] == customer_id]['age'].iloc[0]
            is_premium = customers[customers['customer_id'] == customer_id]['premium_member'].iloc[0]
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
        if self.conn is not None:
            conn = self.conn
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
        customers.to_sql('customers', conn, if_exists='replace', index=False)
        products.to_sql('products', conn, if_exists='replace', index=False)
        orders.to_sql('orders', conn, if_exists='replace', index=False)
        if self.conn is None:
            conn.close()
        print(f"✅ Dados gerados: {len(customers)} clientes, {len(products)} produtos, {len(orders)} pedidos")

    def extract_data(self):
        print("📥 Extraindo dados do banco...")
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
        if self.conn is not None:
            conn = self.conn
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.data['raw'] = pd.read_sql_query(query, conn)
        if self.conn is None:
            conn.close()
        print(f"✅ Extraídos {len(self.data['raw'])} registros")

    def transform_data(self):
        print("🔄 Transformando dados...")
        df = self.data['raw'].copy()
        df['registration_date'] = pd.to_datetime(df['registration_date'], format='mixed')
        df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')
        df = df[(df['age'] >= 18) & (df['age'] <= 80)]
        customer_metrics = df.groupby('customer_id').agg({
            'total_value': ['sum', 'mean', 'count'],
            'order_date': ['min', 'max'],
            'category': lambda x: x.nunique(),
            'discount_applied': 'mean',
            'rating': 'mean'
        }).round(2)
        customer_metrics.columns = [
            'total_spent', 'avg_order_value', 'total_orders',
            'first_order', 'last_order', 'categories_bought',
            'discount_usage', 'avg_rating'
        ]
        reference_date = pd.to_datetime('2024-01-31')
        customer_metrics['days_since_last_order'] = (
            reference_date - customer_metrics['last_order']
        ).dt.days
        customer_info = df[['customer_id', 'age', 'gender', 'city', 
                          'registration_date', 'premium_member']].drop_duplicates()
        final_df = customer_info.merge(customer_metrics, on='customer_id', how='left')
        final_df['is_churn'] = (final_df['days_since_last_order'] > 90).astype(int)
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
        final_df['customer_lifetime_days'] = (
            reference_date - final_df['registration_date']
        ).dt.days
        self.data['transformed'] = final_df
        print(f"✅ Transformação concluída: {len(final_df)} clientes processados")

    def load_data(self):
        print("💾 Carregando dados transformados...")
        if self.conn is not None:
            conn = self.conn
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.data['transformed'].to_sql(
            'customer_analytics', conn, if_exists='replace', index=False
        )
        if self.conn is None:
            conn.close()
        print("✅ Dados carregados na tabela customer_analytics")

    def analyze_data(self):
        print("📊 Iniciando análise exploratória...")
        df = self.data['transformed']
        print("\n=== RESUMO EXECUTIVO ===")
        print(f"Total de clientes: {len(df):,}")
        print(f"Taxa de churn: {df['is_churn'].mean():.1%}")
        print(f"Receita total: R$ {df['total_spent'].sum():,.2f}")
        print(f"Ticket médio: R$ {df['avg_order_value'].mean():.2f}")
        print(f"Clientes premium: {df['premium_member'].mean():.1%}")
        print("\n=== ANÁLISE POR SEGMENTO ===")
        churn_by_premium = df.groupby('premium_member')['is_churn'].mean()
        print(f"Churn - Clientes regulares: {churn_by_premium[0]:.1%}")
        print(f"Churn - Clientes premium: {churn_by_premium[1]:.1%}")
        churn_by_city = df.groupby('city')['is_churn'].mean().sort_values(ascending=False)
        print(f"\nCidades com maior churn:")
        for city, rate in churn_by_city.head(3).items():
            print(f"  {city}: {rate:.1%}")
        self._create_visualizations(df)

    def _create_visualizations(self, df):
        print("📊 Criando visualizações...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dashboard E-commerce Analytics', fontsize=16, fontweight='bold')
        churn_counts = df['is_churn'].value_counts()
        axes[0,0].pie(churn_counts.values, 
                     labels=['Ativo', 'Churn'], 
                     autopct='%1.1f%%',
                     colors=['#2ecc71', '#e74c3c'])
        axes[0,0].set_title('Distribuição de Churn')
        if self.conn is not None:
            conn = self.conn
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
        df_orders = pd.read_sql_query("""
            SELECT 
                strftime('%Y-%m', order_date) as month,
                SUM(quantity * price * (1 - discount_applied * 0.1)) as revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE order_date >= '2023-01-01'
            GROUP BY month
            ORDER BY month
        """, conn)
        if self.conn is None:
            conn.close()
        axes[0,1].plot(range(len(df_orders)), df_orders['revenue'])
        axes[0,1].set_title('Receita Mensal')
        axes[0,1].set_ylabel('Receita (R$)')
        axes[0,1].tick_params(axis='x', rotation=45)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        churn_by_age = df.groupby('age_group')['is_churn'].mean()
        axes[1,0].bar(range(len(churn_by_age)), churn_by_age.values)
        axes[1,0].set_title('Taxa de Churn por Faixa Etária')
        axes[1,0].set_xticks(range(len(churn_by_age)))
        axes[1,0].set_xticklabels(churn_by_age.index)
        axes[1,0].set_ylabel('Taxa de Churn')
        axes[1,1].boxplot([df[df['is_churn']==0]['total_spent'], 
                          df[df['is_churn']==1]['total_spent']])
        axes[1,1].set_title('Distribuição de Gastos por Status')
        axes[1,1].set_xticklabels(['Ativo', 'Churn'])
        axes[1,1].set_ylabel('Total Gasto (R$)')
        plt.tight_layout()
        plt.show()

    def generate_insights(self):
        print("💡 Gerando insights...")
        print("\n" + "="*50)
        print("🎯 INSIGHTS DE NEGÓCIO")
        print("="*50)
        df = self.data['transformed']
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
        print("\n2️⃣ RECOMENDAÇÕES ESTRATÉGICAS:")
        print("   💡 Foco em retenção: Clientes premium têm menor churn")
        print("   💡 Programa de fidelidade para clientes regulares")
        print("   💡 Campanhas de reativação para inativos há 60+ dias")
        print("   💡 Monitorar clientes com baixo rating médio")
        at_risk_customers = df[df['is_churn'] == 1]
        potential_revenue = at_risk_customers['total_spent'].sum()
        print(f"\n3️⃣ IMPACTO FINANCEIRO:")
        print(f"   💰 Receita em risco: R$ {potential_revenue:,.2f}")
        print(f"   📈 Oportunidade de recuperação: {len(at_risk_customers)} clientes")
        
    def run_complete_pipeline(self):
        print("🚀 Iniciando Pipeline ETL E-commerce Analytics\n")
        
        # ETL Process
        self.generate_sample_data()
        self.extract_data()
        self.transform_data()
        self.load_data()
        
        # Analytics
        self.analyze_data()
        
        # Treinamento do modelo preditivo (garante scaler e modelo prontos)
        self.train_predictive_model()
        
        # Insights
        self.generate_insights()
        
        print(f"\n✅ Pipeline concluído! Banco de dados salvo como '{self.db_name}'")
        print("📁 Tabelas criadas: customers, products, orders, customer_analytics, feature_importance, model_predictions")

    def train_predictive_model(self):
        """Treina modelo de predição de churn"""
        print("🤖 Treinando modelo preditivo...")
        df = self.data['transformed'].copy()
        # Encoding categóricas
        le_gender = LabelEncoder()
        le_city = LabelEncoder()
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['city_encoded'] = le_city.fit_transform(df['city'])
        # Salvar as colunas encodadas para acesso nos testes
        self.data['transformed']['gender_encoded'] = df['gender_encoded']
        self.data['transformed']['city_encoded'] = df['city_encoded']
        features = [
            'age', 'gender_encoded', 'city_encoded', 'premium_member',
            'total_spent', 'avg_order_value', 'total_orders',
            'categories_bought', 'discount_usage', 'avg_rating',
            'days_since_last_order', 'customer_lifetime_days'
        ]
        X = df[features]
        y = df['is_churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=20
        )
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        print("\n=== PERFORMANCE DO MODELO ===")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n=== IMPORTÂNCIA DAS FEATURES ===")
        for _, row in feature_importance.head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
        self._save_model_results(feature_importance, y_test, y_pred, y_pred_proba)

    def _save_model_results(self, feature_importance, y_test, y_pred, y_pred_proba):
        """Salva resultados do modelo"""
        if self.conn is not None:
            conn = self.conn
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
        feature_importance.to_sql('feature_importance', conn, if_exists='replace', index=False)
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba
        })
        results_df.to_sql('model_predictions', conn, if_exists='replace', index=False)
        if self.conn is None:
            conn.close()
