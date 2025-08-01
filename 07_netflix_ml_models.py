#!/usr/bin/env python3
"""
Netflix Data Analysis - Machine Learning & AI Models
===================================================

Module complet de Machine Learning et Intelligence Artificielle pour l'analyse
des données Netflix avec modèles prédictifs, classification et recommandations.

Auteur: Bello Soboure
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import sqlite3
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any

# Machine Learning imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, mean_squared_error, r2_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn non disponible. Certaines fonctionnalités ML seront limitées.")

# NLP imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob non disponible. Analyse de sentiment limitée.")

warnings.filterwarnings('ignore')

class NetflixMLAnalyzer:
    """Analyseur Machine Learning pour les données Netflix."""
    
    def __init__(self, csv_path: str = "netflix_data.csv", db_path: str = "netflix_database.db"):
        """Initialise l'analyseur ML."""
        self.csv_path = csv_path
        self.db_path = db_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.results = {}
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.load_data()
        if self.df is not None:
            self.prepare_ml_features()
    
    def load_data(self):
        """Charge les données depuis la base de données ou le CSV."""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT c.*, 
                       GROUP_CONCAT(DISTINCT g.genre_name) as genres,
                       GROUP_CONCAT(DISTINCT co.country_name) as countries
                FROM content c
                LEFT JOIN content_genres cg ON c.content_id = cg.content_id
                LEFT JOIN genres g ON cg.genre_id = g.genre_id
                LEFT JOIN content_countries cc ON c.content_id = cc.content_id
                LEFT JOIN countries co ON cc.country_id = co.country_id
                GROUP BY c.content_id
                """
                self.df = pd.read_sql_query(query, conn)
                conn.close()
                
                column_mapping = {
                    'show_id': 'show_id', 'content_type': 'type', 'title': 'title',
                    'director': 'director', 'cast': 'cast', 'countries': 'country',
                    'date_added': 'date_added', 'release_year': 'release_year',
                    'rating': 'rating', 'duration': 'duration', 'genres': 'listed_in',
                    'description': 'description'
                }
                self.df = self.df.rename(columns=column_mapping)
            else:
                self.df = pd.read_csv(self.csv_path)
                
            print(f"✅ Données chargées : {len(self.df)} contenus")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
    
    def prepare_ml_features(self):
        """Prépare les features pour le Machine Learning."""
        if self.df is None:
            return
        
        print("🔧 Préparation des features ML...")
        
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        self.df['release_year'] = pd.to_numeric(self.df['release_year'], errors='coerce')
        
        # Features temporelles
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df['age_when_added'] = self.df['year_added'] - self.df['release_year']
        
        # Features de durée et texte
        self.df['duration_numeric'] = self.df['duration'].str.extract('(\d+)').astype(float)
        self.df['title_length'] = self.df['title'].str.len()
        self.df['description_length'] = self.df['description'].fillna('').str.len()
        self.df['description_word_count'] = self.df['description'].fillna('').str.split().str.len()
        
        # Features de casting
        self.df['has_director'] = (~self.df['director'].isna()).astype(int)
        self.df['has_cast'] = (~self.df['cast'].isna()).astype(int)
        self.df['cast_count'] = self.df['cast'].fillna('').str.split(',').str.len()
        
        # Features de pays et genres
        self.df['country_count'] = self.df['country'].fillna('').str.split(',').str.len()
        self.df['genre_count'] = self.df['listed_in'].fillna('').str.split(',').str.len()
        
        # Score de popularité
        self.df['popularity_score'] = self.create_popularity_score()
        
        self.df_clean = self.df.dropna(subset=['date_added', 'release_year', 'duration_numeric'])
        print(f"✅ Features préparées : {len(self.df_clean)} contenus utilisables")
    
    def create_popularity_score(self) -> pd.Series:
        """Crée un score de popularité basé sur les features disponibles."""
        score = pd.Series(0.0, index=self.df.index)
        
        current_year = datetime.now().year
        year_factor = (self.df['year_added'] - 2008) / (current_year - 2008)
        score += year_factor * 0.3
        
        country_factor = np.log1p(self.df['country_count']) / np.log1p(self.df['country_count'].max())
        score += country_factor * 0.2
        
        cast_factor = np.log1p(self.df['cast_count']) / np.log1p(self.df['cast_count'].max())
        score += cast_factor * 0.2
        
        desc_factor = np.log1p(self.df['description_length']) / np.log1p(self.df['description_length'].max())
        score += desc_factor * 0.15
        
        genre_factor = np.log1p(self.df['genre_count']) / np.log1p(self.df['genre_count'].max())
        score += genre_factor * 0.15
        
        score = (score - score.min()) / (score.max() - score.min())
        return score
    
    def build_success_prediction_model(self):
        """Construit un modèle de prédiction de succès."""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn requis pour la prédiction de succès")
            return
        
        print("🤖 Construction du modèle de prédiction de succès...")
        
        feature_columns = [
            'release_year', 'duration_numeric', 'age_when_added',
            'title_length', 'description_length', 'description_word_count',
            'has_director', 'has_cast', 'cast_count', 'country_count', 'genre_count'
        ]
        
        X = self.df_clean[feature_columns].fillna(0)
        y = self.df_clean['popularity_score']
        
        # Encodage des variables catégorielles
        type_encoder = LabelEncoder()
        X['type_encoded'] = type_encoder.fit_transform(self.df_clean['type'])
        
        rating_encoder = LabelEncoder()
        X['rating_encoded'] = rating_encoder.fit_transform(self.df_clean['rating'].fillna('Unknown'))
        
        feature_columns.extend(['type_encoded', 'rating_encoded'])
        X = X[feature_columns]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"  Random Forest: R² = {score:.3f}, MSE = {mse:.3f}")
        
        # Sauvegarde du modèle
        self.models['success_prediction'] = model
        self.scalers['success_prediction'] = scaler
        self.encoders['success_prediction'] = {
            'type': type_encoder,
            'rating': rating_encoder,
            'features': feature_columns
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Features - Prédiction de Succès')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_feature_importance_success.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['success_prediction'] = {
            'model': 'Random Forest',
            'r2_score': score,
            'mse': mse,
            'feature_importance': feature_importance
        }
        
        print(f"✅ Modèle de succès construit (R² = {score:.3f})")
    
    def build_genre_classification_model(self):
        """Construit un modèle de classification automatique des genres."""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn requis pour la classification de genres")
            return
        
        print("🎭 Construction du modèle de classification de genres...")
        
        descriptions = self.df_clean['description'].fillna('')
        
        # Préparation des labels (genres principaux)
        genres = []
        for genre_list in self.df_clean['listed_in'].fillna(''):
            if genre_list:
                primary_genre = genre_list.split(',')[0].strip()
                genres.append(primary_genre)
            else:
                genres.append('Unknown')
        
        # Filtrer les genres avec au moins 10 exemples
        genre_counts = pd.Series(genres).value_counts()
        valid_genres = genre_counts[genre_counts >= 10].index.tolist()
        
        mask = pd.Series(genres).isin(valid_genres)
        descriptions_filtered = descriptions[mask]
        genres_filtered = pd.Series(genres)[mask]
        
        print(f"  Genres valides : {len(valid_genres)}")
        print(f"  Échantillons : {len(descriptions_filtered)}")
        
        # Vectorisation TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2), min_df=2)
        X = tfidf.fit_transform(descriptions_filtered)
        y = genres_filtered
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Entraînement Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        print(f"  Random Forest: Accuracy = {score:.3f}")
        
        # Sauvegarde du modèle
        self.models['genre_classification'] = model
        self.vectorizers['genre_classification'] = tfidf
        self.encoders['genre_classification'] = {'valid_genres': valid_genres}
        
        self.results['genre_classification'] = {
            'model': 'Random Forest',
            'accuracy': score,
            'valid_genres': valid_genres
        }
        
        print(f"✅ Modèle de classification construit (Accuracy = {score:.3f})")
    
    def build_recommendation_system(self):
        """Construit un système de recommandation basé sur le contenu."""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn requis pour le système de recommandation")
            return
        
        print("🎯 Construction du système de recommandation...")
        
        # Préparation des features textuelles
        descriptions = self.df_clean['description'].fillna('')
        titles = self.df_clean['title'].fillna('')
        text_features = titles + ' ' + descriptions
        
        # Vectorisation TF-IDF
        tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        text_matrix = tfidf.fit_transform(text_features)
        
        # Features numériques
        numeric_features = ['release_year', 'duration_numeric', 'title_length', 
                          'description_length', 'cast_count', 'country_count', 'genre_count']
        numeric_matrix = self.df_clean[numeric_features].fillna(0)
        
        # Normalisation
        scaler = StandardScaler()
        numeric_matrix_scaled = scaler.fit_transform(numeric_matrix)
        
        # Réduction de dimension pour le texte
        svd = TruncatedSVD(n_components=50, random_state=42)
        text_matrix_reduced = svd.fit_transform(text_matrix)
        
        # Matrice finale
        combined_matrix = np.hstack([text_matrix_reduced, numeric_matrix_scaled])
        
        # Modèle de voisins les plus proches
        nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        nn_model.fit(combined_matrix)
        
        # Sauvegarde
        self.models['recommendation'] = nn_model
        self.vectorizers['recommendation'] = tfidf
        self.scalers['recommendation'] = scaler
        self.encoders['recommendation'] = {
            'svd': svd,
            'numeric_features': numeric_features,
            'content_indices': self.df_clean.index.tolist()
        }
        
        # Test avec quelques exemples
        sample_indices = self.df_clean.sample(3, random_state=42).index
        recommendations = {}
        
        for idx in sample_indices:
            content_title = self.df_clean.loc[idx, 'title']
            recs = self.get_recommendations(content_title, n_recommendations=5)
            recommendations[content_title] = recs
        
        self.results['recommendation'] = {
            'model_type': 'Content-Based Filtering',
            'sample_recommendations': recommendations
        }
        
        print("✅ Système de recommandation construit")
        
        # Affichage d'exemples
        for title, recs in recommendations.items():
            print(f"\n📺 Recommandations pour '{title}':")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")
    
    def get_recommendations(self, title: str, n_recommendations: int = 5) -> List[str]:
        """Obtient des recommandations pour un titre donné."""
        if 'recommendation' not in self.models:
            return []
        
        matches = self.df_clean[self.df_clean['title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            return []
        
        idx = matches.index[0]
        content_idx = self.encoders['recommendation']['content_indices'].index(idx)
        
        distances, indices = self.models['recommendation'].kneighbors(
            [self.get_content_vector(idx)], n_neighbors=n_recommendations+1
        )
        
        recommended_indices = indices[0][1:]
        content_indices = self.encoders['recommendation']['content_indices']
        
        recommendations = []
        for rec_idx in recommended_indices:
            original_idx = content_indices[rec_idx]
            rec_title = self.df_clean.loc[original_idx, 'title']
            recommendations.append(rec_title)
        
        return recommendations
    
    def get_content_vector(self, idx: int) -> np.ndarray:
        """Obtient le vecteur de features pour un contenu."""
        text = self.df_clean.loc[idx, 'title'] + ' ' + self.df_clean.loc[idx, 'description']
        text_vector = self.vectorizers['recommendation'].transform([text])
        text_vector_reduced = self.encoders['recommendation']['svd'].transform(text_vector)
        
        numeric_features = self.encoders['recommendation']['numeric_features']
        numeric_vector = self.df_clean.loc[idx, numeric_features].fillna(0).values.reshape(1, -1)
        numeric_vector_scaled = self.scalers['recommendation'].transform(numeric_vector)
        
        combined_vector = np.hstack([text_vector_reduced, numeric_vector_scaled])
        return combined_vector[0]
    
    def perform_sentiment_analysis(self):
        """Effectue une analyse de sentiment des descriptions."""
        if not TEXTBLOB_AVAILABLE:
            print("⚠️  TextBlob non disponible. Analyse de sentiment basique utilisée.")
            self.basic_sentiment_analysis()
            return
        
        print("😊 Analyse de sentiment des descriptions...")
        
        descriptions = self.df_clean['description'].fillna('')
        sentiments = []
        polarities = []
        
        for desc in descriptions:
            if desc:
                blob = TextBlob(desc)
                sentiment = blob.sentiment
                polarities.append(sentiment.polarity)
                
                if sentiment.polarity > 0.1:
                    sentiments.append('Positive')
                elif sentiment.polarity < -0.1:
                    sentiments.append('Negative')
                else:
                    sentiments.append('Neutral')
            else:
                sentiments.append('Neutral')
                polarities.append(0.0)
        
        self.df_clean = self.df_clean.copy()
        self.df_clean['sentiment'] = sentiments
        self.df_clean['polarity'] = polarities
        
        # Visualisation
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Distribution des Sentiments')
        
        axes[1].hist(polarities, bins=30, alpha=0.7, color='skyblue')
        axes[1].set_title('Distribution de la Polarité')
        axes[1].set_xlabel('Polarité (-1: Négatif, +1: Positif)')
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['sentiment_analysis'] = {
            'sentiment_distribution': sentiment_counts.to_dict(),
            'average_polarity': np.mean(polarities)
        }
        
        print(f"✅ Analyse de sentiment terminée (Polarité moyenne : {np.mean(polarities):.3f})")
    
    def basic_sentiment_analysis(self):
        """Analyse de sentiment basique basée sur des mots-clés."""
        positive_words = ['love', 'amazing', 'great', 'excellent', 'wonderful', 'fantastic', 
                         'adventure', 'comedy', 'funny', 'romantic', 'heartwarming']
        
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'tragic', 'dark', 
                         'violence', 'crime', 'murder', 'death', 'war']
        
        descriptions = self.df_clean['description'].fillna('').str.lower()
        sentiments = []
        
        for desc in descriptions:
            pos_count = sum(1 for word in positive_words if word in desc)
            neg_count = sum(1 for word in negative_words if word in desc)
            
            if pos_count > neg_count:
                sentiments.append('Positive')
            elif neg_count > pos_count:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        
        self.df_clean = self.df_clean.copy()
        self.df_clean['sentiment'] = sentiments
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        self.results['sentiment_analysis'] = {
            'method': 'keyword_based',
            'sentiment_distribution': sentiment_counts.to_dict()
        }
        
        print(f"✅ Analyse de sentiment basique terminée")
    
    def generate_ml_report(self):
        """Génère un rapport complet des analyses ML."""
        print("\n📊 Génération du rapport ML...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_content': len(self.df) if self.df is not None else 0,
                'clean_content': len(self.df_clean) if self.df_clean is not None else 0
            },
            'models_built': list(self.models.keys()),
            'results': self.results
        }
        
        # Sauvegarde du rapport
        os.makedirs('ml_output', exist_ok=True)
        with open('ml_output/ml_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Sauvegarde des modèles
        for model_name, model in self.models.items():
            model_data = {
                'model': model,
                'scaler': self.scalers.get(model_name),
                'encoder': self.encoders.get(model_name),
                'vectorizer': self.vectorizers.get(model_name)
            }
            
            with open(f'ml_output/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        
        print("✅ Rapport ML généré dans ml_output/")
        
        # Résumé des résultats
        print("\n🎯 Résumé des Modèles ML:")
        for model_name, result in self.results.items():
            print(f"\n  📈 {model_name.replace('_', ' ').title()}:")
            if 'r2_score' in result:
                print(f"    - R² Score: {result['r2_score']:.3f}")
            if 'accuracy' in result:
                print(f"    - Accuracy: {result['accuracy']:.3f}")
            if 'average_polarity' in result:
                print(f"    - Polarité moyenne: {result['average_polarity']:.3f}")
    
    def run_complete_analysis(self):
        """Exécute l'analyse ML complète."""
        print("🚀 Démarrage de l'analyse ML complète...")
        
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn requis pour l'analyse ML complète")
            return
        
        if self.df is None:
            print("❌ Aucune donnée disponible")
            return
        
        # Création du dossier de sortie
        os.makedirs('Graphiques générés', exist_ok=True)
        
        try:
            # 1. Modèle de prédiction de succès
            self.build_success_prediction_model()
            
            # 2. Classification de genres
            self.build_genre_classification_model()
            
            # 3. Système de recommandation
            self.build_recommendation_system()
            
            # 4. Analyse de sentiment
            self.perform_sentiment_analysis()
            
            # 5. Génération du rapport
            self.generate_ml_report()
            
            print("\n🎉 Analyse ML complète terminée avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse ML : {e}")

def main():
    """Fonction principale."""
    analyzer = NetflixMLAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
