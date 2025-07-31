"""
🔧 NETFLIX DATA ANALYSIS - PARTIE 4: PIPELINE DE DONNÉES
========================================================

Ce module implémente un pipeline ETL complet pour les données Netflix :
- ETL automatisé avec validation des données
- Contrôles de qualité et cohérence
- Structuration en base de données relationnelle
- Monitoring et logging des processus

Auteur: Bello Soboure
Date: 2025-01-31
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('netflix_pipeline.log'),
        logging.StreamHandler()
    ]
)

class NetflixDataPipeline:
    """Pipeline ETL complet pour les données Netflix"""
    
    def __init__(self, source_file='netflix_data.csv', db_name='netflix_database.db'):
        """Initialise le pipeline de données"""
        self.source_file = source_file
        self.db_name = db_name
        self.logger = logging.getLogger(__name__)
        self.quality_report = {}
        
        # Créer le répertoire de sortie
        self.output_dir = Path('pipeline_output')
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("🔧 Pipeline Netflix initialisé")
        
    def extract_data(self):
        """Phase EXTRACT: Extraction des données source"""
        self.logger.info("📥 Phase EXTRACT: Chargement des données...")
        
        try:
            # Charger les données principales
            self.raw_data = pd.read_csv(self.source_file)
            
            # Métadonnées d'extraction
            self.extraction_metadata = {
                'source_file': self.source_file,
                'extraction_time': datetime.now().isoformat(),
                'total_records': len(self.raw_data),
                'columns': list(self.raw_data.columns),
                'file_size_mb': os.path.getsize(self.source_file) / (1024*1024)
            }
            
            self.logger.info(f"✅ Extraction réussie: {len(self.raw_data)} enregistrements")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur d'extraction: {str(e)}")
            return False
    
    def validate_data_quality(self):
        """Validation de la qualité des données"""
        self.logger.info("🔍 Validation de la qualité des données...")
        
        quality_checks = {}
        
        # 1. Vérification des valeurs manquantes
        missing_data = self.raw_data.isnull().sum()
        quality_checks['missing_values'] = {
            'total_missing': int(missing_data.sum()),
            'missing_by_column': missing_data.to_dict(),
            'missing_percentage': (missing_data / len(self.raw_data) * 100).round(2).to_dict()
        }
        
        # 2. Vérification des doublons
        duplicates = self.raw_data.duplicated().sum()
        quality_checks['duplicates'] = {
            'total_duplicates': int(duplicates),
            'duplicate_percentage': round(duplicates / len(self.raw_data) * 100, 2)
        }
        
        # 3. Validation des types de données
        data_types = {}
        for col in self.raw_data.columns:
            data_types[col] = {
                'dtype': str(self.raw_data[col].dtype),
                'unique_values': int(self.raw_data[col].nunique()),
                'sample_values': self.raw_data[col].dropna().head(3).tolist()
            }
        quality_checks['data_types'] = data_types
        
        # 4. Validation des contraintes métier
        business_rules = {}
        
        # Années de sortie valides
        if 'release_year' in self.raw_data.columns:
            invalid_years = self.raw_data[
                (self.raw_data['release_year'] < 1900) | 
                (self.raw_data['release_year'] > datetime.now().year)
            ]
            business_rules['invalid_release_years'] = len(invalid_years)
        
        # Durées valides (pour les films)
        if 'duration' in self.raw_data.columns and 'type' in self.raw_data.columns:
            movies = self.raw_data[self.raw_data['type'] == 'Movie']
            invalid_durations = movies[
                (pd.to_numeric(movies['duration'], errors='coerce') < 1) |
                (pd.to_numeric(movies['duration'], errors='coerce') > 1000)
            ]
            business_rules['invalid_movie_durations'] = len(invalid_durations)
        
        quality_checks['business_rules'] = business_rules
        
        # 5. Score de qualité global
        total_records = len(self.raw_data)
        quality_score = 100
        
        # Pénalités pour les problèmes de qualité
        if quality_checks['missing_values']['total_missing'] > 0:
            quality_score -= (quality_checks['missing_values']['total_missing'] / total_records) * 20
        
        if quality_checks['duplicates']['total_duplicates'] > 0:
            quality_score -= (quality_checks['duplicates']['total_duplicates'] / total_records) * 30
        
        if sum(business_rules.values()) > 0:
            quality_score -= (sum(business_rules.values()) / total_records) * 25
        
        quality_checks['overall_quality_score'] = max(0, round(quality_score, 2))
        
        self.quality_report = quality_checks
        
        # Sauvegarder le rapport de qualité
        with open(self.output_dir / 'data_quality_report.json', 'w') as f:
            json.dump(quality_checks, f, indent=2, default=str)
        
        self.logger.info(f"✅ Validation terminée - Score qualité: {quality_checks['overall_quality_score']}/100")
        return quality_checks
    
    def transform_data(self):
        """Phase TRANSFORM: Transformation et nettoyage des données"""
        self.logger.info("🔄 Phase TRANSFORM: Transformation des données...")
        
        # Copie des données pour transformation
        self.transformed_data = self.raw_data.copy()
        
        # 1. Nettoyage des valeurs manquantes
        self.transformed_data['genre'] = self.transformed_data['genre'].fillna('Unknown')
        self.transformed_data['director'] = self.transformed_data['director'].fillna('Unknown')
        self.transformed_data['cast'] = self.transformed_data['cast'].fillna('Unknown')
        self.transformed_data['country'] = self.transformed_data['country'].fillna('Unknown')
        self.transformed_data['description'] = self.transformed_data['description'].fillna('')
        
        # 2. Standardisation des types de données
        self.transformed_data['release_year'] = pd.to_numeric(
            self.transformed_data['release_year'], errors='coerce'
        ).fillna(0).astype(int)
        
        self.transformed_data['duration'] = pd.to_numeric(
            self.transformed_data['duration'], errors='coerce'
        ).fillna(0).astype(int)
        
        # 3. Parsing des dates
        self.transformed_data['date_added'] = pd.to_datetime(
            self.transformed_data['date_added'], errors='coerce'
        )
        
        # 4. Création de colonnes dérivées
        self.transformed_data['decade'] = (self.transformed_data['release_year'] // 10) * 10
        self.transformed_data['decade_label'] = self.transformed_data['decade'].astype(str) + 's'
        
        # Extraction de l'année d'ajout
        self.transformed_data['year_added'] = self.transformed_data['date_added'].dt.year
        self.transformed_data['month_added'] = self.transformed_data['date_added'].dt.month
        
        # Catégorisation des durées (pour les films)
        movies_mask = self.transformed_data['type'] == 'Movie'
        self.transformed_data.loc[movies_mask, 'duration_category'] = pd.cut(
            self.transformed_data.loc[movies_mask, 'duration'],
            bins=[0, 90, 120, 150, float('inf')],
            labels=['Court', 'Moyen', 'Long', 'Très long']
        )
        
        # 5. Normalisation des chaînes de caractères
        text_columns = ['title', 'director', 'cast', 'country', 'description', 'genre']
        for col in text_columns:
            if col in self.transformed_data.columns:
                self.transformed_data[col] = self.transformed_data[col].astype(str).str.strip()
        
        # 6. Suppression des doublons
        initial_count = len(self.transformed_data)
        self.transformed_data = self.transformed_data.drop_duplicates()
        duplicates_removed = initial_count - len(self.transformed_data)
        
        if duplicates_removed > 0:
            self.logger.info(f"🧹 {duplicates_removed} doublons supprimés")
        
        # 7. Validation post-transformation
        self.transformation_metadata = {
            'transformation_time': datetime.now().isoformat(),
            'records_before': initial_count,
            'records_after': len(self.transformed_data),
            'duplicates_removed': duplicates_removed,
            'new_columns_created': ['decade', 'decade_label', 'year_added', 'month_added', 'duration_category']
        }
        
        self.logger.info(f"✅ Transformation terminée: {len(self.transformed_data)} enregistrements")
        return True
    
    def create_database_schema(self):
        """Création du schéma de base de données"""
        self.logger.info("🗄️ Création du schéma de base de données...")
        
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Table principale des contenus
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS netflix_content (
                    show_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    director TEXT,
                    cast TEXT,
                    country TEXT,
                    date_added DATE,
                    release_year INTEGER,
                    duration INTEGER,
                    description TEXT,
                    genre TEXT,
                    decade INTEGER,
                    decade_label TEXT,
                    year_added INTEGER,
                    month_added INTEGER,
                    duration_category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table des genres (normalisée)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS genres (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genre_name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table de liaison content-genre (many-to-many)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_genres (
                    content_id TEXT,
                    genre_id INTEGER,
                    PRIMARY KEY (content_id, genre_id),
                    FOREIGN KEY (content_id) REFERENCES netflix_content(show_id),
                    FOREIGN KEY (genre_id) REFERENCES genres(id)
                )
            ''')
            
            # Table des pays (normalisée)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS countries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    country_name TEXT UNIQUE NOT NULL,
                    continent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table de liaison content-country
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_countries (
                    content_id TEXT,
                    country_id INTEGER,
                    PRIMARY KEY (content_id, country_id),
                    FOREIGN KEY (content_id) REFERENCES netflix_content(show_id),
                    FOREIGN KEY (country_id) REFERENCES countries(id)
                )
            ''')
            
            # Table des métriques de qualité
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_run_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON netflix_content(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_year ON netflix_content(release_year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_decade ON netflix_content(decade)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_date_added ON netflix_content(date_added)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Schéma de base de données créé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création schéma: {str(e)}")
            return False
    
    def load_data_to_database(self):
        """Phase LOAD: Chargement des données en base"""
        self.logger.info("📤 Phase LOAD: Chargement en base de données...")
        
        try:
            conn = sqlite3.connect(self.db_name)
            
            # 1. Chargement de la table principale
            self.transformed_data.to_sql(
                'netflix_content', 
                conn, 
                if_exists='replace', 
                index=False,
                method='multi'
            )
            
            # 2. Extraction et chargement des genres uniques
            all_genres = set()
            for genre_list in self.transformed_data['genre'].dropna():
                genres = [g.strip() for g in str(genre_list).split(',')]
                all_genres.update(genres)
            
            genres_df = pd.DataFrame({'genre_name': list(all_genres)})
            genres_df.to_sql('genres', conn, if_exists='replace', index=False)
            
            # 3. Extraction et chargement des pays uniques
            all_countries = set()
            for country_list in self.transformed_data['country'].dropna():
                countries = [c.strip() for c in str(country_list).split(',')]
                all_countries.update(countries)
            
            # Mapping basique des continents
            continent_mapping = {
                'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
                'United Kingdom': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe',
                'Italy': 'Europe', 'Netherlands': 'Europe', 'Belgium': 'Europe', 'Sweden': 'Europe',
                'India': 'Asia', 'China': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia',
                'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
                'Australia': 'Oceania', 'New Zealand': 'Oceania'
            }
            
            countries_data = []
            for country in all_countries:
                continent = continent_mapping.get(country, 'Other')
                countries_data.append({'country_name': country, 'continent': continent})
            
            countries_df = pd.DataFrame(countries_data)
            countries_df.to_sql('countries', conn, if_exists='replace', index=False)
            
            # 4. Chargement des métriques de qualité
            pipeline_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            quality_metrics = []
            
            for metric_name, metric_value in self.quality_report.items():
                if isinstance(metric_value, (int, float)):
                    quality_metrics.append({
                        'pipeline_run_id': pipeline_run_id,
                        'metric_name': metric_name,
                        'metric_value': metric_value,
                        'metric_details': json.dumps(metric_value)
                    })
                elif metric_name == 'overall_quality_score':
                    quality_metrics.append({
                        'pipeline_run_id': pipeline_run_id,
                        'metric_name': metric_name,
                        'metric_value': metric_value,
                        'metric_details': json.dumps(self.quality_report)
                    })
            
            if quality_metrics:
                quality_df = pd.DataFrame(quality_metrics)
                quality_df.to_sql('data_quality_metrics', conn, if_exists='append', index=False)
            
            conn.close()
            
            self.logger.info(f"✅ Données chargées avec succès dans {self.db_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur de chargement: {str(e)}")
            return False
    
    def generate_pipeline_report(self):
        """Génère un rapport complet du pipeline"""
        self.logger.info("📋 Génération du rapport de pipeline...")
        
        report = {
            'pipeline_execution': {
                'start_time': getattr(self, 'pipeline_start_time', datetime.now().isoformat()),
                'end_time': datetime.now().isoformat(),
                'status': 'completed',
                'total_duration_minutes': 0
            },
            'extraction_metadata': getattr(self, 'extraction_metadata', {}),
            'transformation_metadata': getattr(self, 'transformation_metadata', {}),
            'quality_report': self.quality_report,
            'database_info': {
                'database_name': self.db_name,
                'tables_created': ['netflix_content', 'genres', 'countries', 'content_genres', 'content_countries', 'data_quality_metrics'],
                'total_records_loaded': len(getattr(self, 'transformed_data', []))
            }
        }
        
        # Calcul de la durée
        if hasattr(self, 'pipeline_start_time'):
            start = datetime.fromisoformat(self.pipeline_start_time)
            end = datetime.now()
            duration = (end - start).total_seconds() / 60
            report['pipeline_execution']['total_duration_minutes'] = round(duration, 2)
        
        # Sauvegarde du rapport
        report_file = self.output_dir / f'pipeline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Affichage du résumé
        print("\n" + "="*60)
        print("🎯 RAPPORT DE PIPELINE ETL NETFLIX")
        print("="*60)
        print(f"📊 Enregistrements traités: {report['database_info']['total_records_loaded']}")
        print(f"⏱️ Durée d'exécution: {report['pipeline_execution']['total_duration_minutes']} minutes")
        print(f"🎯 Score de qualité: {self.quality_report.get('overall_quality_score', 'N/A')}/100")
        print(f"🗄️ Base de données: {self.db_name}")
        print(f"📁 Rapport sauvegardé: {report_file}")
        
        return report
    
    def run_full_pipeline(self):
        """Exécute le pipeline ETL complet"""
        self.pipeline_start_time = datetime.now().isoformat()
        self.logger.info("🚀 Démarrage du pipeline ETL Netflix")
        
        try:
            # Phase Extract
            if not self.extract_data():
                raise Exception("Échec de l'extraction")
            
            # Validation qualité
            self.validate_data_quality()
            
            # Phase Transform
            if not self.transform_data():
                raise Exception("Échec de la transformation")
            
            # Création du schéma
            if not self.create_database_schema():
                raise Exception("Échec de la création du schéma")
            
            # Phase Load
            if not self.load_data_to_database():
                raise Exception("Échec du chargement")
            
            # Rapport final
            report = self.generate_pipeline_report()
            
            self.logger.info("✅ Pipeline ETL terminé avec succès!")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Échec du pipeline: {str(e)}")
            return None

class DatabaseManager:
    """Gestionnaire pour interroger la base de données Netflix"""
    
    def __init__(self, db_name='netflix_database.db'):
        self.db_name = db_name
    
    def get_connection(self):
        """Obtient une connexion à la base de données"""
        return sqlite3.connect(self.db_name)
    
    def execute_query(self, query, params=None):
        """Exécute une requête SQL"""
        conn = self.get_connection()
        try:
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            return result
        finally:
            conn.close()
    
    def get_content_by_type(self, content_type):
        """Récupère le contenu par type (Movie/TV Show)"""
        query = "SELECT * FROM netflix_content WHERE type = ?"
        return self.execute_query(query, (content_type,))
    
    def get_content_by_decade(self, decade):
        """Récupère le contenu par décennie"""
        query = "SELECT * FROM netflix_content WHERE decade = ?"
        return self.execute_query(query, (decade,))
    
    def get_quality_metrics(self):
        """Récupère les métriques de qualité"""
        query = "SELECT * FROM data_quality_metrics ORDER BY created_at DESC"
        return self.execute_query(query)
    
    def get_database_stats(self):
        """Statistiques générales de la base"""
        stats = {}
        
        # Nombre total d'enregistrements
        stats['total_content'] = self.execute_query(
            "SELECT COUNT(*) as count FROM netflix_content"
        )['count'].iloc[0]
        
        # Répartition par type
        stats['content_by_type'] = self.execute_query(
            "SELECT type, COUNT(*) as count FROM netflix_content GROUP BY type"
        )
        
        # Top 10 genres
        stats['top_genres'] = self.execute_query(
            "SELECT genre_name, COUNT(*) as count FROM genres GROUP BY genre_name ORDER BY count DESC LIMIT 10"
        )
        
        # Contenu par décennie
        stats['content_by_decade'] = self.execute_query(
            "SELECT decade_label, COUNT(*) as count FROM netflix_content GROUP BY decade_label ORDER BY decade"
        )
        
        return stats

def main():
    """Fonction principale pour exécuter le pipeline"""
    print("🔧 NETFLIX DATA PIPELINE - SYSTÈME ETL COMPLET")
    print("="*60)
    
    # Initialiser et exécuter le pipeline
    pipeline = NetflixDataPipeline()
    report = pipeline.run_full_pipeline()
    
    if report:
        print("\n🎉 Pipeline exécuté avec succès!")
        
        # Test de la base de données
        print("\n🔍 Test de la base de données...")
        db_manager = DatabaseManager()
        stats = db_manager.get_database_stats()
        
        print(f"📊 Statistiques de la base:")
        print(f"   • Total contenus: {stats['total_content']}")
        print(f"   • Films: {stats['content_by_type'][stats['content_by_type']['type'] == 'Movie']['count'].iloc[0] if len(stats['content_by_type'][stats['content_by_type']['type'] == 'Movie']) > 0 else 0}")
        print(f"   • Séries: {stats['content_by_type'][stats['content_by_type']['type'] == 'TV Show']['count'].iloc[0] if len(stats['content_by_type'][stats['content_by_type']['type'] == 'TV Show']) > 0 else 0}")
        
        print(f"\n📁 Fichiers générés:")
        print(f"   • Base de données: {pipeline.db_name}")
        print(f"   • Logs: netflix_pipeline.log")
        print(f"   • Rapport qualité: pipeline_output/data_quality_report.json")
        
    else:
        print("❌ Échec du pipeline!")
    
    return report

if __name__ == "__main__":
    results = main()
