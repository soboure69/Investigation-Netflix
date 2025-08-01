#!/usr/bin/env python3
"""
Netflix Data Analysis - Business Intelligence & Strategic Analysis
================================================================

Module complet de Business Intelligence pour l'analyse stratégique des données Netflix.
Analyses ROI, optimisation business, stratégies de contenu et recommandations exécutives.

Auteur: Bello Soboure
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
import sqlite3
import json
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class NetflixBusinessIntelligence:
    """Analyseur Business Intelligence pour Netflix."""
    
    def __init__(self, csv_path: str = "netflix_data.csv", db_path: str = "netflix_database.db"):
        """Initialise l'analyseur BI."""
        self.csv_path = csv_path
        self.db_path = db_path
        self.df = None
        self.bi_results = {}
        self.kpis = {}
        self.recommendations = []
        
        plt.style.use('default')
        sns.set_palette("Set2")
        
        self.load_data()
        if self.df is not None:
            self.prepare_bi_data()
    
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
                
            print(f"✅ Données BI chargées : {len(self.df)} contenus")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données BI : {e}")
    
    def prepare_bi_data(self):
        """Prépare les données pour l'analyse BI."""
        if self.df is None:
            return
        
        print("🔧 Préparation des données BI...")
        
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        self.df['release_year'] = pd.to_numeric(self.df['release_year'], errors='coerce')
        
        # Features business
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        self.df['quarter_added'] = self.df['date_added'].dt.quarter
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df['content_age'] = self.df['year_added'] - self.df['release_year']
        
        self.df['duration_numeric'] = self.df['duration'].str.extract('(\d+)').astype(float)
        
        # Coûts et performance estimés
        self.df['estimated_cost'] = self.estimate_content_cost()
        self.df['performance_score'] = self.calculate_performance_score()
        self.df['estimated_roi'] = self.df['performance_score'] / (self.df['estimated_cost'] / 1000000)
        
        self.df_clean = self.df.dropna(subset=['date_added', 'release_year'])
        
        print(f"✅ Données BI préparées : {len(self.df_clean)} contenus analysables")
    
    def estimate_content_cost(self) -> pd.Series:
        """Estime le coût de production/acquisition du contenu."""
        cost = pd.Series(0.0, index=self.df.index)
        
        # Coût de base par type
        cost += np.where(self.df['type'] == 'Movie', 15_000_000, 8_000_000)
        
        # Facteurs d'ajustement
        duration_factor = np.log1p(self.df['duration_numeric'].fillna(90)) / np.log1p(180)
        cost *= (0.7 + 0.6 * duration_factor)
        
        cast_count = self.df['cast'].fillna('').str.split(',').str.len()
        cast_factor = np.log1p(cast_count) / np.log1p(20)
        cost *= (0.8 + 0.4 * cast_factor)
        
        current_year = datetime.now().year
        year_factor = np.clip((self.df['year_added'] - 2008) / (current_year - 2008), 0.5, 1.5)
        cost *= year_factor
        
        return cost
    
    def calculate_performance_score(self) -> pd.Series:
        """Calcule un score de performance business."""
        score = pd.Series(1.0, index=self.df.index)
        
        genre_count = self.df['listed_in'].fillna('').str.split(',').str.len()
        genre_factor = np.log1p(genre_count) / np.log1p(5)
        score *= (0.8 + 0.4 * genre_factor)
        
        desc_length = self.df['description'].fillna('').str.len()
        desc_factor = np.log1p(desc_length) / np.log1p(500)
        score *= (0.7 + 0.6 * desc_factor)
        
        seasonal_boost = np.where(self.df['month_added'].isin([11, 12, 1]), 1.2, 1.0)
        score *= seasonal_boost
        
        content_age = self.df['year_added'] - self.df['release_year']
        age_factor = np.where(content_age <= 2, 1.3, np.where(content_age >= 20, 1.1, 1.0))
        score *= age_factor
        
        score = (score - score.min()) / (score.max() - score.min()) * 10
        return score
    
    def analyze_roi_by_genre(self):
        """Analyse le ROI par genre."""
        print("💰 Analyse ROI par genre...")
        
        genre_data = []
        for _, row in self.df_clean.iterrows():
            genres = str(row['listed_in']).split(',')
            for genre in genres:
                genre = genre.strip()
                if genre and genre != 'nan':
                    genre_data.append({
                        'genre': genre,
                        'cost': row['estimated_cost'],
                        'performance': row['performance_score'],
                        'roi': row['estimated_roi'],
                        'type': row['type']
                    })
        
        genre_df = pd.DataFrame(genre_data)
        
        genre_analysis = genre_df.groupby('genre').agg({
            'cost': ['mean', 'sum', 'count'],
            'performance': 'mean',
            'roi': 'mean'
        }).round(2)
        
        genre_analysis.columns = ['avg_cost', 'total_cost', 'content_count', 'avg_performance', 'avg_roi']
        genre_analysis = genre_analysis.sort_values('avg_roi', ascending=False)
        
        # Visualisation
        top_genres = genre_analysis.head(15)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].barh(top_genres.index, top_genres['avg_roi'], color='skyblue')
        axes[0, 0].set_title('ROI Moyen par Genre (Top 15)')
        axes[0, 0].set_xlabel('ROI Moyen')
        
        axes[0, 1].scatter(top_genres['avg_cost'], top_genres['avg_performance'], 
                          s=top_genres['content_count']*10, alpha=0.6, color='coral')
        axes[0, 1].set_title('Coût vs Performance par Genre')
        axes[0, 1].set_xlabel('Coût Moyen')
        axes[0, 1].set_ylabel('Performance Moyenne')
        
        axes[1, 0].hist(genre_df['cost']/1000000, bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribution des Coûts de Contenu')
        axes[1, 0].set_xlabel('Coût (M$)')
        
        roi_by_type = genre_df.groupby('type')['roi'].mean()
        axes[1, 1].pie(roi_by_type.values, labels=roi_by_type.index, autopct='%1.1f%%')
        axes[1, 1].set_title('ROI Moyen par Type de Contenu')
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.bi_results['roi_by_genre'] = {
            'top_roi_genres': top_genres.head(10).to_dict('index'),
            'total_investment': genre_df['cost'].sum(),
            'average_roi': genre_df['roi'].mean()
        }
        
        print(f"✅ Analyse ROI terminée - ROI moyen : {genre_df['roi'].mean():.2f}")
    
    def calculate_business_kpis(self):
        """Calcule les KPIs business clés."""
        print("📈 Calcul des KPIs business...")
        
        total_investment = self.df_clean['estimated_cost'].sum()
        total_performance = self.df_clean['performance_score'].sum()
        average_roi = self.df_clean['estimated_roi'].mean()
        
        content_diversity = len(set([g.strip() for genres in self.df_clean['listed_in'].fillna('').str.split(',') for g in genres if g.strip()]))
        geographic_reach = len(set([c.strip() for countries in self.df_clean['country'].fillna('').str.split(',') for c in countries if c.strip()]))
        
        content_freshness = (self.df_clean['content_age'] <= 5).sum() / len(self.df_clean) * 100
        high_performance_content = (self.df_clean['performance_score'] > self.df_clean['performance_score'].quantile(0.75)).sum()
        
        self.kpis = {
            'financial': {
                'total_investment_millions': round(total_investment / 1000000, 2),
                'average_roi': round(average_roi, 2),
                'high_roi_content_percentage': round((self.df_clean['estimated_roi'] > 2).sum() / len(self.df_clean) * 100, 1)
            },
            'content': {
                'total_content_count': len(self.df_clean),
                'content_diversity_score': content_diversity,
                'geographic_reach': geographic_reach,
                'content_freshness_percentage': round(content_freshness, 1),
                'high_performance_content_count': high_performance_content
            }
        }
        
        print("✅ KPIs business calculés")
    
    def generate_executive_recommendations(self):
        """Génère des recommandations exécutives."""
        print("🎯 Génération des recommandations exécutives...")
        
        recommendations = []
        
        # Recommandations basées sur ROI
        if 'roi_by_genre' in self.bi_results:
            top_roi_genres = list(self.bi_results['roi_by_genre']['top_roi_genres'].keys())[:3]
            recommendations.append({
                'category': 'Investment Strategy',
                'title': 'Optimiser les investissements par genre',
                'description': f"Concentrer 60% des nouveaux investissements sur les genres à haut ROI : {', '.join(top_roi_genres)}",
                'impact': 'High',
                'expected_benefit': 'Augmentation du ROI global de 15-25%'
            })
        
        # Recommandations basées sur les KPIs
        if self.kpis:
            if self.kpis['financial']['average_roi'] < 2:
                recommendations.append({
                    'category': 'Financial Optimization',
                    'title': 'Améliorer l\'efficacité des coûts',
                    'description': 'ROI moyen inférieur à 2. Revoir les processus d\'acquisition et de production.',
                    'impact': 'High',
                    'expected_benefit': 'Réduction des coûts de 10-15%'
                })
            
            if self.kpis['content']['content_freshness_percentage'] < 30:
                recommendations.append({
                    'category': 'Content Strategy',
                    'title': 'Rajeunir le catalogue',
                    'description': 'Moins de 30% de contenu récent. Augmenter les acquisitions de contenu frais.',
                    'impact': 'Medium',
                    'expected_benefit': 'Amélioration de l\'engagement utilisateur'
                })
        
        self.recommendations = recommendations
        print("✅ Recommandations exécutives générées")
    
    def create_executive_dashboard(self):
        """Crée un dashboard exécutif."""
        print("📊 Création du dashboard exécutif...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # KPI financiers
        if self.kpis:
            kpi_names = ['Investment\n(M$)', 'Avg ROI', 'High ROI\nContent (%)']
            kpi_values = [
                self.kpis['financial']['total_investment_millions'],
                self.kpis['financial']['average_roi'],
                self.kpis['financial']['high_roi_content_percentage']
            ]
            
            axes[0, 0].bar(kpi_names, kpi_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 0].set_title('KPIs Financiers Clés')
            axes[0, 0].set_ylabel('Valeur')
        
        # Evolution des investissements
        yearly_investment = self.df_clean.groupby('year_added')['estimated_cost'].sum() / 1000000
        axes[0, 1].plot(yearly_investment.index, yearly_investment.values, marker='o', linewidth=3, color='green')
        axes[0, 1].set_title('Évolution des Investissements Annuels')
        axes[0, 1].set_xlabel('Année')
        axes[0, 1].set_ylabel('Investment (M$)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance par type
        performance_by_type = self.df_clean.groupby('type')['performance_score'].mean()
        axes[0, 2].pie(performance_by_type.values, labels=performance_by_type.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Performance Moyenne par Type')
        
        # ROI distribution
        axes[1, 0].hist(self.df_clean['estimated_roi'], bins=30, alpha=0.7, color='orange')
        axes[1, 0].axvline(self.df_clean['estimated_roi'].mean(), color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Distribution du ROI')
        axes[1, 0].set_xlabel('ROI')
        axes[1, 0].set_ylabel('Fréquence')
        
        # Content freshness
        age_bins = [0, 2, 5, 10, 20, 100]
        age_labels = ['0-2 ans', '2-5 ans', '5-10 ans', '10-20 ans', '20+ ans']
        age_distribution = pd.cut(self.df_clean['content_age'], bins=age_bins, labels=age_labels).value_counts()
        
        axes[1, 1].bar(age_distribution.index, age_distribution.values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Distribution de l\'Âge du Contenu')
        axes[1, 1].set_xlabel('Âge du Contenu')
        axes[1, 1].set_ylabel('Nombre de Contenus')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Geographic diversity
        country_counts = {}
        for countries in self.df_clean['country'].fillna(''):
            for country in str(countries).split(','):
                country = country.strip()
                if country and country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + 1
        
        top_countries = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        axes[1, 2].barh(list(top_countries.keys()), list(top_countries.values()), color='teal')
        axes[1, 2].set_title('Top 10 Pays par Nombre de Contenus')
        axes[1, 2].set_xlabel('Nombre de Contenus')
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Dashboard exécutif créé")
    
    def generate_bi_report(self):
        """Génère un rapport BI complet."""
        print("📋 Génération du rapport BI...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': {
                'total_content': len(self.df_clean),
                'total_investment_millions': self.kpis['financial']['total_investment_millions'] if self.kpis else 0,
                'average_roi': self.kpis['financial']['average_roi'] if self.kpis else 0,
                'content_diversity': self.kpis['content']['content_diversity_score'] if self.kpis else 0
            },
            'key_findings': self.bi_results,
            'kpis': self.kpis,
            'recommendations': self.recommendations
        }
        
        # Sauvegarde du rapport
        os.makedirs('bi_output', exist_ok=True)
        with open('bi_output/business_intelligence_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print("✅ Rapport BI généré dans bi_output/")
        
        # Résumé exécutif
        print("\n🎯 Résumé Exécutif:")
        print(f"  📊 Total contenus analysés : {len(self.df_clean):,}")
        if self.kpis:
            print(f"  💰 Investissement total estimé : ${self.kpis['financial']['total_investment_millions']:,.0f}M")
            print(f"  📈 ROI moyen : {self.kpis['financial']['average_roi']:.2f}")
            print(f"  🎭 Diversité des genres : {self.kpis['content']['content_diversity_score']} genres")
        
        print(f"\n💡 Recommandations stratégiques : {len(self.recommendations)}")
        for i, rec in enumerate(self.recommendations[:3], 1):
            print(f"  {i}. {rec['title']} (Impact: {rec['impact']})")
    
    def run_complete_bi_analysis(self):
        """Exécute l'analyse BI complète."""
        print("🚀 Démarrage de l'analyse Business Intelligence complète...")
        
        if self.df is None:
            print("❌ Aucune donnée disponible pour l'analyse BI")
            return
        
        try:
            # Création des dossiers de sortie
            os.makedirs('Graphiques générés', exist_ok=True)
            os.makedirs('bi_output', exist_ok=True)
            
            # Analyses BI
            self.analyze_roi_by_genre()
            self.calculate_business_kpis()
            self.generate_executive_recommendations()
            self.create_executive_dashboard()
            self.generate_bi_report()
            
            print("\n🎉 Analyse Business Intelligence complète terminée avec succès !")
            print("📁 Résultats disponibles dans bi_output/ et Graphiques générés/")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse BI : {e}")

def main():
    """Fonction principale."""
    analyzer = NetflixBusinessIntelligence()
    analyzer.run_complete_bi_analysis()

if __name__ == "__main__":
    main()
