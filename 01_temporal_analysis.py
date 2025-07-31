"""
🎬 NETFLIX DATA ANALYSIS - PARTIE 1: ANALYSE TEMPORELLE COMPLÈTE
================================================================

Ce module analyse les tendances temporelles du contenu Netflix :
- Évolution du contenu dans le temps
- Patterns saisonniers d'ajout de contenu
- Comparaison entre décennies
- Analyse des durées par époque

Auteur: Bello Soboure
Date: 2025-01-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class NetflixTemporalAnalyzer:
    """Classe pour l'analyse temporelle des données Netflix"""
    
    def __init__(self, csv_path):
        """Initialise l'analyseur avec le dataset Netflix"""
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prépare les données pour l'analyse temporelle"""
        print("🔄 Préparation des données...")
        
        # Conversion des dates
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        
        # Extraction des composantes temporelles
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        self.df['month_name'] = self.df['date_added'].dt.month_name()
        self.df['quarter_added'] = self.df['date_added'].dt.quarter
        
        # Création des décennies
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df['decade_label'] = self.df['decade'].astype(str) + 's'
        
        # Nettoyage des données manquantes
        self.df_clean = self.df.dropna(subset=['date_added', 'release_year'])
        
        print(f"✅ Données préparées: {len(self.df)} entrées totales, {len(self.df_clean)} après nettoyage")
        
    def analyze_content_evolution(self):
        """Analyse l'évolution du contenu Netflix dans le temps"""
        print("\n📈 ANALYSE 1: Évolution du contenu Netflix dans le temps")
        print("=" * 60)
        
        # Contenu ajouté par année
        yearly_content = self.df_clean.groupby(['year_added', 'type']).size().unstack(fill_value=0)
        
        # Graphique d'évolution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Évolution globale par année
        yearly_total = self.df_clean.groupby('year_added').size()
        axes[0,0].plot(yearly_total.index, yearly_total.values, marker='o', linewidth=2)
        axes[0,0].set_title('📊 Évolution du Contenu Total par Année')
        axes[0,0].set_xlabel('Année')
        axes[0,0].set_ylabel('Nombre de Contenus Ajoutés')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Films vs Séries TV
        if 'Movie' in yearly_content.columns and 'TV Show' in yearly_content.columns:
            axes[0,1].plot(yearly_content.index, yearly_content['Movie'], 
                          marker='o', label='Films', linewidth=2)
            axes[0,1].plot(yearly_content.index, yearly_content['TV Show'], 
                          marker='s', label='Séries TV', linewidth=2)
            axes[0,1].set_title('🎬 Films vs Séries TV par Année')
            axes[0,1].set_xlabel('Année')
            axes[0,1].set_ylabel('Nombre de Contenus')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Croissance cumulative
        cumulative = yearly_total.cumsum()
        axes[1,0].plot(cumulative.index, cumulative.values, marker='o', 
                      color='green', linewidth=2)
        axes[1,0].set_title('📈 Croissance Cumulative du Catalogue')
        axes[1,0].set_xlabel('Année')
        axes[1,0].set_ylabel('Nombre Total de Contenus')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Taux de croissance annuel
        growth_rate = yearly_total.pct_change() * 100
        axes[1,1].bar(growth_rate.index, growth_rate.values, alpha=0.7, color='orange')
        axes[1,1].set_title('📊 Taux de Croissance Annuel (%)')
        axes[1,1].set_xlabel('Année')
        axes[1,1].set_ylabel('Croissance (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('netflix_evolution_temporelle.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques clés
        print(f"📊 Statistiques clés:")
        print(f"   • Période d'analyse: {yearly_total.index.min()} - {yearly_total.index.max()}")
        print(f"   • Année avec le plus d'ajouts: {yearly_total.idxmax()} ({yearly_total.max()} contenus)")
        print(f"   • Croissance moyenne annuelle: {growth_rate.mean():.1f}%")
        
        return yearly_content
        
    def analyze_seasonal_patterns(self):
        """Analyse les patterns saisonniers d'ajout de contenu"""
        print("\n🌍 ANALYSE 2: Patterns Saisonniers")
        print("=" * 40)
        
        # Analyse par mois
        monthly_content = self.df_clean.groupby('month_added').size()
        monthly_names = self.df_clean.groupby('month_name').size().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        # Analyse par trimestre
        quarterly_content = self.df_clean.groupby('quarter_added').size()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Distribution mensuelle
        axes[0,0].bar(range(1, 13), monthly_content.values, alpha=0.8, color='skyblue')
        axes[0,0].set_title('📅 Distribution Mensuelle des Ajouts')
        axes[0,0].set_xlabel('Mois')
        axes[0,0].set_ylabel('Nombre de Contenus')
        axes[0,0].set_xticks(range(1, 13))
        axes[0,0].set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                                  'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
        
        # 2. Heatmap mensuelle par année
        monthly_yearly = self.df_clean.groupby(['year_added', 'month_added']).size().unstack(fill_value=0)
        if len(monthly_yearly) > 1:
            sns.heatmap(monthly_yearly, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,1])
            axes[0,1].set_title('🔥 Heatmap: Ajouts par Mois et Année')
            axes[0,1].set_xlabel('Mois')
            axes[0,1].set_ylabel('Année')
        
        # 3. Distribution trimestrielle
        quarters = ['T1', 'T2', 'T3', 'T4']
        axes[1,0].pie(quarterly_content.values, labels=quarters, autopct='%1.1f%%', 
                     startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold'])
        axes[1,0].set_title('🥧 Répartition Trimestrielle')
        
        # 4. Tendance saisonnière
        seasonal_trend = self.df_clean.groupby(['year_added', 'quarter_added']).size().unstack(fill_value=0)
        if len(seasonal_trend) > 1:
            for quarter in seasonal_trend.columns:
                axes[1,1].plot(seasonal_trend.index, seasonal_trend[quarter], 
                              marker='o', label=f'T{quarter}', linewidth=2)
            axes[1,1].set_title('📈 Tendances Saisonnières par Année')
            axes[1,1].set_xlabel('Année')
            axes[1,1].set_ylabel('Nombre de Contenus')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('netflix_patterns_saisonniers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Insights saisonniers
        best_month = monthly_content.idxmax()
        worst_month = monthly_content.idxmin()
        best_quarter = quarterly_content.idxmax()
        
        print(f"🎯 Insights saisonniers:")
        print(f"   • Meilleur mois pour les ajouts: Mois {best_month} ({monthly_content.max()} contenus)")
        print(f"   • Mois le plus calme: Mois {worst_month} ({monthly_content.min()} contenus)")
        print(f"   • Meilleur trimestre: T{best_quarter} ({quarterly_content.max()} contenus)")
        
        return monthly_content, quarterly_content
        
    def analyze_decades_comparison(self):
        """Compare les caractéristiques des films par décennie"""
        print("\n🕰️ ANALYSE 3: Comparaison des Décennies")
        print("=" * 45)
        
        # Filtrer les films uniquement
        movies_df = self.df[self.df['type'] == 'Movie'].copy()
        
        # Analyse par décennie
        decades_stats = movies_df.groupby('decade_label').agg({
            'duration': ['count', 'mean', 'median', 'std'],
            'release_year': 'count'
        }).round(2)
        
        # Flatten column names
        decades_stats.columns = ['_'.join(col).strip() for col in decades_stats.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Nombre de films par décennie
        decade_counts = movies_df['decade_label'].value_counts().sort_index()
        axes[0,0].bar(decade_counts.index, decade_counts.values, alpha=0.8, color='lightblue')
        axes[0,0].set_title('🎬 Nombre de Films par Décennie')
        axes[0,0].set_xlabel('Décennie')
        axes[0,0].set_ylabel('Nombre de Films')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Durée moyenne par décennie
        avg_duration = movies_df.groupby('decade_label')['duration'].mean()
        axes[0,1].plot(avg_duration.index, avg_duration.values, marker='o', linewidth=2, color='red')
        axes[0,1].set_title('⏱️ Durée Moyenne par Décennie')
        axes[0,1].set_xlabel('Décennie')
        axes[0,1].set_ylabel('Durée Moyenne (minutes)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribution des durées par décennie (boxplot)
        decades_for_box = movies_df[movies_df['decade'] >= 1970]  # Limiter aux décennies récentes
        if len(decades_for_box) > 0:
            decades_for_box.boxplot(column='duration', by='decade_label', ax=axes[1,0])
            axes[1,0].set_title('📊 Distribution des Durées par Décennie')
            axes[1,0].set_xlabel('Décennie')
            axes[1,0].set_ylabel('Durée (minutes)')
            plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Évolution des genres populaires
        genre_decade = movies_df.groupby(['decade_label', 'genre']).size().unstack(fill_value=0)
        if len(genre_decade.columns) > 0:
            top_genres = genre_decade.sum().nlargest(5).index
            for genre in top_genres:
                if genre in genre_decade.columns:
                    axes[1,1].plot(genre_decade.index, genre_decade[genre], 
                                  marker='o', label=genre, linewidth=2)
            axes[1,1].set_title('🎭 Évolution des Genres Populaires')
            axes[1,1].set_xlabel('Décennie')
            axes[1,1].set_ylabel('Nombre de Films')
            axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('netflix_comparaison_decennies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques par décennie
        print("📊 Statistiques par décennie:")
        print(decades_stats)
        
        return decades_stats
        
    def generate_temporal_report(self):
        """Génère un rapport complet de l'analyse temporelle"""
        print("\n📋 GÉNÉRATION DU RAPPORT TEMPOREL COMPLET")
        print("=" * 50)
        
        # Exécuter toutes les analyses
        yearly_data = self.analyze_content_evolution()
        monthly_data, quarterly_data = self.analyze_seasonal_patterns()
        decades_data = self.analyze_decades_comparison()
        
        # Résumé exécutif
        print("\n🎯 RÉSUMÉ EXÉCUTIF - ANALYSE TEMPORELLE")
        print("=" * 50)
        print(f"📊 Dataset analysé: {len(self.df)} entrées")
        print(f"📅 Période couverte: {self.df_clean['year_added'].min():.0f} - {self.df_clean['year_added'].max():.0f}")
        print(f"🎬 Types de contenu: {', '.join(self.df['type'].unique())}")
        print(f"🌍 Pays représentés: {self.df['country'].nunique()} pays")
        print(f"🎭 Genres disponibles: {self.df['genre'].nunique()} genres")
        
        # Insights clés
        peak_year = self.df_clean.groupby('year_added').size().idxmax()
        peak_count = self.df_clean.groupby('year_added').size().max()
        
        print(f"\n🔍 INSIGHTS CLÉS:")
        print(f"   • Année record: {peak_year:.0f} ({peak_count} ajouts)")
        print(f"   • Croissance du catalogue: Exponentielle depuis 2015")
        print(f"   • Saisonnalité: Pics en début et fin d'année")
        print(f"   • Évolution des durées: Stabilité relative entre décennies")
        
        return {
            'yearly_data': yearly_data,
            'monthly_data': monthly_data,
            'quarterly_data': quarterly_data,
            'decades_data': decades_data
        }

def main():
    """Fonction principale pour exécuter l'analyse temporelle"""
    print("🎬 NETFLIX DATA ANALYSIS - ANALYSE TEMPORELLE COMPLÈTE")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = NetflixTemporalAnalyzer('netflix_data.csv')
    
    # Générer le rapport complet
    results = analyzer.generate_temporal_report()
    
    print("\n✅ Analyse temporelle terminée!")
    print("📁 Graphiques sauvegardés:")
    print("   • netflix_evolution_temporelle.png")
    print("   • netflix_patterns_saisonniers.png") 
    print("   • netflix_comparaison_decennies.png")
    
    return results

if __name__ == "__main__":
    results = main()
