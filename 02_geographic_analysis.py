"""
🌍 NETFLIX DATA ANALYSIS - PARTIE 2: ANALYSE GÉOGRAPHIQUE
=========================================================

Ce module analyse la distribution géographique du contenu Netflix :
- Cartographie du contenu par pays
- Analyse des marchés dominants par genre
- Diversité culturelle et représentation internationale
- Collaborations internationales

Auteur: Bello Soboure
Date: 2025-01-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class NetflixGeographicAnalyzer:
    """Classe pour l'analyse géographique des données Netflix"""
    
    def __init__(self, csv_path):
        """Initialise l'analyseur avec le dataset Netflix"""
        self.df = pd.read_csv(csv_path)
        self.prepare_geographic_data()
        
    def prepare_geographic_data(self):
        """Prépare les données géographiques pour l'analyse"""
        print("🔄 Préparation des données géographiques...")
        
        # Nettoyage des données de pays
        self.df['country'] = self.df['country'].fillna('Unknown')
        
        # Séparation des pays multiples (ex: "United States, Canada")
        self.countries_expanded = []
        for idx, row in self.df.iterrows():
            countries = str(row['country']).split(', ')
            for country in countries:
                country = country.strip()
                if country and country != 'Unknown':
                    new_row = row.copy()
                    new_row['country_clean'] = country
                    self.countries_expanded.append(new_row)
        
        self.df_expanded = pd.DataFrame(self.countries_expanded)
        
        # Standardisation des noms de pays
        country_mapping = {
            'United States': 'USA',
            'United Kingdom': 'UK',
            'South Korea': 'Korea',
            'Soviet Union': 'Russia',
        }
        
        self.df_expanded['country_clean'] = self.df_expanded['country_clean'].replace(country_mapping)
        
        # Création des décennies pour l'analyse temporelle
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df_expanded['decade'] = (self.df_expanded['release_year'] // 10) * 10
        
        print(f"✅ Données géographiques préparées: {len(self.df_expanded)} entrées avec pays définis")
        print(f"📍 Nombre de pays uniques: {self.df_expanded['country_clean'].nunique()}")
        
    def analyze_content_distribution(self):
        """Analyse la distribution du contenu par pays"""
        print("\n🗺️ ANALYSE 1: Distribution Géographique du Contenu")
        print("=" * 55)
        
        # Top pays producteurs
        country_counts = self.df_expanded['country_clean'].value_counts()
        top_countries = country_counts.head(20)
        
        # Analyse par type de contenu
        country_type = self.df_expanded.groupby(['country_clean', 'type']).size().unstack(fill_value=0)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Top 15 pays producteurs
        top_countries.head(15).plot(kind='barh', ax=axes[0,0], color='lightcoral')
        axes[0,0].set_title('🏆 Top 15 Pays Producteurs de Contenu Netflix')
        axes[0,0].set_xlabel('Nombre de Productions')
        axes[0,0].set_ylabel('Pays')
        
        # 2. Répartition Films vs Séries par pays (top 10)
        top_10_countries = top_countries.head(10).index
        country_type_top = country_type.loc[top_10_countries]
        
        if 'Movie' in country_type_top.columns and 'TV Show' in country_type_top.columns:
            country_type_top[['Movie', 'TV Show']].plot(kind='bar', ax=axes[0,1], 
                                                        color=['skyblue', 'lightgreen'])
            axes[0,1].set_title('🎬 Films vs Séries TV par Pays (Top 10)')
            axes[0,1].set_xlabel('Pays')
            axes[0,1].set_ylabel('Nombre de Productions')
            axes[0,1].legend(['Films', 'Séries TV'])
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Distribution cumulative
        cumulative_dist = country_counts.cumsum() / country_counts.sum() * 100
        axes[1,0].plot(range(1, len(cumulative_dist) + 1), cumulative_dist.values, 
                      marker='o', linewidth=2, color='purple')
        axes[1,0].set_title('📈 Distribution Cumulative par Pays')
        axes[1,0].set_xlabel('Rang du Pays')
        axes[1,0].set_ylabel('Pourcentage Cumulé (%)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%')
        axes[1,0].legend()
        
        # 4. Concentration géographique (Pie chart des continents)
        # Mapping pays -> continents (simplifié)
        continent_mapping = {
            'USA': 'Amérique du Nord', 'Canada': 'Amérique du Nord', 'Mexico': 'Amérique du Nord',
            'UK': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe', 
            'Italy': 'Europe', 'Netherlands': 'Europe', 'Belgium': 'Europe', 'Sweden': 'Europe',
            'Norway': 'Europe', 'Denmark': 'Europe', 'Poland': 'Europe', 'Russia': 'Europe',
            'India': 'Asie', 'China': 'Asie', 'Japan': 'Asie', 'Korea': 'Asie', 
            'Thailand': 'Asie', 'Philippines': 'Asie', 'Indonesia': 'Asie', 'Malaysia': 'Asie',
            'Singapore': 'Asie', 'Taiwan': 'Asie', 'Hong Kong': 'Asie',
            'Brazil': 'Amérique du Sud', 'Argentina': 'Amérique du Sud', 'Chile': 'Amérique du Sud',
            'Colombia': 'Amérique du Sud', 'Peru': 'Amérique du Sud', 'Uruguay': 'Amérique du Sud',
            'Egypt': 'Afrique', 'South Africa': 'Afrique', 'Nigeria': 'Afrique', 'Morocco': 'Afrique',
            'Australia': 'Océanie', 'New Zealand': 'Océanie'
        }
        
        self.df_expanded['continent'] = self.df_expanded['country_clean'].map(continent_mapping)
        self.df_expanded['continent'] = self.df_expanded['continent'].fillna('Autres')
        
        continent_counts = self.df_expanded['continent'].value_counts()
        axes[1,1].pie(continent_counts.values, labels=continent_counts.index, autopct='%1.1f%%',
                     startangle=90, colors=plt.cm.Set3.colors)
        axes[1,1].set_title('🌍 Répartition par Continent')
        
        plt.tight_layout()
        plt.savefig('netflix_distribution_geographique.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques clés
        total_countries = len(country_counts)
        top_5_share = (country_counts.head(5).sum() / country_counts.sum()) * 100
        
        print(f"📊 Statistiques géographiques:")
        print(f"   • Nombre total de pays: {total_countries}")
        print(f"   • Top 5 pays représentent: {top_5_share:.1f}% du contenu")
        print(f"   • Pays dominant: {country_counts.index[0]} ({country_counts.iloc[0]} productions)")
        print(f"   • Diversité géographique: {total_countries} pays représentés")
        
        return country_counts, continent_counts
        
    def analyze_genre_by_region(self):
        """Analyse les genres dominants par région"""
        print("\n🎭 ANALYSE 2: Genres Dominants par Région")
        print("=" * 45)
        
        # Top 10 pays pour l'analyse des genres
        top_countries = self.df_expanded['country_clean'].value_counts().head(10).index
        df_top_countries = self.df_expanded[self.df_expanded['country_clean'].isin(top_countries)]
        
        # Matrice pays x genres
        country_genre = df_top_countries.groupby(['country_clean', 'genre']).size().unstack(fill_value=0)
        
        # Normalisation par pays (pourcentages)
        country_genre_pct = country_genre.div(country_genre.sum(axis=1), axis=0) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Heatmap des genres par pays
        top_genres = country_genre.sum().nlargest(15).index
        country_genre_top = country_genre[top_genres]
        
        sns.heatmap(country_genre_top, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('🔥 Heatmap: Genres par Pays (Nombre absolu)')
        axes[0,0].set_xlabel('Genres')
        axes[0,0].set_ylabel('Pays')
        
        # 2. Heatmap en pourcentages
        country_genre_pct_top = country_genre_pct[top_genres]
        sns.heatmap(country_genre_pct_top, annot=True, fmt='.1f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('📊 Heatmap: Genres par Pays (Pourcentages)')
        axes[0,1].set_xlabel('Genres')
        axes[0,1].set_ylabel('Pays')
        
        # 3. Spécialisation par pays (genre dominant)
        dominant_genres = country_genre_pct.idxmax(axis=1)
        dominant_pct = country_genre_pct.max(axis=1)
        
        specialization_df = pd.DataFrame({
            'Pays': dominant_genres.index,
            'Genre_Dominant': dominant_genres.values,
            'Pourcentage': dominant_pct.values
        }).sort_values('Pourcentage', ascending=True)
        
        axes[1,0].barh(specialization_df['Pays'], specialization_df['Pourcentage'], 
                      color='lightgreen', alpha=0.8)
        axes[1,0].set_title('🎯 Spécialisation par Pays (Genre Dominant)')
        axes[1,0].set_xlabel('Pourcentage du Genre Dominant')
        axes[1,0].set_ylabel('Pays')
        
        # 4. Diversité des genres par pays
        genre_diversity = (country_genre > 0).sum(axis=1).sort_values(ascending=True)
        axes[1,1].barh(genre_diversity.index, genre_diversity.values, 
                      color='orange', alpha=0.8)
        axes[1,1].set_title('🌈 Diversité des Genres par Pays')
        axes[1,1].set_xlabel('Nombre de Genres Différents')
        axes[1,1].set_ylabel('Pays')
        
        plt.tight_layout()
        plt.savefig('netflix_genres_par_region.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Insights sur les spécialisations
        print("🎭 Spécialisations régionales:")
        for country, genre in dominant_genres.head(10).items():
            pct = dominant_pct[country]
            print(f"   • {country}: {genre} ({pct:.1f}%)")
            
        return country_genre, dominant_genres
        
    def analyze_cultural_diversity(self):
        """Analyse la diversité culturelle et les collaborations"""
        print("\n🤝 ANALYSE 3: Diversité Culturelle et Collaborations")
        print("=" * 55)
        
        # Collaborations internationales (pays multiples)
        multi_country = self.df[self.df['country'].str.contains(',', na=False)]
        
        # Analyse des collaborations
        collaborations = []
        for countries_str in multi_country['country']:
            countries = [c.strip() for c in str(countries_str).split(',')]
            if len(countries) > 1:
                collaborations.append(countries)
        
        # Paires de pays les plus fréquentes
        country_pairs = []
        for collab in collaborations:
            for i in range(len(collab)):
                for j in range(i+1, len(collab)):
                    pair = tuple(sorted([collab[i], collab[j]]))
                    country_pairs.append(pair)
        
        pair_counts = Counter(country_pairs)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Productions avec collaborations internationales
        collab_by_year = multi_country.groupby('release_year').size()
        axes[0,0].plot(collab_by_year.index, collab_by_year.values, marker='o', linewidth=2)
        axes[0,0].set_title('🤝 Collaborations Internationales par Année')
        axes[0,0].set_xlabel('Année de Sortie')
        axes[0,0].set_ylabel('Nombre de Collaborations')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Top paires de pays collaborateurs
        if pair_counts:
            top_pairs = dict(list(pair_counts.most_common(10)))
            pair_labels = [f"{p[0]} - {p[1]}" for p in top_pairs.keys()]
            axes[0,1].barh(pair_labels, list(top_pairs.values()), color='lightblue')
            axes[0,1].set_title('🌍 Top 10 Collaborations Bilatérales')
            axes[0,1].set_xlabel('Nombre de Collaborations')
        
        # 3. Diversité par décennie
        decade_diversity = self.df_expanded.groupby('decade')['country_clean'].nunique()
        axes[1,0].bar(decade_diversity.index, decade_diversity.values, alpha=0.8, color='gold')
        axes[1,0].set_title('📈 Diversité Géographique par Décennie')
        axes[1,0].set_xlabel('Décennie')
        axes[1,0].set_ylabel('Nombre de Pays Représentés')
        
        # 4. Index de diversité culturelle par genre
        genre_diversity = self.df_expanded.groupby('genre')['country_clean'].nunique().sort_values(ascending=False)
        top_diverse_genres = genre_diversity.head(15)
        axes[1,1].barh(range(len(top_diverse_genres)), top_diverse_genres.values, color='lightcoral')
        axes[1,1].set_yticks(range(len(top_diverse_genres)))
        axes[1,1].set_yticklabels(top_diverse_genres.index)
        axes[1,1].set_title('🎭 Diversité Géographique par Genre')
        axes[1,1].set_xlabel('Nombre de Pays Représentés')
        
        plt.tight_layout()
        plt.savefig('netflix_diversite_culturelle.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques de diversité
        total_collaborations = len(multi_country)
        collab_percentage = (total_collaborations / len(self.df)) * 100
        
        print(f"🌍 Statistiques de diversité:")
        print(f"   • Productions avec collaborations: {total_collaborations} ({collab_percentage:.1f}%)")
        print(f"   • Paires de pays les plus collaboratives: {list(pair_counts.most_common(3))}")
        print(f"   • Genre le plus diversifié: {genre_diversity.index[0]} ({genre_diversity.iloc[0]} pays)")
        
        return pair_counts, genre_diversity
        
    def create_geographic_insights(self):
        """Crée des insights géographiques avancés"""
        print("\n🔍 ANALYSE 4: Insights Géographiques Avancés")
        print("=" * 50)
        
        # Analyse de la concentration géographique
        country_counts = self.df_expanded['country_clean'].value_counts()
        
        # Coefficient de Gini pour mesurer la concentration
        def gini_coefficient(x):
            """Calcule le coefficient de Gini"""
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        gini = gini_coefficient(country_counts.values)
        
        # Analyse des marchés émergents
        recent_years = self.df_expanded[self.df_expanded['release_year'] >= 2015]
        emerging_countries = recent_years['country_clean'].value_counts()
        traditional_countries = self.df_expanded[self.df_expanded['release_year'] < 2015]['country_clean'].value_counts()
        
        # Croissance par pays
        growth_analysis = []
        for country in emerging_countries.index[:20]:
            recent_count = emerging_countries.get(country, 0)
            traditional_count = traditional_countries.get(country, 0)
            if traditional_count > 0:
                growth_rate = ((recent_count - traditional_count) / traditional_count) * 100
            else:
                growth_rate = float('inf') if recent_count > 0 else 0
            growth_analysis.append({
                'country': country,
                'recent_productions': recent_count,
                'traditional_productions': traditional_count,
                'growth_rate': growth_rate
            })
        
        growth_df = pd.DataFrame(growth_analysis)
        growth_df = growth_df[growth_df['growth_rate'] != float('inf')].sort_values('growth_rate', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Courbe de Lorenz (concentration géographique)
        sorted_counts = np.sort(country_counts.values)
        cumulative_counts = np.cumsum(sorted_counts)
        cumulative_pct = cumulative_counts / cumulative_counts[-1]
        country_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        
        axes[0,0].plot(country_pct, cumulative_pct, linewidth=2, label='Courbe de Lorenz')
        axes[0,0].plot([0, 1], [0, 1], '--', color='red', alpha=0.7, label='Égalité parfaite')
        axes[0,0].fill_between(country_pct, cumulative_pct, country_pct, alpha=0.3)
        axes[0,0].set_title(f'📊 Concentration Géographique (Gini: {gini:.3f})')
        axes[0,0].set_xlabel('Proportion Cumulative des Pays')
        axes[0,0].set_ylabel('Proportion Cumulative du Contenu')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Marchés émergents vs traditionnels
        top_growth = growth_df.head(10)
        axes[0,1].barh(top_growth['country'], top_growth['growth_rate'], color='lightgreen')
        axes[0,1].set_title('🚀 Top 10 Marchés en Croissance')
        axes[0,1].set_xlabel('Taux de Croissance (%)')
        
        # 3. Évolution de la diversité dans le temps
        yearly_diversity = self.df_expanded.groupby('release_year')['country_clean'].nunique()
        axes[1,0].plot(yearly_diversity.index, yearly_diversity.values, marker='o', linewidth=2)
        axes[1,0].set_title('📈 Évolution de la Diversité Géographique')
        axes[1,0].set_xlabel('Année')
        axes[1,0].set_ylabel('Nombre de Pays Représentés')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Parts de marché par région
        continent_evolution = self.df_expanded.groupby(['release_year', 'continent']).size().unstack(fill_value=0)
        continent_pct = continent_evolution.div(continent_evolution.sum(axis=1), axis=0) * 100
        
        for continent in continent_pct.columns:
            if continent != 'Autres':
                axes[1,1].plot(continent_pct.index, continent_pct[continent], 
                              marker='o', label=continent, linewidth=2)
        axes[1,1].set_title('🌍 Évolution des Parts de Marché par Continent')
        axes[1,1].set_xlabel('Année')
        axes[1,1].set_ylabel('Part de Marché (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('netflix_insights_geographiques.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Insights géographiques:")
        print(f"   • Coefficient de Gini: {gini:.3f} (0=égalité parfaite, 1=concentration maximale)")
        print(f"   • Marché en plus forte croissance: {growth_df.iloc[0]['country']} (+{growth_df.iloc[0]['growth_rate']:.1f}%)")
        print(f"   • Diversité géographique: {yearly_diversity.iloc[-1]} pays en {yearly_diversity.index[-1]}")
        
        return gini, growth_df
        
    def generate_geographic_report(self):
        """Génère un rapport complet de l'analyse géographique"""
        print("\n📋 GÉNÉRATION DU RAPPORT GÉOGRAPHIQUE COMPLET")
        print("=" * 55)
        
        # Exécuter toutes les analyses
        country_counts, continent_counts = self.analyze_content_distribution()
        country_genre, dominant_genres = self.analyze_genre_by_region()
        collaborations, genre_diversity = self.analyze_cultural_diversity()
        gini, growth_data = self.create_geographic_insights()
        
        # Résumé exécutif
        print("\n🎯 RÉSUMÉ EXÉCUTIF - ANALYSE GÉOGRAPHIQUE")
        print("=" * 55)
        print(f"🌍 Pays analysés: {len(country_counts)} pays")
        print(f"🏆 Pays dominant: {country_counts.index[0]} ({country_counts.iloc[0]} productions)")
        print(f"📊 Concentration géographique: {gini:.3f}")
        print(f"🤝 Collaborations internationales: {len(collaborations)} paires de pays")
        
        # Recommandations stratégiques
        print(f"\n💡 RECOMMANDATIONS STRATÉGIQUES:")
        print(f"   • Diversification: Explorer les marchés émergents identifiés")
        print(f"   • Collaborations: Renforcer les partenariats internationaux")
        print(f"   • Localisation: Adapter le contenu aux préférences régionales")
        print(f"   • Expansion: Cibler les régions sous-représentées")
        
        return {
            'country_distribution': country_counts,
            'continent_distribution': continent_counts,
            'genre_specialization': dominant_genres,
            'collaborations': collaborations,
            'geographic_concentration': gini,
            'growth_markets': growth_data
        }

def main():
    """Fonction principale pour exécuter l'analyse géographique"""
    print("🌍 NETFLIX DATA ANALYSIS - ANALYSE GÉOGRAPHIQUE COMPLÈTE")
    print("=" * 65)
    
    # Initialiser l'analyseur
    analyzer = NetflixGeographicAnalyzer('netflix_data.csv')
    
    # Générer le rapport complet
    results = analyzer.generate_geographic_report()
    
    print("\n✅ Analyse géographique terminée!")
    print("📁 Graphiques sauvegardés:")
    print("   • netflix_distribution_geographique.png")
    print("   • netflix_genres_par_region.png")
    print("   • netflix_diversite_culturelle.png")
    print("   • netflix_insights_geographiques.png")
    
    return results

if __name__ == "__main__":
    results = main()
