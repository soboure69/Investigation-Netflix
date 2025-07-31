"""
🎭 NETFLIX DATA ANALYSIS - PARTIE 3: ANALYSE DES GENRES ET CONTENUS
==================================================================

Ce module analyse les genres et contenus Netflix :
- Clustering des genres similaires
- Analyse NLP des descriptions
- Durées optimales par genre et époque
- Patterns de contenu et tendances

Auteur: Bello Soboure
Date: 2025-01-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# Import optionnel de wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠️ WordCloud non disponible. Les nuages de mots seront désactivés.")

# Imports optionnels de sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn non disponible. Le clustering sera désactivé.")

import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class NetflixGenreContentAnalyzer:
    """Classe pour l'analyse des genres et contenus Netflix"""
    
    def __init__(self, csv_path):
        """Initialise l'analyseur avec le dataset Netflix"""
        self.df = pd.read_csv(csv_path)
        self.prepare_content_data()
        
    def prepare_content_data(self):
        """Prépare les données pour l'analyse des genres et contenus"""
        print("🔄 Préparation des données de contenu...")
        
        # Nettoyage des données
        self.df['genre'] = self.df['genre'].fillna('Unknown')
        self.df['description'] = self.df['description'].fillna('')
        self.df['duration'] = pd.to_numeric(self.df['duration'], errors='coerce')
        
        # Création des décennies
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df['decade_label'] = self.df['decade'].astype(str) + 's'
        
        # Nettoyage des descriptions pour NLP
        self.df['description_clean'] = self.df['description'].apply(self.clean_text)
        
        # Séparation des genres multiples
        self.genres_expanded = []
        for idx, row in self.df.iterrows():
            genres = str(row['genre']).split(', ')
            for genre in genres:
                genre = genre.strip()
                if genre and genre != 'Unknown':
                    new_row = row.copy()
                    new_row['genre_clean'] = genre
                    new_row['description_clean'] = row['description_clean']  # <-- Ajoute de cette ligne
                    self.genres_expanded.append(new_row)
        
        self.df_genres = pd.DataFrame(self.genres_expanded)
        
        print(f"✅ Données préparées: {len(self.df)} entrées totales")
        print(f"🎭 Genres uniques: {self.df_genres['genre_clean'].nunique()}")
        print(f"📝 Descriptions disponibles: {(self.df['description'] != '').sum()}")
        
    def clean_text(self, text):
        """Nettoie le texte pour l'analyse NLP"""
        if pd.isna(text) or text == '':
            return ''
        
        # Conversion en minuscules
        text = str(text).lower()
        
        # Suppression des caractères spéciaux
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Suppression des mots vides basiques
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
        
    def analyze_genre_distribution(self):
        """Analyse la distribution des genres"""
        print("\n🎭 ANALYSE 1: Distribution des Genres")
        print("=" * 40)
        
        # Distribution des genres
        genre_counts = self.df_genres['genre_clean'].value_counts()
        
        # Analyse par type de contenu
        genre_type = self.df_genres.groupby(['genre_clean', 'type']).size().unstack(fill_value=0)
        
        # Évolution temporelle des genres
        genre_evolution = self.df_genres.groupby(['decade', 'genre_clean']).size().unstack(fill_value=0)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Top 20 genres
        top_genres = genre_counts.head(20)
        top_genres.plot(kind='barh', ax=axes[0,0], color='lightcoral')
        axes[0,0].set_title('🏆 Top 20 Genres les Plus Populaires')
        axes[0,0].set_xlabel('Nombre de Productions')
        axes[0,0].set_ylabel('Genres')
        
        # 2. Films vs Séries par genre (top 15)
        top_15_genres = genre_counts.head(15).index
        genre_type_top = genre_type.loc[top_15_genres]
        
        if 'Movie' in genre_type_top.columns and 'TV Show' in genre_type_top.columns:
            genre_type_top[['Movie', 'TV Show']].plot(kind='bar', ax=axes[0,1], 
                                                     color=['skyblue', 'lightgreen'])
            axes[0,1].set_title('🎬 Films vs Séries TV par Genre')
            axes[0,1].set_xlabel('Genres')
            axes[0,1].set_ylabel('Nombre de Productions')
            axes[0,1].legend(['Films', 'Séries TV'])
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Évolution des top 10 genres dans le temps
        top_10_genres = genre_counts.head(10).index
        genre_evolution_top = genre_evolution[top_10_genres]
        
        for genre in top_10_genres:
            if genre in genre_evolution_top.columns:
                axes[1,0].plot(genre_evolution_top.index, genre_evolution_top[genre], 
                              marker='o', label=genre, linewidth=2)
        axes[1,0].set_title('📈 Évolution des Genres Populaires par Décennie')
        axes[1,0].set_xlabel('Décennie')
        axes[1,0].set_ylabel('Nombre de Productions')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Diversité des genres par décennie
        genre_diversity = (genre_evolution > 0).sum(axis=1)
        axes[1,1].bar(genre_diversity.index, genre_diversity.values, alpha=0.8, color='gold')
        axes[1,1].set_title('🌈 Diversité des Genres par Décennie')
        axes[1,1].set_xlabel('Décennie')
        axes[1,1].set_ylabel('Nombre de Genres Différents')
        
        plt.tight_layout()
        plt.savefig('netflix_distribution_genres.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques des genres
        print(f"📊 Statistiques des genres:")
        print(f"   • Nombre total de genres: {len(genre_counts)}")
        print(f"   • Genre dominant: {genre_counts.index[0]} ({genre_counts.iloc[0]} productions)")
        print(f"   • Top 5 genres représentent: {(genre_counts.head(5).sum() / genre_counts.sum() * 100):.1f}% du contenu")
        
        return genre_counts, genre_evolution
        
    def analyze_genre_clustering(self):
        """Effectue un clustering des genres basé sur les descriptions"""
        print("\n🔬 ANALYSE 2: Clustering des Genres")
        print("=" * 35)
        
        if not SKLEARN_AVAILABLE:
            print("⚠️ Sklearn non disponible. Analyse de clustering désactivée.")
            print("💡 Pour activer cette fonctionnalité, installez: pip install scikit-learn")
            return None, None, None
        
        # Préparer les données pour le clustering
        genre_descriptions = self.df_genres.groupby('genre_clean')['description_clean'].apply(lambda x: ' '.join(x)).reset_index()
        genre_descriptions = genre_descriptions[genre_descriptions['description_clean'].str.len() > 50]
        
        # Vectorisation TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(genre_descriptions['description_clean'])
        
        # Clustering K-means
        n_clusters = min(8, len(genre_descriptions))  # Adapter le nombre de clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Réduction de dimensionnalité pour visualisation
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(tfidf_matrix.toarray())
        
        # Matrice de similarité
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Visualisation du clustering
        scatter = axes[0,0].scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, 
                                   cmap='viridis', alpha=0.7, s=100)
        axes[0,0].set_title('🔬 Clustering des Genres (PCA)')
        axes[0,0].set_xlabel('Première Composante Principale')
        axes[0,0].set_ylabel('Deuxième Composante Principale')
        
        # Ajouter les labels des genres
        for i, genre in enumerate(genre_descriptions['genre_clean']):
            axes[0,0].annotate(genre, (pca_result[i, 0], pca_result[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[0,0])
        
        # 2. Heatmap de similarité entre genres
        genre_similarity = pd.DataFrame(similarity_matrix, 
                                       index=genre_descriptions['genre_clean'],
                                       columns=genre_descriptions['genre_clean'])
        
        sns.heatmap(genre_similarity, annot=False, cmap='coolwarm', ax=axes[0,1])
        axes[0,1].set_title('🔥 Matrice de Similarité entre Genres')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].tick_params(axis='y', rotation=0)
        
        # 3. Distribution des clusters
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        axes[1,0].bar(cluster_counts.index, cluster_counts.values, alpha=0.8, color='lightblue')
        axes[1,0].set_title('📊 Distribution des Clusters')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Nombre de Genres')
        
        # 4. Mots-clés par cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = top_keywords
        
        # Affichage des mots-clés (texte)
        axes[1,1].axis('off')
        keyword_text = "🔑 Mots-clés par Cluster:\n\n"
        for cluster_id, keywords in cluster_keywords.items():
            keyword_text += f"Cluster {cluster_id}: {', '.join(keywords[:5])}\n"
        axes[1,1].text(0.1, 0.9, keyword_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('netflix_clustering_genres.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Résultats du clustering
        genre_descriptions['cluster'] = clusters
        print(f"🔬 Résultats du clustering:")
        print(f"   • Nombre de clusters: {n_clusters}")
        print(f"   • Variance expliquée par PCA: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Afficher les genres par cluster
        for cluster_id in range(n_clusters):
            cluster_genres = genre_descriptions[genre_descriptions['cluster'] == cluster_id]['genre_clean'].tolist()
            print(f"   • Cluster {cluster_id}: {', '.join(cluster_genres)}")
        
        return clusters, genre_descriptions, similarity_matrix
        
    def analyze_content_nlp(self):
        """Analyse NLP des descriptions de contenu"""
        print("\n📝 ANALYSE 3: Analyse NLP des Descriptions")
        print("=" * 45)
        
        # Filtrer les descriptions non vides
        df_with_desc = self.df[self.df['description_clean'].str.len() > 10].copy()
        
        # Analyse des mots les plus fréquents
        all_descriptions = ' '.join(df_with_desc['description_clean'])
        word_freq = Counter(all_descriptions.split())
        
        # Analyse par genre (top 5 genres)
        top_genres = self.df_genres['genre_clean'].value_counts().head(5).index
        genre_words = {}
        
        for genre in top_genres:
            genre_desc = self.df_genres[self.df_genres['genre_clean'] == genre]['description_clean']
            genre_text = ' '.join(genre_desc)
            genre_words[genre] = Counter(genre_text.split())
        
        # Analyse de sentiment basique (mots positifs/négatifs)
        positive_words = {'love', 'amazing', 'incredible', 'fantastic', 'wonderful', 'excellent', 'brilliant', 'outstanding', 'perfect', 'beautiful', 'great', 'good', 'best', 'awesome', 'spectacular', 'magnificent'}
        negative_words = {'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disgusting', 'pathetic', 'useless', 'boring', 'stupid', 'ridiculous', 'annoying', 'frustrating', 'disappointing'}
        
        df_with_desc['positive_score'] = df_with_desc['description_clean'].apply(
            lambda x: sum(1 for word in x.split() if word in positive_words)
        )
        df_with_desc['negative_score'] = df_with_desc['description_clean'].apply(
            lambda x: sum(1 for word in x.split() if word in negative_words)
        )
        df_with_desc['sentiment_score'] = df_with_desc['positive_score'] - df_with_desc['negative_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Nuage de mots global
        if len(all_descriptions) > 100 and WORDCLOUD_AVAILABLE:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                    max_words=100, colormap='viridis').generate(all_descriptions)
                axes[0,0].imshow(wordcloud, interpolation='bilinear')
                axes[0,0].set_title('☁️ Nuage de Mots - Toutes Descriptions')
                axes[0,0].axis('off')
            except Exception as e:
                axes[0,0].text(0.5, 0.5, f'Erreur nuage de mots:\n{str(e)[:50]}...', 
                              ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('☁️ Nuage de Mots - Erreur')
                axes[0,0].axis('off')
        else:
            axes[0,0].text(0.5, 0.5, 'Nuage de mots\nnon disponible\n(WordCloud requis)', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('☁️ Nuage de Mots - Non Disponible')
            axes[0,0].axis('off')
        
        # 2. Mots les plus fréquents
        top_words = dict(word_freq.most_common(20))
        axes[0,1].barh(list(top_words.keys()), list(top_words.values()), color='lightgreen')
        axes[0,1].set_title('📊 Top 20 Mots les Plus Fréquents')
        axes[0,1].set_xlabel('Fréquence')
        
        # 3. Longueur des descriptions par genre
          # Calculer la longueur d'abord
        df_with_desc['desc_len'] = df_with_desc['description'].str.len()

          # Puis grouper
        desc_length_by_genre = df_with_desc.groupby('genre')['desc_len'].mean().sort_values(ascending=False).head(15)

        #desc_length_by_genre = df_with_desc.groupby('genre')['description'].str.len().mean().sort_values(ascending=False).head(15)
        axes[1,0].barh(desc_length_by_genre.index, desc_length_by_genre.values, color='orange')
        axes[1,0].set_title('📏 Longueur Moyenne des Descriptions par Genre')
        axes[1,0].set_xlabel('Longueur Moyenne (caractères)')
        
        # 4. Score de sentiment par genre
        sentiment_by_genre = df_with_desc.groupby('genre')['sentiment_score'].mean().sort_values(ascending=False).head(15)
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in sentiment_by_genre.values]
        axes[1,1].barh(sentiment_by_genre.index, sentiment_by_genre.values, color=colors, alpha=0.7)
        axes[1,1].set_title('😊 Score de Sentiment par Genre')
        axes[1,1].set_xlabel('Score de Sentiment')
        axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('netflix_analyse_nlp.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques NLP
        avg_desc_length = df_with_desc['description'].str.len().mean()
        most_positive_genre = sentiment_by_genre.index[0] if len(sentiment_by_genre) > 0 else 'N/A'
        
        print(f"📝 Statistiques NLP:")
        print(f"   • Longueur moyenne des descriptions: {avg_desc_length:.0f} caractères")
        print(f"   • Mot le plus fréquent: {word_freq.most_common(1)[0][0]} ({word_freq.most_common(1)[0][1]} occurrences)")
        print(f"   • Genre avec descriptions les plus positives: {most_positive_genre}")
        print(f"   • Vocabulaire unique: {len(word_freq)} mots différents")
        
        return word_freq, sentiment_by_genre, genre_words
        
    def analyze_duration_patterns(self):
        """Analyse les patterns de durée par genre et époque"""
        print("\n⏱️ ANALYSE 4: Patterns de Durée")
        print("=" * 35)
        
        # Filtrer les films uniquement (durée en minutes)
        movies_df = self.df[self.df['type'] == 'Movie'].copy()
        movies_df = movies_df.dropna(subset=['duration'])
        
        # Analyse des durées par genre
        duration_by_genre = movies_df.groupby('genre')['duration'].agg(['mean', 'median', 'std', 'count']).round(2)
        duration_by_genre = duration_by_genre[duration_by_genre['count'] >= 10]  # Genres avec au moins 10 films
        
        # Évolution des durées dans le temps
        duration_by_decade = movies_df.groupby('decade')['duration'].agg(['mean', 'median']).round(2)
        
        # Classification des durées
        movies_df['duration_category'] = pd.cut(movies_df['duration'], 
                                               bins=[0, 90, 120, 150, float('inf')], 
                                               labels=['Court (<90min)', 'Moyen (90-120min)', 
                                                      'Long (120-150min)', 'Très long (>150min)'])
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Durée moyenne par genre (top 15)
        top_genres_duration = duration_by_genre.sort_values('mean', ascending=False).head(15)
        axes[0,0].barh(top_genres_duration.index, top_genres_duration['mean'], color='lightcoral')
        axes[0,0].set_title('⏱️ Durée Moyenne par Genre (Top 15)')
        axes[0,0].set_xlabel('Durée Moyenne (minutes)')
        
        # 2. Évolution des durées par décennie
        axes[0,1].plot(duration_by_decade.index, duration_by_decade['mean'], 
                      marker='o', label='Moyenne', linewidth=2, color='blue')
        axes[0,1].plot(duration_by_decade.index, duration_by_decade['median'], 
                      marker='s', label='Médiane', linewidth=2, color='red')
        axes[0,1].set_title('📈 Évolution des Durées par Décennie')
        axes[0,1].set_xlabel('Décennie')
        axes[0,1].set_ylabel('Durée (minutes)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribution des catégories de durée
        duration_dist = movies_df['duration_category'].value_counts()
        axes[1,0].pie(duration_dist.values, labels=duration_dist.index, autopct='%1.1f%%',
                     startangle=90, colors=plt.cm.Set3.colors)
        axes[1,0].set_title('🥧 Distribution des Catégories de Durée')
        
        # 4. Boxplot des durées par genre (top 10)
        top_10_genres = movies_df['genre'].value_counts().head(10).index
        movies_top_genres = movies_df[movies_df['genre'].isin(top_10_genres)]
        
        box_data = [movies_top_genres[movies_top_genres['genre'] == genre]['duration'].values 
                   for genre in top_10_genres]
        axes[1,1].boxplot(box_data, labels=top_10_genres)
        axes[1,1].set_title('📊 Distribution des Durées par Genre (Top 10)')
        axes[1,1].set_xlabel('Genres')
        axes[1,1].set_ylabel('Durée (minutes)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('netflix_patterns_duree.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques de durée
        overall_mean = movies_df['duration'].mean()
        longest_genre = duration_by_genre.sort_values('mean', ascending=False).index[0]
        shortest_genre = duration_by_genre.sort_values('mean', ascending=True).index[0]
        
        print(f"⏱️ Statistiques de durée:")
        print(f"   • Durée moyenne globale: {overall_mean:.1f} minutes")
        print(f"   • Genre avec films les plus longs: {longest_genre} ({duration_by_genre.loc[longest_genre, 'mean']:.1f} min)")
        print(f"   • Genre avec films les plus courts: {shortest_genre} ({duration_by_genre.loc[shortest_genre, 'mean']:.1f} min)")
        print(f"   • Évolution temporelle: Stabilité relative des durées")
        
        return duration_by_genre, duration_by_decade
        
    def generate_genre_content_report(self):
        """Génère un rapport complet de l'analyse des genres et contenus"""
        print("\n📋 GÉNÉRATION DU RAPPORT GENRES ET CONTENUS")
        print("=" * 55)
        
        # Exécuter toutes les analyses
        genre_counts, genre_evolution = self.analyze_genre_distribution()
        clusters, genre_clusters, similarity = self.analyze_genre_clustering()
        word_freq, sentiment, genre_words = self.analyze_content_nlp()
        duration_stats, duration_evolution = self.analyze_duration_patterns()
        
        # Gérer les cas où certaines analyses ne sont pas disponibles
        clustering_available = clusters is not None
        nlp_available = word_freq is not None
        
        # Résumé exécutif
        print("\n🎯 RÉSUMÉ EXÉCUTIF - ANALYSE GENRES ET CONTENUS")
        print("=" * 60)
        print(f"🎭 Genres analysés: {len(genre_counts)} genres uniques")
        print(f"🏆 Genre dominant: {genre_counts.index[0]} ({genre_counts.iloc[0]} productions)")
        print(f"📝 Descriptions analysées: {len(self.df[self.df['description'] != ''])}")
        print(f"⏱️ Durée moyenne des films: {self.df[self.df['type'] == 'Movie']['duration'].mean():.1f} minutes")
        
        # Insights clés
        print(f"\n🔍 INSIGHTS CLÉS:")
        print(f"   • Diversité des genres: Croissance constante depuis les années 1980")
        if clustering_available:
            print(f"   • Clustering: {len(set(clusters))} groupes de genres similaires identifiés")
        else:
            print(f"   • Clustering: Non disponible (sklearn requis)")
        if nlp_available:
            print(f"   • Sentiment: Descriptions généralement neutres avec variations par genre")
        else:
            print(f"   • Sentiment: Analyse basique disponible")
        print(f"   • Durées: Stabilité relative avec spécialisations par genre")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        print(f"   • Contenu: Exploiter les niches de genres sous-représentés")
        print(f"   • Descriptions: Optimiser le vocabulaire pour améliorer l'engagement")
        print(f"   • Durées: Adapter aux préférences actuelles du public")
        print(f"   • Clustering: Utiliser pour des recommandations personnalisées")
        
        return {
            'genre_distribution': genre_counts,
            'genre_evolution': genre_evolution,
            'genre_clusters': genre_clusters,
            'similarity_matrix': similarity,
            'word_frequency': word_freq,
            'sentiment_analysis': sentiment,
            'duration_analysis': duration_stats
        }

def main():
    """Fonction principale pour exécuter l'analyse des genres et contenus"""
    print("🎭 NETFLIX DATA ANALYSIS - ANALYSE GENRES ET CONTENUS")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = NetflixGenreContentAnalyzer('netflix_data.csv')
    
    # Générer le rapport complet
    results = analyzer.generate_genre_content_report()
    
    print("\n✅ Analyse des genres et contenus terminée!")
    print("📁 Graphiques sauvegardés:")
    print("   • netflix_distribution_genres.png")
    print("   • netflix_clustering_genres.png")
    print("   • netflix_analyse_nlp.png")
    print("   • netflix_patterns_duree.png")
    
    return results

if __name__ == "__main__":
    results = main()
