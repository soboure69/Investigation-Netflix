#!/usr/bin/env python3
"""
Netflix Data Analysis - Interactive Dashboard
============================================

Dashboard interactif complet pour l'exploration et la visualisation 
des données Netflix avec Streamlit.

Fonctionnalités :
- Vue d'ensemble des données avec métriques clés
- Analyses temporelles interactives
- Exploration géographique avec cartes
- Analyse des genres et contenus
- Recherche et filtrage avancés
- Visualisations dynamiques et exportables

Auteur: Assistant IA
Date: 2025-07-31
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import warnings
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple

# Configuration de la page
st.set_page_config(
    page_title="Netflix Data Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppression des warnings
warnings.filterwarnings('ignore')

class NetflixDashboard:
    """Dashboard interactif pour l'analyse des données Netflix."""
    
    def __init__(self):
        """Initialise le dashboard."""
        self.db_path = "netflix_database.db"
        self.csv_path = "netflix_data.csv"
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Charge les données depuis la base de données ou le CSV."""
        try:
            # Essayer de charger depuis la base de données
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
                
                # Renommer les colonnes pour correspondre au format original
                column_mapping = {
                    'show_id': 'show_id',
                    'content_type': 'type',
                    'title': 'title',
                    'director': 'director',
                    'cast': 'cast',
                    'countries': 'country',
                    'date_added': 'date_added',
                    'release_year': 'release_year',
                    'rating': 'rating',
                    'duration': 'duration',
                    'genres': 'listed_in',
                    'description': 'description'
                }
                
                self.df = self.df.rename(columns=column_mapping)
                
            else:
                # Charger depuis le CSV
                self.df = pd.read_csv(self.csv_path)
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            # Fallback vers le CSV
            try:
                self.df = pd.read_csv(self.csv_path)
            except Exception as csv_error:
                st.error(f"Impossible de charger les données : {csv_error}")
                return
        
        # Préparation des données
        self.prepare_data()
    
    def prepare_data(self):
        """Prépare les données pour l'analyse."""
        if self.df is None:
            return
            
        # Nettoyage des données
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        self.df['release_year'] = pd.to_numeric(self.df['release_year'], errors='coerce')
        
        # Extraction de features temporelles
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        self.df['quarter_added'] = self.df['date_added'].dt.quarter
        self.df['decade'] = (self.df['release_year'] // 10) * 10
        self.df['decade_label'] = self.df['decade'].astype(str) + 's'
        
        # Nettoyage des durées
        self.df['duration_numeric'] = self.df['duration'].str.extract('(\d+)').astype(float)
        
        # Préparation des genres et pays
        self.df['listed_in'] = self.df['listed_in'].fillna('Unknown')
        self.df['country'] = self.df['country'].fillna('Unknown')
        
        # Dataset propre sans valeurs manquantes critiques
        self.df_clean = self.df.dropna(subset=['date_added', 'release_year'])
    
    def create_sidebar(self):
        """Crée la barre latérale avec les filtres."""
        st.sidebar.title("🎬 Netflix Dashboard")
        st.sidebar.markdown("---")
        
        # Filtres
        st.sidebar.subheader("📊 Filtres")
        
        # Filtre par type
        content_types = ['Tous'] + sorted(self.df['type'].unique().tolist())
        selected_type = st.sidebar.selectbox("Type de contenu", content_types)
        
        # Filtre par période
        if not self.df_clean.empty:
            min_year = int(self.df_clean['year_added'].min())
            max_year = int(self.df_clean['year_added'].max())
            year_range = st.sidebar.slider(
                "Période d'ajout", 
                min_year, max_year, 
                (min_year, max_year)
            )
        else:
            year_range = (2008, 2021)
        
        # Filtre par pays (top 10)
        countries = self.get_top_countries(10)
        selected_countries = st.sidebar.multiselect(
            "Pays", 
            ['Tous'] + countries,
            default=['Tous']
        )
        
        # Filtre par genre (top 10)
        genres = self.get_top_genres(10)
        selected_genres = st.sidebar.multiselect(
            "Genres",
            ['Tous'] + genres,
            default=['Tous']
        )
        
        # Application des filtres
        filtered_df = self.apply_filters(
            selected_type, year_range, selected_countries, selected_genres
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(f"📈 **{len(filtered_df)}** contenus sélectionnés")
        
        return filtered_df
    
    def get_top_countries(self, n: int) -> List[str]:
        """Retourne les top N pays."""
        countries = []
        for country_list in self.df['country'].dropna():
            if country_list != 'Unknown':
                countries.extend([c.strip() for c in str(country_list).split(',')])
        
        country_counts = pd.Series(countries).value_counts()
        return country_counts.head(n).index.tolist()
    
    def get_top_genres(self, n: int) -> List[str]:
        """Retourne les top N genres."""
        genres = []
        for genre_list in self.df['listed_in'].dropna():
            if genre_list != 'Unknown':
                genres.extend([g.strip() for g in str(genre_list).split(',')])
        
        genre_counts = pd.Series(genres).value_counts()
        return genre_counts.head(n).index.tolist()
    
    def apply_filters(self, content_type: str, year_range: Tuple[int, int], 
                     countries: List[str], genres: List[str]) -> pd.DataFrame:
        """Applique les filtres sélectionnés."""
        filtered_df = self.df_clean.copy()
        
        # Filtre par type
        if content_type != 'Tous':
            filtered_df = filtered_df[filtered_df['type'] == content_type]
        
        # Filtre par année
        filtered_df = filtered_df[
            (filtered_df['year_added'] >= year_range[0]) & 
            (filtered_df['year_added'] <= year_range[1])
        ]
        
        # Filtre par pays
        if 'Tous' not in countries and countries:
            country_mask = filtered_df['country'].str.contains(
                '|'.join(countries), case=False, na=False
            )
            filtered_df = filtered_df[country_mask]
        
        # Filtre par genre
        if 'Tous' not in genres and genres:
            genre_mask = filtered_df['listed_in'].str.contains(
                '|'.join(genres), case=False, na=False
            )
            filtered_df = filtered_df[genre_mask]
        
        return filtered_df
    
    def create_overview_metrics(self, df: pd.DataFrame):
        """Crée les métriques de vue d'ensemble."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📺 Total Contenus", 
                f"{len(df):,}",
                delta=f"{len(df) - len(self.df_clean):+,}" if len(df) != len(self.df_clean) else None
            )
        
        with col2:
            movies_count = len(df[df['type'] == 'Movie'])
            st.metric("🎬 Films", f"{movies_count:,}")
        
        with col3:
            shows_count = len(df[df['type'] == 'TV Show'])
            st.metric("📺 Séries TV", f"{shows_count:,}")
        
        with col4:
            if not df.empty:
                avg_year = df['release_year'].mean()
                st.metric("📅 Année Moyenne", f"{avg_year:.0f}")
    
    def create_temporal_analysis(self, df: pd.DataFrame):
        """Crée les analyses temporelles."""
        st.subheader("📈 Analyse Temporelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution par année
            yearly_data = df.groupby('year_added').size().reset_index()
            yearly_data.columns = ['Année', 'Nombre de contenus']
            
            fig_yearly = px.line(
                yearly_data, 
                x='Année', 
                y='Nombre de contenus',
                title="Évolution du nombre de contenus ajoutés par année",
                markers=True
            )
            fig_yearly.update_layout(height=400)
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            # Distribution par mois
            monthly_data = df.groupby('month_added').size().reset_index()
            monthly_data.columns = ['Mois', 'Nombre de contenus']
            monthly_data['Mois_nom'] = monthly_data['Mois'].map({
                1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Aoû', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'
            })
            
            fig_monthly = px.bar(
                monthly_data,
                x='Mois_nom',
                y='Nombre de contenus',
                title="Distribution saisonnière des ajouts",
                color='Nombre de contenus',
                color_continuous_scale='viridis'
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Heatmap par année et mois
        if len(df) > 0:
            heatmap_data = df.groupby(['year_added', 'month_added']).size().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=[f"{i:02d}" for i in range(1, 13)],
                y=heatmap_data.index.astype(str),
                title="Heatmap des ajouts par année et mois",
                labels={'x': 'Mois', 'y': 'Année', 'color': 'Nombre de contenus'},
                color_continuous_scale='viridis'
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def create_geographic_analysis(self, df: pd.DataFrame):
        """Crée les analyses géographiques."""
        st.subheader("🌍 Analyse Géographique")
        
        # Préparation des données pays
        countries_data = []
        for _, row in df.iterrows():
            if pd.notna(row['country']) and row['country'] != 'Unknown':
                countries = [c.strip() for c in str(row['country']).split(',')]
                for country in countries:
                    countries_data.append({
                        'country': country,
                        'type': row['type'],
                        'release_year': row['release_year']
                    })
        
        countries_df = pd.DataFrame(countries_data)
        
        if not countries_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top pays
                top_countries = countries_df['country'].value_counts().head(15)
                
                fig_countries = px.bar(
                    x=top_countries.values,
                    y=top_countries.index,
                    orientation='h',
                    title="Top 15 des pays producteurs",
                    labels={'x': 'Nombre de contenus', 'y': 'Pays'},
                    color=top_countries.values,
                    color_continuous_scale='viridis'
                )
                fig_countries.update_layout(height=500)
                st.plotly_chart(fig_countries, use_container_width=True)
            
            with col2:
                # Distribution par type et pays (top 10)
                top_10_countries = countries_df['country'].value_counts().head(10).index
                filtered_countries = countries_df[countries_df['country'].isin(top_10_countries)]
                
                type_country = filtered_countries.groupby(['country', 'type']).size().unstack(fill_value=0)
                
                fig_type_country = px.bar(
                    type_country,
                    title="Distribution Films vs Séries par pays (Top 10)",
                    labels={'value': 'Nombre de contenus', 'index': 'Pays'},
                    color_discrete_map={'Movie': '#FF6B6B', 'TV Show': '#4ECDC4'}
                )
                fig_type_country.update_layout(height=500)
                st.plotly_chart(fig_type_country, use_container_width=True)
    
    def create_genre_analysis(self, df: pd.DataFrame):
        """Crée les analyses de genres."""
        st.subheader("🎭 Analyse des Genres")
        
        # Préparation des données genres
        genres_data = []
        for _, row in df.iterrows():
            if pd.notna(row['listed_in']) and row['listed_in'] != 'Unknown':
                genres = [g.strip() for g in str(row['listed_in']).split(',')]
                for genre in genres:
                    genres_data.append({
                        'genre': genre,
                        'type': row['type'],
                        'release_year': row['release_year'],
                        'duration_numeric': row['duration_numeric']
                    })
        
        genres_df = pd.DataFrame(genres_data)
        
        if not genres_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top genres
                top_genres = genres_df['genre'].value_counts().head(15)
                
                fig_genres = px.pie(
                    values=top_genres.values,
                    names=top_genres.index,
                    title="Distribution des genres (Top 15)"
                )
                fig_genres.update_traces(textposition='inside', textinfo='percent+label')
                fig_genres.update_layout(height=500)
                st.plotly_chart(fig_genres, use_container_width=True)
            
            with col2:
                # Évolution des genres dans le temps
                genre_evolution = genres_df.groupby(['release_year', 'genre']).size().unstack(fill_value=0)
                top_5_genres = genres_df['genre'].value_counts().head(5).index
                
                fig_evolution = go.Figure()
                for genre in top_5_genres:
                    if genre in genre_evolution.columns:
                        fig_evolution.add_trace(go.Scatter(
                            x=genre_evolution.index,
                            y=genre_evolution[genre],
                            mode='lines+markers',
                            name=genre,
                            line=dict(width=2)
                        ))
                
                fig_evolution.update_layout(
                    title="Évolution des top 5 genres dans le temps",
                    xaxis_title="Année de sortie",
                    yaxis_title="Nombre de contenus",
                    height=500
                )
                st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Analyse des durées par genre
            if 'duration_numeric' in genres_df.columns:
                duration_by_genre = genres_df.groupby('genre')['duration_numeric'].agg(['mean', 'count']).reset_index()
                duration_by_genre = duration_by_genre[duration_by_genre['count'] >= 5]  # Au moins 5 contenus
                duration_by_genre = duration_by_genre.sort_values('mean', ascending=False).head(15)
                
                fig_duration = px.bar(
                    duration_by_genre,
                    x='genre',
                    y='mean',
                    title="Durée moyenne par genre (genres avec 5+ contenus)",
                    labels={'mean': 'Durée moyenne', 'genre': 'Genre'},
                    color='mean',
                    color_continuous_scale='viridis'
                )
                fig_duration.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_duration, use_container_width=True)
    
    def create_content_explorer(self, df: pd.DataFrame):
        """Crée l'explorateur de contenu."""
        st.subheader("🔍 Explorateur de Contenu")
        
        # Barre de recherche
        search_term = st.text_input("🔎 Rechercher dans les titres, descriptions, acteurs...")
        
        # Filtres supplémentaires
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rating_filter = st.selectbox(
                "Classification",
                ['Toutes'] + sorted(df['rating'].dropna().unique().tolist())
            )
        
        with col2:
            if not df.empty and 'duration_numeric' in df.columns:
                duration_range = st.slider(
                    "Durée (minutes/saisons)",
                    int(df['duration_numeric'].min()),
                    int(df['duration_numeric'].max()),
                    (int(df['duration_numeric'].min()), int(df['duration_numeric'].max()))
                )
            else:
                duration_range = (0, 300)
        
        with col3:
            sort_by = st.selectbox(
                "Trier par",
                ['Titre', 'Année de sortie', 'Année d\'ajout', 'Durée']
            )
        
        # Application des filtres
        filtered_content = df.copy()
        
        if search_term:
            search_mask = (
                filtered_content['title'].str.contains(search_term, case=False, na=False) |
                filtered_content['description'].str.contains(search_term, case=False, na=False) |
                filtered_content['cast'].str.contains(search_term, case=False, na=False) |
                filtered_content['director'].str.contains(search_term, case=False, na=False)
            )
            filtered_content = filtered_content[search_mask]
        
        if rating_filter != 'Toutes':
            filtered_content = filtered_content[filtered_content['rating'] == rating_filter]
        
        if 'duration_numeric' in filtered_content.columns:
            filtered_content = filtered_content[
                (filtered_content['duration_numeric'] >= duration_range[0]) &
                (filtered_content['duration_numeric'] <= duration_range[1])
            ]
        
        # Tri
        sort_mapping = {
            'Titre': 'title',
            'Année de sortie': 'release_year',
            'Année d\'ajout': 'year_added',
            'Durée': 'duration_numeric'
        }
        
        if sort_mapping[sort_by] in filtered_content.columns:
            filtered_content = filtered_content.sort_values(sort_mapping[sort_by], ascending=False)
        
        # Affichage des résultats
        st.write(f"**{len(filtered_content)}** contenus trouvés")
        
        # Tableau interactif
        if not filtered_content.empty:
            display_columns = ['title', 'type', 'release_year', 'rating', 'duration', 'listed_in', 'country']
            display_df = filtered_content[display_columns].head(100)  # Limiter à 100 résultats
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    'title': 'Titre',
                    'type': 'Type',
                    'release_year': 'Année',
                    'rating': 'Classification',
                    'duration': 'Durée',
                    'listed_in': 'Genres',
                    'country': 'Pays'
                }
            )
    
    def create_insights_summary(self, df: pd.DataFrame):
        """Crée un résumé des insights clés."""
        st.subheader("💡 Insights Clés")
        
        insights = []
        
        if not df.empty:
            # Insight 1: Croissance
            if 'year_added' in df.columns:
                yearly_growth = df.groupby('year_added').size()
                if len(yearly_growth) > 1:
                    recent_years = yearly_growth.tail(3)
                    growth_rate = ((recent_years.iloc[-1] - recent_years.iloc[0]) / recent_years.iloc[0]) * 100
                    insights.append(f"📈 Croissance de {growth_rate:.1f}% sur les 3 dernières années")
            
            # Insight 2: Type dominant
            type_dist = df['type'].value_counts()
            dominant_type = type_dist.index[0]
            percentage = (type_dist.iloc[0] / len(df)) * 100
            insights.append(f"🎬 {dominant_type}s représentent {percentage:.1f}% du contenu")
            
            # Insight 3: Pays principal
            countries_data = []
            for country_list in df['country'].dropna():
                if country_list != 'Unknown':
                    countries_data.extend([c.strip() for c in str(country_list).split(',')])
            
            if countries_data:
                top_country = pd.Series(countries_data).value_counts().index[0]
                country_count = pd.Series(countries_data).value_counts().iloc[0]
                insights.append(f"🌍 {top_country} domine avec {country_count} contenus")
            
            # Insight 4: Genre populaire
            genres_data = []
            for genre_list in df['listed_in'].dropna():
                if genre_list != 'Unknown':
                    genres_data.extend([g.strip() for g in str(genre_list).split(',')])
            
            if genres_data:
                top_genre = pd.Series(genres_data).value_counts().index[0]
                genre_count = pd.Series(genres_data).value_counts().iloc[0]
                insights.append(f"🎭 {top_genre} est le genre le plus populaire ({genre_count} contenus)")
        
        # Affichage des insights
        for insight in insights:
            st.info(insight)
        
        # Recommandations
        st.subheader("🎯 Recommandations")
        recommendations = [
            "📊 Analyser les tendances saisonnières pour optimiser les lancements",
            "🌐 Explorer les marchés émergents pour la diversification géographique",
            "🎬 Équilibrer le portfolio entre films et séries selon les préférences régionales",
            "📈 Investir dans les genres en croissance identifiés dans l'analyse temporelle"
        ]
        
        for rec in recommendations:
            st.success(rec)
    
    def run(self):
        """Lance le dashboard principal."""
        if self.df is None:
            st.error("❌ Impossible de charger les données. Vérifiez que le fichier netflix_data.csv existe.")
            return
        
        # Titre principal
        st.title("🎬 Netflix Data Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar avec filtres
        filtered_df = self.create_sidebar()
        
        # Onglets principaux
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Vue d'ensemble", 
            "📈 Temporel", 
            "🌍 Géographique", 
            "🎭 Genres", 
            "🔍 Explorateur",
            "💡 Insights"
        ])
        
        with tab1:
            st.header("📊 Vue d'ensemble")
            self.create_overview_metrics(filtered_df)
            
            # Graphiques de base
            if not filtered_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution par type
                    type_dist = filtered_df['type'].value_counts()
                    fig_type = px.pie(
                        values=type_dist.values,
                        names=type_dist.index,
                        title="Distribution par type de contenu"
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
                
                with col2:
                    # Distribution par décennie
                    decade_dist = filtered_df['decade_label'].value_counts().sort_index()
                    fig_decade = px.bar(
                        x=decade_dist.index,
                        y=decade_dist.values,
                        title="Distribution par décennie de sortie",
                        labels={'x': 'Décennie', 'y': 'Nombre de contenus'},
                        color=decade_dist.values,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_decade, use_container_width=True)
        
        with tab2:
            self.create_temporal_analysis(filtered_df)
        
        with tab3:
            self.create_geographic_analysis(filtered_df)
        
        with tab4:
            self.create_genre_analysis(filtered_df)
        
        with tab5:
            self.create_content_explorer(filtered_df)
        
        with tab6:
            self.create_insights_summary(filtered_df)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                🎬 Netflix Data Analytics Dashboard | 
                Développé avec Streamlit & Plotly | 
                Données mises à jour automatiquement
            </div>
            """, 
            unsafe_allow_html=True
        )

def main():
    """Fonction principale."""
    try:
        dashboard = NetflixDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du dashboard : {e}")
        st.info("Vérifiez que le fichier netflix_data.csv est présent dans le répertoire.")

if __name__ == "__main__":
    main()
