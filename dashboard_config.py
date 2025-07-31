#!/usr/bin/env python3
"""
Configuration du Dashboard Netflix
=================================

Configuration centralisée pour le dashboard interactif Netflix.
"""

import os

# Configuration de base
DASHBOARD_CONFIG = {
    'title': 'Netflix Data Analytics Dashboard',
    'icon': '🎬',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Chemins des fichiers
DATA_PATHS = {
    'database': 'netflix_database.db',
    'csv': 'netflix_data.csv',
    'pipeline_output': 'pipeline_output/',
    'graphics': 'Graphiques générés/'
}

# Configuration des graphiques
PLOT_CONFIG = {
    'color_schemes': {
        'primary': 'viridis',
        'secondary': 'plasma',
        'categorical': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    },
    'default_height': 400,
    'large_height': 600
}

# Mapping des pays vers continents
CONTINENT_MAPPING = {
    'United States': 'North America',
    'India': 'Asia',
    'United Kingdom': 'Europe',
    'Canada': 'North America',
    'France': 'Europe',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'Germany': 'Europe',
    'Spain': 'Europe',
    'Italy': 'Europe',
    'Australia': 'Oceania',
    'Brazil': 'South America',
    'Mexico': 'North America',
    'Turkey': 'Asia',
    'Netherlands': 'Europe',
    'Belgium': 'Europe',
    'Argentina': 'South America',
    'Egypt': 'Africa',
    'South Africa': 'Africa',
    'Nigeria': 'Africa',
    'China': 'Asia',
    'Russia': 'Europe',
    'Poland': 'Europe',
    'Sweden': 'Europe',
    'Norway': 'Europe',
    'Denmark': 'Europe',
    'Finland': 'Europe',
    'Ireland': 'Europe',
    'Portugal': 'Europe',
    'Greece': 'Europe',
    'Switzerland': 'Europe',
    'Austria': 'Europe',
    'Czech Republic': 'Europe',
    'Hungary': 'Europe',
    'Romania': 'Europe',
    'Bulgaria': 'Europe',
    'Croatia': 'Europe',
    'Serbia': 'Europe',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Lithuania': 'Europe',
    'Latvia': 'Europe',
    'Estonia': 'Europe'
}

# Configuration des filtres
FILTER_CONFIG = {
    'max_countries': 15,
    'max_genres': 15,
    'max_results_display': 100,
    'min_genre_count': 5  # Minimum de contenus pour afficher un genre dans les stats
}

# Messages et textes
UI_TEXTS = {
    'loading': "⏳ Chargement des données...",
    'error_data': "❌ Impossible de charger les données. Vérifiez que le fichier netflix_data.csv existe.",
    'no_results': "Aucun résultat trouvé avec les filtres actuels.",
    'search_placeholder': "🔎 Rechercher dans les titres, descriptions, acteurs...",
    'insights_title': "💡 Insights Clés",
    'recommendations_title': "🎯 Recommandations"
}

# Configuration des insights automatiques
INSIGHTS_CONFIG = {
    'growth_years': 3,  # Nombre d'années pour calculer la croissance
    'min_content_threshold': 10,  # Minimum de contenus pour générer un insight
    'top_n_display': 5  # Nombre d'éléments à afficher dans les tops
}

def get_data_path(key: str) -> str:
    """Retourne le chemin d'un fichier de données."""
    return DATA_PATHS.get(key, '')

def get_plot_config(key: str):
    """Retourne une configuration de graphique."""
    return PLOT_CONFIG.get(key)

def get_continent(country: str) -> str:
    """Retourne le continent d'un pays."""
    return CONTINENT_MAPPING.get(country, 'Unknown')
