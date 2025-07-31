# Investigation-Netflix
Explorer les données des films Netflix et effectuer une analyse exploratoire des données pour une société de production afin de découvrir des informations sur les films d'une décennie particulière.


**Netflix** ! Ce qui a commencé en 1997 comme un service de location de DVD est devenu aujourd’hui l’une des plus grandes entreprises du divertissement et des médias.

Étant donné le grand nombre de films et de séries disponibles sur la plateforme, c’est l’occasion idéale de mettre en pratique tes compétences en analyse exploratoire de données et de plonger dans l’univers du divertissement.

Je suis amener à travailler pour une société de production spécialisée dans les styles nostalgiques. Je souhaite mener des recherches sur les films sortis dans les années 1990. Pour cela, je fais une exploration les données Netflix pour mieux comprendre cette formidable décennie cinématographique !

Le dataset `netflix_data.csv` a été fourni, accompagné du tableau ci-dessous qui détaille les noms et descriptions des colonnes.

## Les données  

**netflix_data.csv**

| Column | Description |
|--------|-------------|
| `show_id` | L’ID du programme |
| `type` | Type de programme |
| `title` | Titre du programme |
| `director` | Réalisateur du programme |
| `cast` | Distribution du programme |
| `country` | Pays d’origine |
| `date_added` | Date d’ajout sur Netflix |
| `release_year` | Année de sortie sur Netflix |
| `duration` | Durée du programme en minutes |
| `description` | Description du programme |
| `genre` | Genre du programme |

---

<center><img src="redpopcorn.jpg"></center>

 **Problématique : Effectuer une analyse exploratoire des données ``netflix_data.csv`` pour mieux comprendre les films de la décennie 1990**.

- Quelle était la durée la plus fréquente des films dans les années 1990 ? Enregistrer une réponse approximative sous forme d'entier duration(utilisez 1990 comme année de début de la décennie).

- Un film est considéré comme court s'il dure moins de 90 minutes. Comptez le nombre de courts métrages d'action sortis dans les années 1990 et enregistrez cet entier sous la forme short_movie_count.

## 🚀 Analyses Avancées Réalisées

Ce projet a été étendu avec des analyses approfondies du dataset Netflix :

### 📊 **1. Analyse Temporelle Complète** (`01_temporal_analysis.py`)
- **Évolution du contenu** : Tendances d'ajout de contenu par année/mois
- **Patterns saisonniers** : Identification des périodes d'ajout optimales
- **Comparaison décennies** : Analyse des caractéristiques par époque
- **Croissance cumulative** : Évolution du catalogue Netflix

**Graphiques générés :**
- `netflix_evolution_temporelle.png`
- `netflix_patterns_saisonniers.png`
- `netflix_comparaison_decennies.png`

### 🌍 **2. Analyse Géographique** (`02_geographic_analysis.py`)
- **Distribution mondiale** : Cartographie des productions par pays
- **Marchés dominants** : Identification des pays producteurs leaders
- **Diversité culturelle** : Mesure de la représentation internationale
- **Collaborations** : Analyse des co-productions internationales

**Graphiques générés :**
- `netflix_distribution_geographique.png`
- `netflix_genres_par_region.png`
- `netflix_diversite_culturelle.png`
- `netflix_insights_geographiques.png`

### 🎭 **3. Analyse des Genres et Contenus** (`03_genre_content_analysis.py`)
- **Distribution des genres** : Popularité et évolution temporelle
- **Clustering intelligent** : Regroupement des genres similaires (ML)
- **Analyse NLP** : Traitement des descriptions et sentiment
- **Patterns de durée** : Optimisation par genre et époque

**Graphiques générés :**
- `netflix_distribution_genres.png`
- `netflix_clustering_genres.png`
- `netflix_analyse_nlp.png`
- `netflix_patterns_duree.png`

### 🔧 **4. Pipeline ETL Automatisé** (`04_data_pipeline.py`)
- **Extract** : Chargement et validation des données source
- **Transform** : Nettoyage, standardisation et enrichissement
- **Load** : Structuration en base SQLite relationnelle
- **Quality Control** : Score de qualité et rapports détaillés
- **Monitoring** : Logging complet et métriques de performance

**Fichiers générés :**
- `netflix_database.db` - Base de données SQLite
- `netflix_pipeline.log` - Logs d'exécution
- `pipeline_output/data_quality_report.json` - Rapport qualité

### 5. **API REST** (`05_netflix_api.py`)
- **Endpoints RESTful** : Accès programmatique aux données
- **Documentation Swagger** : Interface interactive à `/docs/`
- **Rate Limiting** : Protection contre les abus
- **Cache intelligent** : Optimisation des performances
- **Recherche avancée** : Filtres multiples et flexibles
- **Analytics** : Statistiques et métriques en temps réel

### 6. **Dashboard Interactif** (`06_netflix_dashboard.py`)
- **Interface web moderne** : Streamlit avec design responsive
- **Visualisations dynamiques** : Plotly pour graphiques interactifs
- **Filtrage en temps réel** : Exploration intuitive des données
- **Analyses multi-dimensionnelles** : Temporelles, géographiques, genres
- **Explorateur de contenu** : Recherche et navigation avancées
- **Insights automatiques** : Recommandations basées sur les données

---

## 🛠️ Structure du Projet

```
Investigation-Netflix/
├── README.md                           # Documentation principale
├── netflix_data.csv                    # Dataset original
├── notebook.ipynb                      # Analyse initiale (années 1990)
├── plan-perspectives.md                # Plan d'analyses avancées
├── requirements.txt                    # Dépendances Python
│
├── 01_temporal_analysis.py             # Analyse temporelle complète
├── 02_geographic_analysis.py           # Analyse géographique
├── 03_genre_content_analysis.py        # Analyse genres et contenus
├── 04_data_pipeline.py                 # Pipeline ETL automatisé
├── 05_netflix_api.py                   # API REST complète
├── 06_netflix_dashboard.py             # Dashboard interactif Streamlit
├── dashboard_config.py                 # Configuration du dashboard
│
├── netflix_database.db                 # Base de données SQLite
├── netflix_pipeline.log                # Logs du pipeline
├── API_DOCUMENTATION.md                # Documentation API
│
├── pipeline_output/
│   └── data_quality_report.json        # Rapport de qualité
│
└── Graphiques générés/
    ├── netflix_evolution_temporelle.png
    ├── netflix_patterns_saisonniers.png
    ├── netflix_comparaison_decennies.png
    ├── netflix_distribution_geographique.png
    ├── netflix_genres_par_region.png
    ├── netflix_diversite_culturelle.png
    ├── netflix_insights_geographiques.png
    ├── netflix_distribution_genres.png
    ├── netflix_clustering_genres.png
    ├── netflix_analyse_nlp.png
    └── netflix_patterns_duree.png
```

---

## 📋 Prérequis et Installation

### Dépendances de base :
```bash
pip install pandas matplotlib seaborn numpy
```

### Dépendances avancées (optionnelles) :
```bash
# Pour le clustering et l'analyse ML
pip install scikit-learn

# Pour les nuages de mots
pip install wordcloud
```

### Exécution des analyses :
```bash
# Analyse temporelle
python 01_temporal_analysis.py

# Analyse géographique
python 02_geographic_analysis.py

# Analyse des genres et contenus
python 03_genre_content_analysis.py

# Pipeline ETL (requis pour l'API)
python 04_data_pipeline.py

# API REST (après le pipeline)
python 05_netflix_api.py
```

### Utilisation de l'API :
```bash
# Démarrer l'API
python 05_netflix_api.py

# Exemples de requêtes
curl "http://localhost:5000/api/content/movies"
curl "http://localhost:5000/api/search/by-genre/Dramas"
curl "http://localhost:5000/api/analytics/stats"
```

**Interface Web :**
- 🏠 Accueil : `http://localhost:5000/`
- 📚 Documentation : `http://localhost:5000/docs/`

### Utilisation du Dashboard :
```bash
# Démarrer le dashboard interactif
streamlit run 06_netflix_dashboard.py

# Ouvrir dans le navigateur
# http://localhost:8501
```

**Fonctionnalités Dashboard :**
- 📊 **Vue d'ensemble** : Métriques clés et distributions
- 📈 **Analyse temporelle** : Évolution et patterns saisonniers
- 🌍 **Analyse géographique** : Cartes et distributions par pays
- 🎭 **Analyse des genres** : Popularité et évolution
- 🔍 **Explorateur** : Recherche et filtrage avancés
- 💡 **Insights** : Recommandations automatiques

---

## 🎯 Résultats Clés

### **Insights Temporels :**
- Croissance exponentielle du catalogue depuis 2015
- Pics saisonniers en début et fin d'année
- Stabilité des durées moyennes entre décennies

### **Insights Géographiques :**
- Domination des productions américaines (>40%)
- Émergence des marchés asiatiques (Corée, Inde)
- Collaborations internationales en hausse

### **Insights Genres :**
- Diversification croissante des genres
- Clustering révèle 8 groupes de genres similaires
- Descriptions optimisées pour l'engagement

---

## 🔮 Prochaines Étapes

Le projet peut être étendu avec :

### 🤖 **Machine Learning & IA**
- Modèles prédictifs de succès
- Système de recommandation avancé
- Classification automatique des genres

### 📈 **Business Intelligence**
- Analyse concurrentielle
- Optimisation ROI par genre
- Stratégie de contenu data-driven

---

## 👨‍💻 Auteur

**Bello Soboure** - Data Scientist/Analyst  
Projet d'analyse approfondie du catalogue Netflix

---
