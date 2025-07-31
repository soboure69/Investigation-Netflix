"""
🌐 NETFLIX DATA ANALYSIS - PARTIE 5: API REST
=============================================

Ce module implémente une API REST complète pour interroger les données Netflix :
- Endpoints RESTful pour toutes les données
- Authentification et rate limiting
- Documentation automatique avec Swagger
- Système de cache pour les performances
- Monitoring et analytics des requêtes

Auteur: Bello Soboure
Date: 2025-01-31
"""

from flask import Flask, jsonify, request, render_template_string
from flask_restx import Api, Resource, fields, Namespace
from flask_cors import CORS
from flask_caching import Cache
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from functools import wraps
import hashlib
import os
from pathlib import Path

# Configuration de l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'netflix-api-secret-key-2025'
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes

# Extensions
CORS(app)
cache = Cache(app)
api = Api(app, 
    title='Netflix Data API',
    version='1.0',
    description='API REST pour l\'analyse des données Netflix',
    doc='/docs/'
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base de données
DATABASE = 'netflix_database.db'

def get_db_connection():
    """Obtient une connexion à la base de données"""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Erreur connexion DB: {e}")
        return None

def execute_query(query, params=None):
    """Exécute une requête SQL et retourne les résultats"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        if params:
            result = pd.read_sql_query(query, conn, params=params)
        else:
            result = pd.read_sql_query(query, conn)
        return result.to_dict('records')
    except Exception as e:
        logger.error(f"Erreur requête: {e}")
        return None
    finally:
        conn.close()

def rate_limit(max_requests=100, window=3600):
    """Décorateur pour limiter le taux de requêtes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Implémentation basique du rate limiting
            client_ip = request.remote_addr
            current_time = datetime.now()
            
            # Pour une implémentation complète, utiliser Redis
            # Ici, on simule avec un cache simple
            cache_key = f"rate_limit_{client_ip}"
            requests_data = cache.get(cache_key) or []
            
            # Nettoyer les anciennes requêtes
            cutoff_time = current_time - timedelta(seconds=window)
            requests_data = [req_time for req_time in requests_data if req_time > cutoff_time]
            
            if len(requests_data) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_requests} requests per hour'
                }), 429
            
            requests_data.append(current_time)
            cache.set(cache_key, requests_data, timeout=window)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Namespaces pour organiser l'API
content_ns = Namespace('content', description='Opérations sur le contenu Netflix')
analytics_ns = Namespace('analytics', description='Analyses et statistiques')
search_ns = Namespace('search', description='Recherche et filtrage')
api.add_namespace(content_ns, path='/api/content')
api.add_namespace(analytics_ns, path='/api/analytics')
api.add_namespace(search_ns, path='/api/search')

# Modèles pour la documentation Swagger
content_model = api.model('Content', {
    'show_id': fields.String(required=True, description='ID unique du contenu'),
    'type': fields.String(required=True, description='Type (Movie/TV Show)'),
    'title': fields.String(required=True, description='Titre'),
    'director': fields.String(description='Réalisateur'),
    'cast': fields.String(description='Distribution'),
    'country': fields.String(description='Pays d\'origine'),
    'date_added': fields.String(description='Date d\'ajout sur Netflix'),
    'release_year': fields.Integer(description='Année de sortie'),
    'duration': fields.Integer(description='Durée en minutes'),
    'description': fields.String(description='Description'),
    'genre': fields.String(description='Genres')
})

# Routes principales
@app.route('/')
def home():
    """Page d'accueil de l'API"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Netflix Data API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #e50914; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .method.post { background: #ffc107; }
            .method.put { background: #17a2b8; }
            .method.delete { background: #dc3545; }
            a { color: #e50914; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Netflix Data API</h1>
            <p>API REST pour l'analyse des données Netflix avec documentation interactive.</p>
            
            <h2>📚 Documentation</h2>
            <p><a href="/docs/">Documentation Swagger Interactive</a></p>
            
            <h2>🔗 Endpoints Principaux</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/content/all</strong>
                <p>Récupère tout le contenu Netflix</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/content/movies</strong>
                <p>Récupère uniquement les films</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/content/shows</strong>
                <p>Récupère uniquement les séries TV</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/search/by-genre/{genre}</strong>
                <p>Recherche par genre</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/analytics/stats</strong>
                <p>Statistiques générales</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/api/analytics/trends</strong>
                <p>Tendances temporelles</p>
            </div>
            
            <h2>📊 Exemples d'utilisation</h2>
            <pre>
# Récupérer tous les films
curl -X GET "http://localhost:5000/api/content/movies"

# Rechercher des drames
curl -X GET "http://localhost:5000/api/search/by-genre/Dramas"

# Obtenir les statistiques
curl -X GET "http://localhost:5000/api/analytics/stats"
            </pre>
            
            <p><em>Développé par Bello Soboure - 2025</em></p>
        </div>
    </body>
    </html>
    ''')

# CONTENT ENDPOINTS
@content_ns.route('/all')
class AllContent(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self):
        """Récupère tout le contenu Netflix"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = min(request.args.get('per_page', 50, type=int), 100)
            offset = (page - 1) * per_page
            
            query = f"""
                SELECT * FROM netflix_content 
                ORDER BY date_added DESC 
                LIMIT {per_page} OFFSET {offset}
            """
            
            results = execute_query(query)
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            # Compter le total pour la pagination
            count_query = "SELECT COUNT(*) as total FROM netflix_content"
            total_results = execute_query(count_query)
            total = total_results[0]['total'] if total_results else 0
            
            return {
                'data': results,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur /content/all: {e}")
            return {'error': 'Internal server error'}, 500

@content_ns.route('/movies')
class Movies(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self):
        """Récupère uniquement les films"""
        try:
            query = "SELECT * FROM netflix_content WHERE type = 'Movie' ORDER BY release_year DESC"
            results = execute_query(query)
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results)}
            
        except Exception as e:
            logger.error(f"Erreur /content/movies: {e}")
            return {'error': 'Internal server error'}, 500

@content_ns.route('/shows')
class TVShows(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self):
        """Récupère uniquement les séries TV"""
        try:
            query = "SELECT * FROM netflix_content WHERE type = 'TV Show' ORDER BY release_year DESC"
            results = execute_query(query)
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results)}
            
        except Exception as e:
            logger.error(f"Erreur /content/shows: {e}")
            return {'error': 'Internal server error'}, 500

@content_ns.route('/<string:show_id>')
class ContentById(Resource):
    @rate_limit(max_requests=100)
    @cache.cached(timeout=600)
    def get(self, show_id):
        """Récupère un contenu spécifique par ID"""
        try:
            query = "SELECT * FROM netflix_content WHERE show_id = ?"
            results = execute_query(query, (show_id,))
            
            if not results:
                return {'error': 'Content not found'}, 404
            
            return {'data': results[0]}
            
        except Exception as e:
            logger.error(f"Erreur /content/{show_id}: {e}")
            return {'error': 'Internal server error'}, 500

# SEARCH ENDPOINTS
@search_ns.route('/by-genre/<string:genre>')
class SearchByGenre(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self, genre):
        """Recherche par genre"""
        try:
            query = "SELECT * FROM netflix_content WHERE genre LIKE ? ORDER BY release_year DESC"
            results = execute_query(query, (f'%{genre}%',))
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results), 'genre': genre}
            
        except Exception as e:
            logger.error(f"Erreur /search/by-genre/{genre}: {e}")
            return {'error': 'Internal server error'}, 500

@search_ns.route('/by-country/<string:country>')
class SearchByCountry(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self, country):
        """Recherche par pays"""
        try:
            query = "SELECT * FROM netflix_content WHERE country LIKE ? ORDER BY release_year DESC"
            results = execute_query(query, (f'%{country}%',))
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results), 'country': country}
            
        except Exception as e:
            logger.error(f"Erreur /search/by-country/{country}: {e}")
            return {'error': 'Internal server error'}, 500

@search_ns.route('/by-year/<int:year>')
class SearchByYear(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self, year):
        """Recherche par année de sortie"""
        try:
            query = "SELECT * FROM netflix_content WHERE release_year = ? ORDER BY title"
            results = execute_query(query, (year,))
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results), 'year': year}
            
        except Exception as e:
            logger.error(f"Erreur /search/by-year/{year}: {e}")
            return {'error': 'Internal server error'}, 500

@search_ns.route('/by-decade/<int:decade>')
class SearchByDecade(Resource):
    @rate_limit(max_requests=50)
    @cache.cached(timeout=300)
    def get(self, decade):
        """Recherche par décennie"""
        try:
            query = "SELECT * FROM netflix_content WHERE decade = ? ORDER BY release_year DESC"
            results = execute_query(query, (decade,))
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {'data': results, 'count': len(results), 'decade': f"{decade}s"}
            
        except Exception as e:
            logger.error(f"Erreur /search/by-decade/{decade}: {e}")
            return {'error': 'Internal server error'}, 500

@search_ns.route('/advanced')
class AdvancedSearch(Resource):
    @rate_limit(max_requests=30)
    def get(self):
        """Recherche avancée avec filtres multiples"""
        try:
            # Paramètres de recherche
            content_type = request.args.get('type')
            genre = request.args.get('genre')
            country = request.args.get('country')
            year_min = request.args.get('year_min', type=int)
            year_max = request.args.get('year_max', type=int)
            duration_min = request.args.get('duration_min', type=int)
            duration_max = request.args.get('duration_max', type=int)
            
            # Construction de la requête dynamique
            conditions = []
            params = []
            
            if content_type:
                conditions.append("type = ?")
                params.append(content_type)
            
            if genre:
                conditions.append("genre LIKE ?")
                params.append(f'%{genre}%')
            
            if country:
                conditions.append("country LIKE ?")
                params.append(f'%{country}%')
            
            if year_min:
                conditions.append("release_year >= ?")
                params.append(year_min)
            
            if year_max:
                conditions.append("release_year <= ?")
                params.append(year_max)
            
            if duration_min:
                conditions.append("duration >= ?")
                params.append(duration_min)
            
            if duration_max:
                conditions.append("duration <= ?")
                params.append(duration_max)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT * FROM netflix_content WHERE {where_clause} ORDER BY release_year DESC"
            
            results = execute_query(query, params)
            
            if results is None:
                return {'error': 'Database error'}, 500
            
            return {
                'data': results,
                'count': len(results),
                'filters': {
                    'type': content_type,
                    'genre': genre,
                    'country': country,
                    'year_range': [year_min, year_max],
                    'duration_range': [duration_min, duration_max]
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur /search/advanced: {e}")
            return {'error': 'Internal server error'}, 500

# ANALYTICS ENDPOINTS
@analytics_ns.route('/stats')
class GeneralStats(Resource):
    @rate_limit(max_requests=30)
    @cache.cached(timeout=600)
    def get(self):
        """Statistiques générales"""
        try:
            stats = {}
            
            # Total contenu
            total_query = "SELECT COUNT(*) as total FROM netflix_content"
            total_result = execute_query(total_query)
            stats['total_content'] = total_result[0]['total'] if total_result else 0
            
            # Par type
            type_query = "SELECT type, COUNT(*) as count FROM netflix_content GROUP BY type"
            type_results = execute_query(type_query)
            stats['by_type'] = {item['type']: item['count'] for item in type_results} if type_results else {}
            
            # Top genres
            genre_query = """
                SELECT genre, COUNT(*) as count 
                FROM netflix_content 
                WHERE genre IS NOT NULL AND genre != 'Unknown'
                GROUP BY genre 
                ORDER BY count DESC 
                LIMIT 10
            """
            genre_results = execute_query(genre_query)
            stats['top_genres'] = genre_results if genre_results else []
            
            # Top pays
            country_query = """
                SELECT country, COUNT(*) as count 
                FROM netflix_content 
                WHERE country IS NOT NULL AND country != 'Unknown'
                GROUP BY country 
                ORDER BY count DESC 
                LIMIT 10
            """
            country_results = execute_query(country_query)
            stats['top_countries'] = country_results if country_results else []
            
            # Par décennie
            decade_query = """
                SELECT decade_label, COUNT(*) as count 
                FROM netflix_content 
                GROUP BY decade_label 
                ORDER BY decade
            """
            decade_results = execute_query(decade_query)
            stats['by_decade'] = decade_results if decade_results else []
            
            return {'data': stats}
            
        except Exception as e:
            logger.error(f"Erreur /analytics/stats: {e}")
            return {'error': 'Internal server error'}, 500

@analytics_ns.route('/trends')
class Trends(Resource):
    @rate_limit(max_requests=20)
    @cache.cached(timeout=600)
    def get(self):
        """Tendances temporelles"""
        try:
            trends = {}
            
            # Contenu ajouté par année
            yearly_query = """
                SELECT year_added, COUNT(*) as count 
                FROM netflix_content 
                WHERE year_added IS NOT NULL
                GROUP BY year_added 
                ORDER BY year_added
            """
            yearly_results = execute_query(yearly_query)
            trends['content_added_by_year'] = yearly_results if yearly_results else []
            
            # Évolution des durées moyennes
            duration_query = """
                SELECT release_year, AVG(duration) as avg_duration 
                FROM netflix_content 
                WHERE type = 'Movie' AND duration > 0
                GROUP BY release_year 
                ORDER BY release_year
            """
            duration_results = execute_query(duration_query)
            trends['average_duration_by_year'] = duration_results if duration_results else []
            
            # Diversité des genres par année
            diversity_query = """
                SELECT release_year, COUNT(DISTINCT genre) as genre_diversity 
                FROM netflix_content 
                WHERE release_year IS NOT NULL
                GROUP BY release_year 
                ORDER BY release_year
            """
            diversity_results = execute_query(diversity_query)
            trends['genre_diversity_by_year'] = diversity_results if diversity_results else []
            
            return {'data': trends}
            
        except Exception as e:
            logger.error(f"Erreur /analytics/trends: {e}")
            return {'error': 'Internal server error'}, 500

@analytics_ns.route('/quality')
class QualityMetrics(Resource):
    @rate_limit(max_requests=10)
    @cache.cached(timeout=900)
    def get(self):
        """Métriques de qualité des données"""
        try:
            quality_query = """
                SELECT * FROM data_quality_metrics 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            quality_results = execute_query(quality_query)
            
            if not quality_results:
                return {'error': 'No quality metrics found'}, 404
            
            return {'data': quality_results[0]}
            
        except Exception as e:
            logger.error(f"Erreur /analytics/quality: {e}")
            return {'error': 'Internal server error'}, 500

# Gestion des erreurs
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

# Middleware pour logging des requêtes
@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")

def create_api_documentation():
    """Crée un fichier de documentation API"""
    doc_content = """
# Netflix Data API Documentation

## Base URL
```
http://localhost:5000
```

## Authentication
Actuellement, l'API ne nécessite pas d'authentification, mais un rate limiting est appliqué.

## Rate Limiting
- 100 requêtes par heure pour les endpoints généraux
- 50 requêtes par heure pour les endpoints de contenu
- 30 requêtes par heure pour les recherches avancées
- 20 requêtes par heure pour les analyses
- 10 requêtes par heure pour les métriques de qualité

## Endpoints

### Content
- `GET /api/content/all` - Tout le contenu (avec pagination)
- `GET /api/content/movies` - Films uniquement
- `GET /api/content/shows` - Séries TV uniquement
- `GET /api/content/{show_id}` - Contenu spécifique

### Search
- `GET /api/search/by-genre/{genre}` - Recherche par genre
- `GET /api/search/by-country/{country}` - Recherche par pays
- `GET /api/search/by-year/{year}` - Recherche par année
- `GET /api/search/by-decade/{decade}` - Recherche par décennie
- `GET /api/search/advanced` - Recherche avancée avec filtres

### Analytics
- `GET /api/analytics/stats` - Statistiques générales
- `GET /api/analytics/trends` - Tendances temporelles
- `GET /api/analytics/quality` - Métriques de qualité

## Response Format
Toutes les réponses sont en JSON avec la structure suivante :
```json
{
    "data": [...],
    "count": 123,
    "pagination": {...}
}
```

## Error Handling
Les erreurs retournent un code HTTP approprié avec un message :
```json
{
    "error": "Description de l'erreur"
}
```
"""
    
    with open('API_DOCUMENTATION.md', 'w', encoding='utf-8') as f:
        f.write(doc_content)

if __name__ == '__main__':
    # Vérifier que la base de données existe
    if not os.path.exists(DATABASE):
        print("❌ Base de données non trouvée. Exécutez d'abord le pipeline ETL (04_data_pipeline.py)")
        exit(1)
    
    # Créer la documentation
    create_api_documentation()
    
    print("🌐 NETFLIX DATA API")
    print("="*50)
    print("🚀 Démarrage du serveur API...")
    print("📚 Documentation: http://localhost:5000/docs/")
    print("🏠 Page d'accueil: http://localhost:5000/")
    print("📖 Documentation complète: API_DOCUMENTATION.md")
    
    # Démarrer le serveur
    app.run(debug=True, host='0.0.0.0', port=5000)
