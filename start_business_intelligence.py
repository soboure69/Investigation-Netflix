#!/usr/bin/env python3
"""
Netflix Data Analysis - Business Intelligence Launcher
=====================================================

Script de lancement pour les analyses Business Intelligence Netflix.
Vérifie les dépendances, exécute les analyses BI et génère les rapports exécutifs.

Auteur: Bello Soboure
Date: 2025-08-01
"""

import os
import sys
import subprocess
import importlib
from datetime import datetime

def check_dependencies():
    """Vérifie les dépendances nécessaires pour les analyses BI."""
    print("🔍 Vérification des dépendances BI...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sqlite3', 'json', 'warnings'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MANQUANT")
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants : {', '.join(missing_packages)}")
        print("Installez-les avec : pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Toutes les dépendances BI sont disponibles")
    return True

def check_data_files():
    """Vérifie la présence des fichiers de données nécessaires."""
    print("\n📁 Vérification des fichiers de données...")
    
    required_files = [
        'netflix_data.csv',
        'netflix_database.db'  # Optionnel mais recommandé
    ]
    
    files_status = {}
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            files_status[file] = f"✅ Présent ({size:.1f} MB)"
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            files_status[file] = "❌ Manquant"
            print(f"  ❌ {file} - MANQUANT")
    
    # Au moins le CSV doit être présent
    if not os.path.exists('netflix_data.csv'):
        print("\n❌ Le fichier netflix_data.csv est requis pour les analyses BI")
        return False
    
    print("✅ Fichiers de données vérifiés")
    return True

def run_bi_analysis():
    """Exécute l'analyse Business Intelligence."""
    print("\n🚀 Lancement de l'analyse Business Intelligence...")
    
    try:
        # Import et exécution du module BI
        from business_intelligence import NetflixBusinessIntelligence
        
        print("📊 Initialisation de l'analyseur BI...")
        analyzer = NetflixBusinessIntelligence()
        
        print("🔄 Exécution de l'analyse BI complète...")
        analyzer.run_complete_bi_analysis()
        
        print("✅ Analyse BI terminée avec succès !")
        return True
        
    except ImportError:
        print("❌ Module business_intelligence non trouvé")
        print("Assurez-vous que 09_business_intelligence.py est présent")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse BI : {e}")
        return False

def run_bi_evaluation():
    """Exécute l'évaluation des analyses BI."""
    print("\n📋 Lancement de l'évaluation BI...")
    
    try:
        # Import et exécution du module d'évaluation BI
        from bi_evaluation import NetflixBIEvaluator
        
        print("🔍 Initialisation de l'évaluateur BI...")
        evaluator = NetflixBIEvaluator()
        
        print("🔄 Exécution de l'évaluation BI complète...")
        evaluator.run_complete_evaluation()
        
        print("✅ Évaluation BI terminée avec succès !")
        return True
        
    except ImportError:
        print("❌ Module bi_evaluation non trouvé")
        print("Assurez-vous que 10_bi_evaluation.py est présent")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation BI : {e}")
        return False

def create_output_directories():
    """Crée les dossiers de sortie nécessaires."""
    print("\n📁 Création des dossiers de sortie...")
    
    directories = [
        'bi_output',
        'Graphiques générés'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Dossier créé : {directory}")
        else:
            print(f"  ✅ Dossier existant : {directory}")

def display_results_summary():
    """Affiche un résumé des résultats générés."""
    print("\n📊 RÉSUMÉ DES RÉSULTATS BI :")
    print("=" * 50)
    
    # Vérification des fichiers de sortie
    bi_files = [
        'bi_output/business_intelligence_report.json',
        'bi_output/executive_summary_report.json'
    ]
    
    graph_files = [
        'Graphiques générés/netflix_roi_analysis.png',
        'Graphiques générés/netflix_executive_dashboard.png',
        'Graphiques générés/netflix_final_executive_dashboard.png'
    ]
    
    print("\n📋 Rapports BI générés :")
    for file in bi_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} - Non généré")
    
    print("\n📊 Graphiques BI générés :")
    for file in graph_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} - Non généré")
    
    # Instructions pour visualiser les résultats
    print("\n💡 INSTRUCTIONS :")
    print("  📋 Consultez les rapports JSON dans bi_output/")
    print("  📊 Visualisez les graphiques dans Graphiques générés/")
    print("  🎯 Le rapport exécutif contient les insights stratégiques")

def main():
    """Fonction principale du lanceur BI."""
    print("🎯 NETFLIX BUSINESS INTELLIGENCE LAUNCHER")
    print("=" * 50)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("👤 Auteur : Bello Soboure")
    print("🎯 Objectif : Analyses stratégiques et insights business")
    
    # Étapes de vérification et d'exécution
    steps = [
        ("Vérification des dépendances", check_dependencies),
        ("Vérification des données", check_data_files),
        ("Création des dossiers", lambda: create_output_directories() or True),
        ("Analyse Business Intelligence", run_bi_analysis),
        ("Évaluation BI", run_bi_evaluation),
        ("Résumé des résultats", lambda: display_results_summary() or True)
    ]
    
    print(f"\n🚀 Exécution en {len(steps)} étapes :")
    
    for i, (step_name, step_function) in enumerate(steps, 1):
        print(f"\n📍 Étape {i}/{len(steps)} : {step_name}")
        print("-" * 40)
        
        try:
            success = step_function()
            if not success:
                print(f"❌ Échec à l'étape {i} : {step_name}")
                print("🛑 Arrêt du processus")
                return False
        except Exception as e:
            print(f"❌ Erreur à l'étape {i} : {e}")
            print("🛑 Arrêt du processus")
            return False
    
    print("\n🎉 BUSINESS INTELLIGENCE COMPLÉTÉ AVEC SUCCÈS !")
    print("=" * 50)
    print("📊 Toutes les analyses BI ont été exécutées")
    print("📋 Rapports exécutifs générés")
    print("🎯 Insights stratégiques disponibles")
    print("💡 Consultez bi_output/ pour les résultats détaillés")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Processus interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        sys.exit(1)
