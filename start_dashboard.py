#!/usr/bin/env python3
"""
Lanceur du Dashboard Netflix
===========================

Script de démarrage rapide pour le dashboard interactif Netflix.
Vérifie les dépendances et lance l'interface Streamlit.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Vérifie et installe les dépendances nécessaires."""
    required_packages = [
        'streamlit>=1.28.0',
        'plotly>=5.17.0',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package_name} - Manquant")
    
    if missing_packages:
        print(f"\n📦 Installation des dépendances manquantes...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installé avec succès")
            except subprocess.CalledProcessError:
                print(f"❌ Erreur lors de l'installation de {package}")
                return False
    
    return True

def check_data_files():
    """Vérifie la présence des fichiers de données."""
    required_files = ['netflix_data.csv']
    optional_files = ['netflix_database.db']
    
    all_present = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Trouvé")
        else:
            print(f"❌ {file} - Manquant (requis)")
            all_present = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file} - Trouvé (données enrichies disponibles)")
        else:
            print(f"⚠️  {file} - Manquant (optionnel - exécutez 04_data_pipeline.py pour l'enrichissement)")
    
    return all_present

def launch_dashboard():
    """Lance le dashboard Streamlit."""
    print("\n🚀 Lancement du dashboard Netflix...")
    print("📊 Interface disponible sur : http://localhost:8501")
    print("🔄 Appuyez sur Ctrl+C pour arrêter le dashboard")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            '06_netflix_dashboard.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n✋ Dashboard arrêté par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors du lancement : {e}")

def main():
    """Fonction principale."""
    print("🎬 Netflix Data Dashboard - Lanceur")
    print("=" * 40)
    
    # Vérification des dépendances
    print("\n📋 Vérification des dépendances...")
    if not check_dependencies():
        print("\n❌ Impossible d'installer toutes les dépendances")
        print("💡 Essayez : pip install -r requirements.txt")
        return
    
    # Vérification des fichiers de données
    print("\n📁 Vérification des fichiers de données...")
    if not check_data_files():
        print("\n❌ Fichiers de données manquants")
        print("💡 Assurez-vous que netflix_data.csv est présent")
        return
    
    # Lancement du dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
