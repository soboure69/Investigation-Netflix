#!/usr/bin/env python3
"""
Netflix Data Analysis - Business Intelligence Evaluation & Testing
================================================================

Module d'évaluation et de tests pour les analyses Business Intelligence.
Tests des KPIs, validation des recommandations et génération de rapports exécutifs.

Auteur: Bello Soboure
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

class NetflixBIEvaluator:
    """Évaluateur des analyses Business Intelligence Netflix."""
    
    def __init__(self):
        """Initialise l'évaluateur BI."""
        self.bi_report = None
        self.evaluation_results = {}
        self.strategic_insights = []
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.load_bi_results()
    
    def load_bi_results(self):
        """Charge les résultats de l'analyse BI."""
        try:
            if os.path.exists('bi_output/business_intelligence_report.json'):
                with open('bi_output/business_intelligence_report.json', 'r', encoding='utf-8') as f:
                    self.bi_report = json.load(f)
                print("✅ Rapport BI chargé pour évaluation")
            else:
                print("⚠️ Aucun rapport BI trouvé. Exécutez d'abord 09_business_intelligence.py")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du rapport BI : {e}")
    
    def evaluate_financial_performance(self):
        """Évalue les performances financières."""
        print("💰 Évaluation des performances financières...")
        
        if not self.bi_report or 'kpis' not in self.bi_report:
            print("❌ Données KPIs non disponibles")
            return
        
        financial_kpis = self.bi_report['kpis']['financial']
        
        # Benchmarks de l'industrie (estimations)
        industry_benchmarks = {
            'average_roi': 2.5,
            'high_roi_content_percentage': 25.0,
            'cost_efficiency_threshold': 1.5
        }
        
        # Évaluation par rapport aux benchmarks
        performance_evaluation = {}
        
        # ROI moyen
        roi_performance = financial_kpis['average_roi'] / industry_benchmarks['average_roi']
        performance_evaluation['roi_vs_industry'] = {
            'value': financial_kpis['average_roi'],
            'benchmark': industry_benchmarks['average_roi'],
            'performance_ratio': roi_performance,
            'status': 'Excellent' if roi_performance > 1.2 else 'Bon' if roi_performance > 0.8 else 'À améliorer'
        }
        
        # Pourcentage de contenu à haut ROI
        high_roi_performance = financial_kpis['high_roi_content_percentage'] / industry_benchmarks['high_roi_content_percentage']
        performance_evaluation['high_roi_content'] = {
            'value': financial_kpis['high_roi_content_percentage'],
            'benchmark': industry_benchmarks['high_roi_content_percentage'],
            'performance_ratio': high_roi_performance,
            'status': 'Excellent' if high_roi_performance > 1.2 else 'Bon' if high_roi_performance > 0.8 else 'À améliorer'
        }
        
        # Score global de performance financière
        overall_score = (roi_performance + high_roi_performance) / 2
        performance_evaluation['overall_financial_score'] = {
            'score': overall_score,
            'grade': 'A' if overall_score > 1.2 else 'B' if overall_score > 1.0 else 'C' if overall_score > 0.8 else 'D'
        }
        
        self.evaluation_results['financial_performance'] = performance_evaluation
        
        # Visualisation de l'évaluation financière
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Comparaison ROI
        categories = ['Netflix', 'Industrie']
        roi_values = [financial_kpis['average_roi'], industry_benchmarks['average_roi']]
        colors = ['#1f77b4', '#ff7f0e']
        
        axes[0].bar(categories, roi_values, color=colors)
        axes[0].set_title('ROI Moyen vs Industrie')
        axes[0].set_ylabel('ROI')
        for i, v in enumerate(roi_values):
            axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
        
        # Performance du contenu à haut ROI
        high_roi_values = [financial_kpis['high_roi_content_percentage'], industry_benchmarks['high_roi_content_percentage']]
        axes[1].bar(categories, high_roi_values, color=colors)
        axes[1].set_title('Contenu à Haut ROI vs Industrie')
        axes[1].set_ylabel('Pourcentage (%)')
        for i, v in enumerate(high_roi_values):
            axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        # Score global
        score_data = [overall_score, 1.0]  # 1.0 = benchmark
        axes[2].bar(['Performance Netflix', 'Benchmark Industrie'], score_data, color=['green' if overall_score > 1 else 'orange', 'gray'])
        axes[2].set_title('Score Global de Performance')
        axes[2].set_ylabel('Score')
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_financial_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Évaluation financière terminée - Score global : {performance_evaluation['overall_financial_score']['grade']}")
    
    def analyze_strategic_positioning(self):
        """Analyse le positionnement stratégique."""
        print("🎯 Analyse du positionnement stratégique...")
        
        if not self.bi_report:
            return
        
        # Analyse de la diversification
        content_kpis = self.bi_report['kpis']['content']
        
        strategic_metrics = {
            'content_diversity': content_kpis['content_diversity_score'],
            'geographic_reach': content_kpis['geographic_reach'],
            'content_freshness': content_kpis['content_freshness_percentage'],
            'total_content': content_kpis['total_content_count']
        }
        
        # Benchmarks stratégiques
        strategic_benchmarks = {
            'content_diversity': 50,  # Nombre de genres
            'geographic_reach': 100,  # Nombre de pays
            'content_freshness': 40,  # % de contenu récent
            'content_volume': 5000   # Nombre total de contenus
        }
        
        # Évaluation stratégique
        strategic_evaluation = {}
        
        for metric, value in strategic_metrics.items():
            if metric == 'total_content':
                benchmark = strategic_benchmarks['content_volume']
            else:
                benchmark = strategic_benchmarks[metric]
            
            performance_ratio = value / benchmark
            strategic_evaluation[metric] = {
                'value': value,
                'benchmark': benchmark,
                'performance_ratio': performance_ratio,
                'status': 'Fort' if performance_ratio > 1.2 else 'Satisfaisant' if performance_ratio > 0.8 else 'Faible'
            }
        
        # Score de positionnement stratégique
        avg_performance = np.mean([eval_data['performance_ratio'] for eval_data in strategic_evaluation.values()])
        strategic_evaluation['strategic_positioning_score'] = {
            'score': avg_performance,
            'level': 'Leader' if avg_performance > 1.2 else 'Concurrent Fort' if avg_performance > 1.0 else 'Suiveur'
        }
        
        self.evaluation_results['strategic_positioning'] = strategic_evaluation
        
        # Visualisation du positionnement stratégique
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Radar chart des métriques stratégiques
        metrics = list(strategic_metrics.keys())
        values = [strategic_evaluation[m]['performance_ratio'] for m in metrics]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Fermer le cercle
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 2, 1, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax_radar.fill(angles, values, alpha=0.25, color='blue')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 2)
        ax_radar.set_title('Positionnement Stratégique\n(1.0 = Benchmark)')
        ax_radar.grid(True)
        
        # Comparaison diversité
        axes[0, 1].bar(['Netflix', 'Benchmark'], 
                      [strategic_metrics['content_diversity'], strategic_benchmarks['content_diversity']], 
                      color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('Diversité du Contenu')
        axes[0, 1].set_ylabel('Nombre de Genres')
        
        # Fraîcheur du contenu
        axes[1, 0].pie([strategic_metrics['content_freshness'], 100 - strategic_metrics['content_freshness']], 
                      labels=['Contenu Récent', 'Contenu Ancien'], 
                      autopct='%1.1f%%', colors=['lightgreen', 'lightgray'])
        axes[1, 0].set_title('Fraîcheur du Contenu')
        
        # Score de positionnement
        positioning_score = strategic_evaluation['strategic_positioning_score']['score']
        axes[1, 1].bar(['Score Netflix'], [positioning_score], color='green' if positioning_score > 1 else 'orange')
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Benchmark')
        axes[1, 1].set_title('Score de Positionnement Stratégique')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 2)
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_strategic_positioning.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Analyse stratégique terminée - Niveau : {strategic_evaluation['strategic_positioning_score']['level']}")
    
    def generate_executive_insights(self):
        """Génère des insights exécutifs avancés."""
        print("💡 Génération d'insights exécutifs avancés...")
        
        insights = []
        
        # Insights basés sur l'évaluation financière
        if 'financial_performance' in self.evaluation_results:
            financial_eval = self.evaluation_results['financial_performance']
            
            if financial_eval['overall_financial_score']['score'] > 1.2:
                insights.append({
                    'category': 'Performance Financière',
                    'type': 'Opportunité',
                    'insight': 'Performance financière excellente - Position de leader sur le marché',
                    'action': 'Maintenir la stratégie actuelle et explorer l\'expansion internationale',
                    'priority': 'Stratégique'
                })
            elif financial_eval['overall_financial_score']['score'] < 0.8:
                insights.append({
                    'category': 'Performance Financière',
                    'type': 'Risque',
                    'insight': 'Performance financière sous les standards de l\'industrie',
                    'action': 'Révision urgente de la stratégie d\'investissement et d\'acquisition',
                    'priority': 'Critique'
                })
        
        # Insights basés sur le positionnement stratégique
        if 'strategic_positioning' in self.evaluation_results:
            strategic_eval = self.evaluation_results['strategic_positioning']
            
            if strategic_eval['content_diversity']['performance_ratio'] < 0.8:
                insights.append({
                    'category': 'Diversification',
                    'type': 'Opportunité',
                    'insight': 'Faible diversité de genres - Opportunité d\'expansion du catalogue',
                    'action': 'Investir dans des genres sous-représentés à fort potentiel',
                    'priority': 'Haute'
                })
            
            if strategic_eval['content_freshness']['performance_ratio'] < 0.7:
                insights.append({
                    'category': 'Innovation',
                    'type': 'Risque',
                    'insight': 'Catalogue vieillissant - Risque de perte d\'audience',
                    'action': 'Accélérer l\'acquisition de contenu récent et les productions originales',
                    'priority': 'Haute'
                })
        
        # Insights basés sur les recommandations BI
        if self.bi_report and 'recommendations' in self.bi_report:
            high_impact_recs = [r for r in self.bi_report['recommendations'] if r.get('impact') == 'High']
            if len(high_impact_recs) > 2:
                insights.append({
                    'category': 'Optimisation',
                    'type': 'Action',
                    'insight': f'{len(high_impact_recs)} recommandations à fort impact identifiées',
                    'action': 'Prioriser l\'implémentation des recommandations à fort impact',
                    'priority': 'Immédiate'
                })
        
        # Insights de croissance
        if self.bi_report and 'executive_summary' in self.bi_report:
            total_investment = self.bi_report['executive_summary']['total_investment_millions']
            if total_investment > 50000:  # Plus de 50B$
                insights.append({
                    'category': 'Croissance',
                    'type': 'Opportunité',
                    'insight': 'Investissement massif dans le contenu - Position dominante',
                    'action': 'Optimiser le ROI et explorer de nouveaux marchés géographiques',
                    'priority': 'Stratégique'
                })
        
        self.strategic_insights = insights
        
        print(f"✅ {len(insights)} insights exécutifs générés")
    
    def create_executive_summary_report(self):
        """Crée un rapport de synthèse exécutif."""
        print("📋 Création du rapport de synthèse exécutif...")
        
        # Compilation du rapport exécutif
        executive_report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {
                'financial_grade': self.evaluation_results.get('financial_performance', {}).get('overall_financial_score', {}).get('grade', 'N/A'),
                'strategic_level': self.evaluation_results.get('strategic_positioning', {}).get('strategic_positioning_score', {}).get('level', 'N/A'),
                'total_insights': len(self.strategic_insights),
                'critical_actions': len([i for i in self.strategic_insights if i.get('priority') == 'Critique'])
            },
            'key_metrics': self.bi_report['executive_summary'] if self.bi_report else {},
            'performance_evaluation': self.evaluation_results,
            'strategic_insights': self.strategic_insights,
            'action_plan': self.create_action_plan()
        }
        
        # Sauvegarde du rapport exécutif
        os.makedirs('bi_output', exist_ok=True)
        with open('bi_output/executive_summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(executive_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Dashboard exécutif final
        self.create_final_executive_dashboard()
        
        print("✅ Rapport de synthèse exécutif créé")
        
        # Affichage du résumé
        print("\n🎯 SYNTHÈSE EXÉCUTIVE NETFLIX:")
        print("=" * 50)
        
        if self.bi_report:
            summary = self.bi_report['executive_summary']
            print(f"📊 Portfolio: {summary['total_content']:,} contenus")
            print(f"💰 Investissement: ${summary['total_investment_millions']:,.0f}M")
            print(f"📈 ROI Moyen: {summary['average_roi']:.2f}")
            print(f"🎭 Diversité: {summary['content_diversity']} genres")
        
        if 'financial_performance' in self.evaluation_results:
            grade = self.evaluation_results['financial_performance']['overall_financial_score']['grade']
            print(f"🏆 Note Financière: {grade}")
        
        if 'strategic_positioning' in self.evaluation_results:
            level = self.evaluation_results['strategic_positioning']['strategic_positioning_score']['level']
            print(f"🎯 Positionnement: {level}")
        
        print(f"\n💡 Insights Stratégiques: {len(self.strategic_insights)}")
        for insight in self.strategic_insights[:3]:
            print(f"  • {insight['insight']} ({insight['priority']})")
    
    def create_action_plan(self):
        """Crée un plan d'action basé sur les insights."""
        action_plan = []
        
        # Priorisation des actions
        critical_actions = [i for i in self.strategic_insights if i.get('priority') == 'Critique']
        high_actions = [i for i in self.strategic_insights if i.get('priority') == 'Haute']
        strategic_actions = [i for i in self.strategic_insights if i.get('priority') == 'Stratégique']
        
        # Plan d'action structuré
        if critical_actions:
            action_plan.append({
                'phase': 'Immédiat (0-3 mois)',
                'priority': 'Critique',
                'actions': [a['action'] for a in critical_actions],
                'expected_impact': 'Stabilisation des performances'
            })
        
        if high_actions:
            action_plan.append({
                'phase': 'Court terme (3-12 mois)',
                'priority': 'Haute',
                'actions': [a['action'] for a in high_actions],
                'expected_impact': 'Amélioration significative des KPIs'
            })
        
        if strategic_actions:
            action_plan.append({
                'phase': 'Long terme (12+ mois)',
                'priority': 'Stratégique',
                'actions': [a['action'] for a in strategic_actions],
                'expected_impact': 'Renforcement de la position concurrentielle'
            })
        
        return action_plan
    
    def create_final_executive_dashboard(self):
        """Crée le dashboard exécutif final."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Scores de performance
        if 'financial_performance' in self.evaluation_results and 'strategic_positioning' in self.evaluation_results:
            financial_score = self.evaluation_results['financial_performance']['overall_financial_score']['score']
            strategic_score = self.evaluation_results['strategic_positioning']['strategic_positioning_score']['score']
            
            scores = [financial_score, strategic_score]
            labels = ['Performance\nFinancière', 'Positionnement\nStratégique']
            colors = ['green' if s > 1 else 'orange' if s > 0.8 else 'red' for s in scores]
            
            axes[0, 0].bar(labels, scores, color=colors)
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Scores de Performance Globaux')
            axes[0, 0].set_ylabel('Score (1.0 = Benchmark)')
            axes[0, 0].set_ylim(0, 2)
        
        # Distribution des insights par priorité
        if self.strategic_insights:
            priority_counts = {}
            for insight in self.strategic_insights:
                priority = insight.get('priority', 'Autre')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            axes[0, 1].pie(priority_counts.values(), labels=priority_counts.keys(), autopct='%1.0f%%')
            axes[0, 1].set_title('Distribution des Insights par Priorité')
        
        # Métriques clés
        if self.bi_report and 'executive_summary' in self.bi_report:
            summary = self.bi_report['executive_summary']
            metrics = ['Contenus', 'Investment (M$)', 'ROI', 'Genres']
            values = [
                summary['total_content'] / 1000,  # En milliers
                summary['total_investment_millions'] / 1000,  # En milliards
                summary['average_roi'],
                summary['content_diversity']
            ]
            
            axes[0, 2].bar(metrics, values, color=['blue', 'green', 'orange', 'purple'])
            axes[0, 2].set_title('Métriques Clés (Échelles Ajustées)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Matrice risques/opportunités
        if self.strategic_insights:
            risks = [i for i in self.strategic_insights if i.get('type') == 'Risque']
            opportunities = [i for i in self.strategic_insights if i.get('type') == 'Opportunité']
            
            risk_count = len(risks)
            opp_count = len(opportunities)
            
            axes[1, 0].bar(['Risques', 'Opportunités'], [risk_count, opp_count], 
                          color=['red', 'green'], alpha=0.7)
            axes[1, 0].set_title('Matrice Risques vs Opportunités')
            axes[1, 0].set_ylabel('Nombre d\'Insights')
        
        # Timeline du plan d'action
        action_plan = self.create_action_plan()
        if action_plan:
            phases = [phase['phase'] for phase in action_plan]
            action_counts = [len(phase['actions']) for phase in action_plan]
            
            axes[1, 1].barh(phases, action_counts, color=['red', 'orange', 'green'])
            axes[1, 1].set_title('Plan d\'Action par Phase')
            axes[1, 1].set_xlabel('Nombre d\'Actions')
        
        # Radar final de santé business
        if 'financial_performance' in self.evaluation_results and 'strategic_positioning' in self.evaluation_results:
            categories = ['ROI', 'Diversité', 'Fraîcheur', 'Portée Géo']
            
            financial_eval = self.evaluation_results['financial_performance']
            strategic_eval = self.evaluation_results['strategic_positioning']
            
            values = [
                financial_eval['roi_vs_industry']['performance_ratio'],
                strategic_eval['content_diversity']['performance_ratio'],
                strategic_eval['content_freshness']['performance_ratio'],
                strategic_eval['geographic_reach']['performance_ratio']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 3, 6, projection='polar')
            ax_radar.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax_radar.fill(angles, values, alpha=0.25, color='blue')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories)
            ax_radar.set_ylim(0, 2)
            ax_radar.set_title('Santé Business Globale')
            ax_radar.grid(True)
        
        plt.tight_layout()
        plt.savefig('Graphiques générés/netflix_final_executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_evaluation(self):
        """Exécute l'évaluation BI complète."""
        print("🚀 Démarrage de l'évaluation Business Intelligence complète...")
        
        if not self.bi_report:
            print("❌ Aucun rapport BI disponible pour évaluation")
            return
        
        try:
            os.makedirs('Graphiques générés', exist_ok=True)
            os.makedirs('bi_output', exist_ok=True)
            
            self.evaluate_financial_performance()
            self.analyze_strategic_positioning()
            self.generate_executive_insights()
            self.create_executive_summary_report()
            
            print("\n🎉 Évaluation Business Intelligence complète terminée !")
            print("📁 Rapports disponibles dans bi_output/")
            print("📊 Dashboards dans Graphiques générés/")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation BI : {e}")

def main():
    """Fonction principale."""
    evaluator = NetflixBIEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
