"""
Script Principal - Semaines 2 & 3
Preprocessing + Modeling
"""

import os
import sys

def run_week2():
    """Exécuter la semaine 2"""
    print("\n" + "="*60)
    print("SEMAINE 2: PREPROCESSING & FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    import preprocessing
    
def run_week3():
    """Exécuter la semaine 3"""
    print("\n" + "="*60)
    print("SEMAINE 3: MODELING (BOOSTING) & MLFLOW")
    print("="*60 + "\n")
    
    import modeling

def main():
    """Exécution complète"""
    print("\n" + "🧪 "*20)
    print("PROJET: PRÉDICTION DE CONSOMMATION DE CANNABIS")
    print("🧪 "*20 + "\n")
    
    # Créer le dossier data s'il n'existe pas
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✓ Dossier 'data' créé\n")
    
    # Semaine 2
    run_week2()
    
    # Semaine 3
    run_week3()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLET TERMINÉ!")
    print("="*60)
    print("\n📊 Fichiers générés:")
    print("  - data/X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("  - data/scaler.pkl, feature_names.pkl")
    print("  - data/best_model.pkl")
    print("  - data/preprocessing_results.png")
    print("  - data/modeling_results.png")
    print("\n🔬 Pour voir les expériences MLflow:")
    print("  mlflow ui")
    print("  Puis ouvrir: http://localhost:5000\n")

if __name__ == "__main__":
    main()
