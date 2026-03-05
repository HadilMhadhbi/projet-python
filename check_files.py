"""
Script de Vérification - Fichiers Générés
"""

import os

def check_files():
    """Vérifier quels fichiers sont générés"""
    
    print("="*60)
    print("VÉRIFICATION DES FICHIERS GÉNÉRÉS")
    print("="*60 + "\n")
    
    # Fichiers attendus
    files_week1 = [
        "data/demographic_distribution.png",
        "data/correlation_heatmap.png",
        "data/personality_boxplots.png",
        "data/cannabis_distribution.png"
    ]
    
    files_week2 = [
        "data/X_train.npy",
        "data/X_test.npy",
        "data/y_train.npy",
        "data/y_test.npy",
        "data/scaler.pkl",
        "data/feature_names.pkl",
        "data/preprocessing_results.png"
    ]
    
    files_week3 = [
        "data/best_model.pkl",
        "data/modeling_results.png",
        "mlruns/"
    ]
    
    # Vérification Semaine 1
    print("📅 SEMAINE 1 - EDA:")
    week1_ok = 0
    for f in files_week1:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        if exists:
            week1_ok += 1
    print(f"  → {week1_ok}/{len(files_week1)} fichiers\n")
    
    # Vérification Semaine 2
    print("📅 SEMAINE 2 - PREPROCESSING:")
    week2_ok = 0
    for f in files_week2:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        if exists:
            week2_ok += 1
    print(f"  → {week2_ok}/{len(files_week2)} fichiers\n")
    
    # Vérification Semaine 3
    print("📅 SEMAINE 3 - MODELING:")
    week3_ok = 0
    for f in files_week3:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        if exists:
            week3_ok += 1
    print(f"  → {week3_ok}/{len(files_week3)} fichiers\n")
    
    # Résumé
    total = week1_ok + week2_ok + week3_ok
    total_expected = len(files_week1) + len(files_week2) + len(files_week3)
    
    print("="*60)
    print(f"TOTAL: {total}/{total_expected} fichiers générés")
    print("="*60 + "\n")
    
    # Recommandations
    if week2_ok < len(files_week2):
        print("⚠️  Lancez: python week2_preprocessing.py")
    if week3_ok < len(files_week3):
        print("⚠️  Lancez: python week3_modeling.py")
    if total == total_expected:
        print("🎉 TOUT EST PRÊT! Projet complet!")

if __name__ == "__main__":
    check_files()
