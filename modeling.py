"""
Week 3: Modeling (Boosting) & MLflow
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

def load_preprocessed_data():
    """Charger les données preprocessées"""
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    return X_train, X_test, y_train, y_test

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boosting Classifier"""
    print("\n=== Gradient Boosting ===")
    
    with mlflow.start_run(run_name="GradientBoosting"):
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        print(f"AUC: {auc:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # MLflow logging
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.sklearn.log_model(model, "model")
        
        return model, auc, y_pred, y_proba

def train_adaboost(X_train, y_train, X_test, y_test):
    """AdaBoost Classifier"""
    print("\n=== AdaBoost ===")
    
    with mlflow.start_run(run_name="AdaBoost"):
        model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        print(f"AUC: {auc:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 1.0)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.sklearn.log_model(model, "model")
        
        return model, auc, y_pred, y_proba

def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost Classifier"""
    print("\n=== XGBoost ===")
    
    with mlflow.start_run(run_name="XGBoost"):
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        print(f"AUC: {auc:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.xgboost.log_model(model, "model")
        
        return model, auc, y_pred, y_proba

def compare_models(results):
    """Comparer les modèles"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC comparison
    models = list(results.keys())
    aucs = [results[m]['auc'] for m in models]
    
    axes[0].bar(models, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('AUC Score')
    axes[0].set_title('Comparaison des Modèles')
    axes[0].set_ylim([0.7, 1.0])
    
    for i, v in enumerate(aucs):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Confusion Matrix du meilleur modèle
    best_model = max(results.keys(), key=lambda k: results[k]['auc'])
    cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'Confusion Matrix - {best_model}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('data/modeling_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualisations sauvegardées: data/modeling_results.png")
    
    return best_model

def save_best_model(best_model_name, results):
    """Sauvegarder le meilleur modèle"""
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, 'data/best_model.pkl')
    print(f"\n✓ Meilleur modèle ({best_model_name}) sauvegardé: data/best_model.pkl")

if __name__ == "__main__":
    print("=== SEMAINE 3: MODELING (BOOSTING) & MLFLOW ===\n")
    
    # Charger les données
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Configuration MLflow
    mlflow.set_experiment("Cannabis_Prediction_Boosting")
    
    # Entraîner les modèles
    results = {}
    
    gb_model, gb_auc, gb_pred, gb_proba = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['GradientBoosting'] = {'model': gb_model, 'auc': gb_auc, 'y_pred': gb_pred, 'y_test': y_test}
    
    ada_model, ada_auc, ada_pred, ada_proba = train_adaboost(X_train, y_train, X_test, y_test)
    results['AdaBoost'] = {'model': ada_model, 'auc': ada_auc, 'y_pred': ada_pred, 'y_test': y_test}
    
    xgb_model, xgb_auc, xgb_pred, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
    results['XGBoost'] = {'model': xgb_model, 'auc': xgb_auc, 'y_pred': xgb_pred, 'y_test': y_test}
    
    # Comparer et sauvegarder
    best_model = compare_models(results)
    save_best_model(best_model, results)
    
    print(f"\n🏆 Meilleur modèle: {best_model} (AUC: {results[best_model]['auc']:.4f})")
    print("\n✓ Pour voir les expériences MLflow, lancez: mlflow ui")
