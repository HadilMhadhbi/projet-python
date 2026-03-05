# 🧪 Projet de Prédiction de Consommation de Drogues

## 📋 Description
Ce projet implémente un pipeline complet de data science pour prédire la consommation de cannabis basé sur des caractéristiques démographiques et de personnalité, inspiré du projet [Python-for-Data-Science-Project](https://github.com/haythemghz/Python-for-Data-Science-Project/).

## 🗓️ Planning de 3 Semaines

### 📅 Semaine 1: Setup, Scraping & EDA
- Configuration de l'environnement
- Chargement et exploration des données
- Analyse exploratoire complète
- Visualisations des distributions

### 📅 Semaine 2: Preprocessing & Feature Engineering
- Encodage des variables catégorielles
- Création de nouvelles caractéristiques
- Sélection des features importantes
- Gestion du déséquilibre des classes

### 📅 Semaine 3: Modeling (Boosting) & MLflow
- Implémentation d'algorithmes de boosting
- Optimisation des hyperparamètres
- Suivi des expériences avec MLflow
- Comparaison et sélection du meilleur modèle

## 🚀 Installation

1. **Cloner le projet**
```bash
git clone <votre-repo>
cd projet-python
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Vérifier les données**
Assurez-vous d'avoir le fichier `drug_consumption.data` dans le répertoire racine.

## 🎯 Utilisation

### Option 1: Script Principal (Recommandé)
```bash
python main.py
```

### Option 2: Exécution par Module
```bash
# Scraping
python scraping.py

# Exploration des données
python data_exploration.py

# Modélisation
python modeling.py

# API
python app.py
```

### Option 3: API FastAPI
```bash
# Démarrer l'API
python app.py
# ou
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📊 Suivi des Expériences MLflow

Après avoir exécuté la semaine 3, lancez l'interface MLflow :
```bash
mlflow ui
```
Puis ouvrez http://localhost:5000 dans votre navigateur.

## 📁 Structure du Projet

```
projet-python/
├── main.py                          # Point d'entrée principal
├── scraping.py                      # Module de scraping
├── data_exploration.py              # Analyse exploratoire
├── modeling.py                      # Modélisation et ML
├── app.py                          # API FastAPI
├── config.py                       # Configuration
├── requirements.txt                # Dépendances
├── drug_consumption.data          # Dataset principal
├── data/                          # Données et résultats
│   ├── *.png                     # Visualisations
│   ├── *.csv                     # Données processées
│   └── *.pkl                     # Modèles sauvegardés
├── models/                        # Modèles entraînés
└── mlruns/                       # Expériences MLflow
```

## 🛠️ Technologies Utilisées

- **Python 3.8+**: Langage principal
- **Pandas/NumPy**: Manipulation des données
- **Scikit-learn**: Machine Learning
- **XGBoost**: Gradient Boosting avancé
- **MLflow**: Suivi des expériences
- **Matplotlib/Seaborn**: Visualisations
- **SMOTE**: Gestion du déséquilibre des classes

## 📈 Modèles Implémentés

1. **Gradient Boosting Classifier**
2. **AdaBoost Classifier**
3. **XGBoost Classifier**

Chaque modèle est évalué avec :
- Cross-validation 5-fold
- Métriques AUC-ROC
- Optimisation des hyperparamètres
- Suivi MLflow

## 📊 Résultats Attendus

- Modèles de prédiction avec AUC > 0.80
- Comparaison détaillée des algorithmes
- Visualisations des performances
- Pipeline reproductible

## 🔄 Workflow Complet

1. **EDA**: Analyse exploratoire des données
2. **Preprocessing**: Nettoyage et transformation
3. **Feature Engineering**: Création de nouvelles variables
4. **Modeling**: Entraînement des modèles de boosting
5. **Evaluation**: Comparaison et sélection
6. **Tracking**: Suivi avec MLflow

## 📝 Fichiers Générés

- `data/demographic_distribution.png`: Distributions démographiques
- `data/correlation_matrix.png`: Matrice de corrélation
- `data/preprocessing_results.png`: Résultats du preprocessing
- `data/modeling_results.png`: Comparaison des modèles
- `data/final_model.pkl`: Meilleur modèle sauvegardé
- `RAPPORT_FINAL.md`: Rapport complet du projet

## 🎯 Objectifs Pédagogiques

- Maîtriser le pipeline complet de data science
- Comprendre les algorithmes de boosting
- Utiliser MLflow pour le suivi d'expériences
- Appliquer les bonnes pratiques de preprocessing
- Évaluer et comparer des modèles ML

## 🚀 Extensions Possibles

1. **Déploiement**: API FastAPI pour les prédictions
2. **Interface**: Dashboard interactif
3. **Monitoring**: Suivi en production
4. **Multi-target**: Prédiction d'autres drogues

## 📞 Support

Pour toute question ou problème, consultez :
- Les logs d'exécution
- L'interface MLflow
- Les visualisations générées

---
*Projet réalisé dans le cadre du cours Python for Data Science*