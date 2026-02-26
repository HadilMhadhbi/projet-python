# projet-python
# Python-for-Data-Science-Project
# Drug Consumption Analysis & Prediction

## 👩‍💻 Équipe
- Member : **Hadil Mhadhbi**
- Tuteur : Haythem Ghazouani

---

## 📌 Description du Projet

Ce projet vise à analyser et prédire la consommation de cannabis en utilisant le dataset **Drug Consumption (Quantified)**.

L’objectif est de mettre en œuvre un pipeline complet de Machine Learning, incluant :

- Analyse exploratoire des données (EDA)
- Prétraitement des données
- Modélisation et comparaison de plusieurs algorithmes
- Évaluation des performances

---

## 🎯 Objectifs

- Analyser les caractéristiques démographiques et psychologiques des individus  
- Étudier la relation entre les traits de personnalité et la consommation de drogue  
- Prédire : **Cannabis usage (User / Non-User)**  
- Comparer plusieurs modèles de classification  

---

## 📊 Dataset

- **Source :** UCI Machine Learning Repository  
- **Nom :** Drug Consumption (Quantified)  
- **Taille :** 1885 lignes, 12 variables principales  
- **Target :** Cannabis consumption  

### Features principales :

**Données démographiques :**
- Age  
- Gender  
- Education  
- Country  
- Ethnicity  

**Scores psychologiques :**
- Neuroticism  
- Extraversion  
- Openness  
- Agreeableness  
- Conscientiousness  
- Impulsiveness  
- Sensation Seeking  

**Transformation de la Target :**
- CL0 → Non-User (0)  
- CL1–CL6 → User (1)

---

## 🗺 Roadmap (6 semaines)

### Phase 1 : Fondations (Semaine 1)

- [x] Choix et validation du dataset  
- [x] Structure GitHub  
- [x] README et documentation  
- [x] Analyse Exploratoire des Données (EDA)  
- [ ] Nettoyage des données  

---

### Phase 2 : Pipeline Machine Learning (Semaines 2-3)

- [ ] Prétraitement des données  
- [ ] Encodage et normalisation  
- [ ] Train/Test Split  
- [ ] Modélisation :
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine  
- [ ] Évaluation (Accuracy, Confusion Matrix, F1-score)  
- [ ] Comparaison des modèles  

---

### Phase 3 : Amélioration & Optimisation (Semaines 4-6)

- [ ] Feature importance analysis  
- [ ] Hyperparameter tuning (GridSearchCV)  
- [ ] Cross-validation  
- [ ] Visualisations avancées  
- [ ] Conclusion et insights  

---

## ⚙ Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/drug-consumption-project.git
cd drug-consumption-project

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
