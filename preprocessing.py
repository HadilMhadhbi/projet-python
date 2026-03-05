"""
Week 2: Preprocessing & Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Charger les données"""
    df = pd.read_csv('drug_consumption.csv')
    print(f"Dataset chargé: {df.shape}")
    return df

def encode_target(df):
    """Encoder la variable cible Cannabis"""
    # CL0-CL1: Non-user (0), CL2-CL6: User (1)
    target_map = {'CL0': 0, 'CL1': 0, 'CL2': 1, 'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1}
    df['Cannabis_Binary'] = df['Cannabis'].map(target_map)
    print(f"\nDistribution cible:\n{df['Cannabis_Binary'].value_counts(normalize=True)}")
    return df

def feature_engineering(df):
    """Créer de nouvelles features"""
    # Personality risk score
    df['Personality_Risk'] = (df['Nscore'] + df['Impulsive'] + df['SS']) / 3
    
    # Age-Education interaction
    df['Age_Education'] = df['Age'] * df['Education']
    
    print(f"\nNouvelles features créées: Personality_Risk, Age_Education")
    return df

def preprocess_and_split(df):
    """Preprocessing et split des données"""
    # Features
    feature_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                   'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
                   'Impulsive', 'SS', 'Personality_Risk', 'Age_Education']
    
    X = df[feature_cols]
    y = df['Cannabis_Binary']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE pour équilibrer
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\nTrain shape après SMOTE: {X_train_balanced.shape}")
    print(f"Distribution après SMOTE: {np.bincount(y_train_balanced)}")
    
    # Sauvegarder
    np.save('data/X_train.npy', X_train_balanced)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train_balanced)
    np.save('data/y_test.npy', y_test)
    joblib.dump(scaler, 'data/scaler.pkl')
    joblib.dump(feature_cols, 'data/feature_names.pkl')
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def visualize_preprocessing(X_train, y_train):
    """Visualiser les résultats du preprocessing"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution des classes
    axes[0].bar(['Non-User', 'User'], np.bincount(y_train))
    axes[0].set_title('Distribution après SMOTE')
    axes[0].set_ylabel('Count')
    
    # Correlation des features
    df_train = pd.DataFrame(X_train[:, :12])
    corr = df_train.corr()
    sns.heatmap(corr, ax=axes[1], cmap='coolwarm', center=0)
    axes[1].set_title('Corrélation des Features')
    
    plt.tight_layout()
    plt.savefig('data/preprocessing_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualisations sauvegardées: data/preprocessing_results.png")

if __name__ == "__main__":
    print("=== SEMAINE 2: PREPROCESSING & FEATURE ENGINEERING ===\n")
    
    df = load_data()
    df = encode_target(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    visualize_preprocessing(X_train, y_train)
    
    print("\n✓ Preprocessing terminé!")
