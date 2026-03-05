"""
EDA (Exploratory Data Analysis) - Week 1
Drug Consumption Dataset Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

column_names = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
    'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh',
    'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]

def load_data(file_path="drug_consumption.data"):
    """Load dataset"""
    df = pd.read_csv(file_path, names=column_names)
    print(f"Dataset shape: {df.shape}")
    return df

def explore_data(df):
    """Exploratory Data Analysis"""
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== Statistical Summary ===")
    print(df.describe())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Cannabis distribution
    plt.figure(figsize=(8,5))
    df['Cannabis'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Cannabis Consumption Distribution")
    plt.ylabel("Percentage")
    plt.savefig("data/cannabis_distribution.png")
    plt.close()
    
    # Demographic features
    categorical_features = ['Gender', 'Education', 'Country', 'Ethnicity']
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()
    for i, feature in enumerate(categorical_features):
        sns.countplot(x=feature, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig("data/demographic_features.png")
    plt.close()
    
    # Personality scores
    personality_features = ['Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS']
    plt.figure(figsize=(15,10))
    for i, feature in enumerate(personality_features):
        plt.subplot(3,3,i+1)
        sns.boxplot(y=df[feature])
        plt.title(feature)
    plt.tight_layout()
    plt.savefig("data/personality_boxplots.png")
    plt.close()
    
    # Correlation heatmap
    numerical_features = ['Age','Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS']
    plt.figure(figsize=(10,8))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("data/correlation_heatmap.png")
    plt.close()
    
    print("\nEDA completed. Plots saved in 'data/' directory.")
    return df

if __name__ == "__main__":
    df = load_data()
    df = explore_data(df)