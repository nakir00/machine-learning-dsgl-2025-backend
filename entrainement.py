
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import warnings
# Pour la gestion du déséquilibre
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import joblib

print("=" * 80)
print("DÉTECTION DE FRAUDE BANCAIRE - MACHINE LEARNING")
print("=" * 80)


print("\n question 1: Chargement des données...")

df = pd.read_csv('static/data/creditcarddata.csv')
print(f"✓ Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nAperçu des données :")
print(df.head())
print("\nInformations sur les données :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())



print("\n" + "="*80)
print("questions 2 : Modèles de classification pour la détection de fraude")
print("="*80)

modeles_liste = [
    "1. Régression Logistique (Logistic Regression)",
    "2. Arbre de Décision (Decision Tree)",
    "3. Forêt Aléatoire (Random Forest)",
    "4. Gradient Boosting",
    "5. Support Vector Machine (SVM)",
    "6. K-Nearest Neighbors (KNN)",
    "7. Naive Bayes"
]

for modele in modeles_liste:
    print(f"  {modele}")


print("\n" + "="*80)
print("[TÂCHE 3] Préparation des données")
print("="*80)

# 3.1 Vérification des valeurs manquantes
print("\n3.1 Vérification des valeurs manquantes :")
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)
if valeurs_manquantes.sum() > 0:
    print("⚠ Valeurs manquantes détectées. Traitement en cours...")
    df = df.dropna()
    print(f"✓ Valeurs manquantes supprimées. Nouvelles dimensions : {df.shape}")
else:
    print("✓ Aucune valeur manquante détectée")

# 3.2 Vérification des doublons
print("\n3.2 Vérification des doublons :")
nb_doublons = df.duplicated().sum()
print(f"Nombre de doublons : {nb_doublons}")
if nb_doublons > 0:
    print("⚠ Doublons détectés. Suppression en cours...")
    df = df.drop_duplicates()
    print(f"✓ Doublons supprimés. Nouvelles dimensions : {df.shape}")
else:
    print("✓ Aucun doublon détecté")

# 3.3 Vérification et traitement des valeurs aberrantes (outliers)
print("\n3.3 Vérification et traitement des valeurs aberrantes :")
colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.drop('PotentialFraud')
print(f"Colonnes numériques analysées : {list(colonnes_numeriques)}")

# Fonction pour détecter les outliers avec IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, lower_bound, upper_bound

# Fonction pour calculer le Z-score
def calculate_zscore(data, column):
    mean = data[column].mean()
    std = data[column].std()
    zscore = (data[column] - mean) / std
    return zscore

# Analyse des outliers pour chaque colonne numérique
print("\nAnalyse détaillée des valeurs aberrantes :")
outliers_info = {}

for col in colonnes_numeriques:
    outliers_mask, lower, upper = detect_outliers_iqr(df, col)
    nb_outliers = outliers_mask.sum()
    pct_outliers = (nb_outliers / len(df)) * 100
    
    print(f"\n  {col} :")
    print(f"    - Nombre d'outliers (IQR) : {nb_outliers} ({pct_outliers:.2f}%)")
    print(f"    - Limites : [{lower:.2f}, {upper:.2f}]")
    print(f"    - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
    
    outliers_info[col] = {
        'count': nb_outliers,
        'percentage': pct_outliers,
        'lower': lower,
        'upper': upper
    }

# Traitement spécifique pour TransactionAmount (approche recommandée)
print("\n⚠ Traitement des outliers pour TransactionAmount :")
print("  → Création de features enrichies au lieu de supprimer les outliers")

# 1. Créer un indicateur de transaction élevée (> 99e percentile)
percentile_99 = df['TransactionAmount'].quantile(0.99)
df['TransactionAmount_is_high'] = (df['TransactionAmount'] > percentile_99).astype(int)
print(f"  ✓ Feature 'TransactionAmount_is_high' créée (seuil: {percentile_99:.2f})")
print(f"    Transactions élevées détectées : {df['TransactionAmount_is_high'].sum()}")

# 2. Calculer le Z-score comme feature
df['TransactionAmount_zscore'] = calculate_zscore(df, 'TransactionAmount')
print(f"  ✓ Feature 'TransactionAmount_zscore' créée")
print(f"    Z-score min: {df['TransactionAmount_zscore'].min():.2f}, max: {df['TransactionAmount_zscore'].max():.2f}")

# 3. Transformation logarithmique pour réduire l'asymétrie
df['TransactionAmount_log'] = np.log1p(df['TransactionAmount'])
print(f"  ✓ Feature 'TransactionAmount_log' créée (log1p transformation)")

# 4. Créer des catégories de montant
bins = [0, 10, 50, 100, 500, np.inf]
labels = ['Très faible', 'Faible', 'Moyen', 'Élevé', 'Très élevé']
df['TransactionAmount_category'] = pd.cut(df['TransactionAmount'], bins=bins, labels=labels)
df['TransactionAmount_category'] = df['TransactionAmount_category'].cat.codes
print(f"  ✓ Feature 'TransactionAmount_category' créée (5 catégories)")

# Vérification des valeurs impossibles pour Age
print("\n  Vérification de l'âge :")
age_invalid = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
if age_invalid > 0:
    print(f"  ⚠ {age_invalid} âges invalides détectés. Correction en cours...")
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
    print(f"  ✓ Âges invalides supprimés")
else:
    print(f"  ✓ Tous les âges sont valides (0-120 ans)")

# Visualisation de l'impact de la transformation logarithmique

# Graphique 1 : Distribution originale
plt.figure(figsize=(10, 6))
plt.hist(df['TransactionAmount'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution originale de TransactionAmount', fontsize=14, fontweight='bold')
plt.xlabel('Montant', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.axvline(percentile_99, color='red', linestyle='--', linewidth=2, label=f'99e percentile ({percentile_99:.2f})')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/preprocessing/distribution_originale.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : distribution_originale.png")

# Graphique 2 : Distribution log-transformée
plt.figure(figsize=(10, 6))
plt.hist(df['TransactionAmount_log'], bins=50, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution log-transformée de TransactionAmount', fontsize=14, fontweight='bold')
plt.xlabel('log(Montant + 1)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/preprocessing/distribution_log.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : distribution_log.png")

# Graphique 3 : Distribution des Z-scores
plt.figure(figsize=(10, 6))
plt.hist(df['TransactionAmount_zscore'], bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution des Z-scores de TransactionAmount', fontsize=14, fontweight='bold')
plt.xlabel('Z-score', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.axvline(-3, color='red', linestyle='--', linewidth=2, label='Seuil -3')
plt.axvline(3, color='red', linestyle='--', linewidth=2, label='Seuil +3')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/preprocessing/distribution_zscore.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : distribution_zscore.png")

print("\n✓ Traitement des valeurs aberrantes terminé")
print(f"  Dimensions finales : {df.shape}")

# 3.4 Analyse du déséquilibre de classes
print("\n3.4 Analyse du déséquilibre de classes :")
distribution_classe = df['PotentialFraud'].value_counts()
print(distribution_classe)
print(f"\nProportion de fraudes : {distribution_classe[1]/len(df)*100:.2f}%")
print(f"Proportion de non-fraudes : {distribution_classe[0]/len(df)*100:.2f}%")

# Visualisation du déséquilibre - Graphique 1 : Bar plot
plt.figure(figsize=(8, 6))
distribution_classe.plot(kind='bar', color=['green', 'red'])
plt.title('Distribution des classes', fontsize=14, fontweight='bold')
plt.xlabel('PotentialFraud', fontsize=12)
plt.ylabel('Nombre d\'observations', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/preprocessing/distribution_classes_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : distribution_classes_barplot.png")

# Visualisation du déséquilibre - Graphique 2 : Pie chart
plt.figure(figsize=(8, 8))
plt.pie(distribution_classe, labels=['Non-Fraude', 'Fraude'], autopct='%1.1f%%', 
        colors=['green', 'red'], startangle=90, textprops={'fontsize': 12})
plt.title('Proportion des classes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/preprocessing/distribution_classes_piechart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : distribution_classes_piechart.png")

# Gestion du déséquilibre si nécessaire
if distribution_classe[1] / distribution_classe[0] < 0.3:
    print("\n⚠ Déséquilibre important détecté. Application de SMOTE recommandée.")
    desequilibre = True
else:
    desequilibre = False


print("\n" + "="*80)
print(" question 4 : Division des données")
print("="*80)

X = df.drop('PotentialFraud', axis=1)
y = df['PotentialFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"  Données d'entraînement : {X_train.shape[0]} observations ({70}%)")
print(f"  Données de test : {X_test.shape[0]} observations ({30}%)")
print(f"\nDistribution dans le train :")
print(y_train.value_counts())
print(f"\nDistribution dans le test :")
print(y_test.value_counts())


if desequilibre:
    print("\n  Application de SMOTE pour équilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  Après SMOTE - Train : {X_train.shape[0]} observations")
    print(f"Distribution après SMOTE :")
    print(y_train.value_counts())



print("\n" + "="*80)
print("NORMALISATION DES DONNÉES")
print("="*80)

print("\nPlusieurs méthodes de normalisation disponibles :")
print("  1. StandardScaler : (X - mean) / std → Distribution N(0,1)")
print("  2. MinMaxScaler : (X - min) / (max - min) → Échelle [0,1]")
print("  3. RobustScaler : Utilise la médiane et IQR → Robuste aux outliers")

print("\nStatistiques AVANT normalisation (échantillon) :")
print(X_train[['Age', 'TransactionAmount']].describe())

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

print("\n→ Application de StandardScaler (méthode principale)")
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)
print("  ✓ StandardScaler appliqué")

print("\n→ Application de RobustScaler (robuste aux outliers)")
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)
print("  ✓ RobustScaler appliqué")

print("\n→ Application de MinMaxScaler (échelle 0-1)")
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)
print("  ✓ MinMaxScaler appliqué")

X_train_scaled = X_train_minmax
X_test_scaled = X_test_minmax
scaler = scaler_minmax

print("\n✓ Normalisation principale : StandardScaler sélectionné")

X_train_scaled_df = pd.DataFrame(
    X_train_scaled, 
    columns=X_train.columns
)
print("\nStatistiques APRÈS normalisation (échantillon) :")
print(X_train_scaled_df[['Age', 'TransactionAmount']].describe())

colonnes_viz = ['Age', 'TransactionAmount']

for idx, col in enumerate(colonnes_viz):
    # Graphique AVANT normalisation
    plt.figure(figsize=(10, 6))
    plt.hist(X_train[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.title(f'{col} - AVANT normalisation', fontsize=14, fontweight='bold')
    plt.xlabel('Valeur', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'static/images/normalisation/avant/{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : 06_normalisation_avant_{col}.png")
    
    # Graphique APRÈS normalisation
    plt.figure(figsize=(10, 6))
    col_index = list(X_train.columns).index(col)
    plt.hist(X_train_scaled[:, col_index], bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.title(f'{col} - APRÈS normalisation (StandardScaler)', fontsize=14, fontweight='bold')
    plt.xlabel('Valeur normalisée', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'static/images/normalisation/apres/{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : 07_normalisation_apres_{col}.png")

print("\n" + "="*80)


print("\n" + "="*80)
print("question 5 : Entraînement des modèles")
print("="*80)

modeles = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

modeles_entraines = {}
for nom, modele in modeles.items():
    print(f"\n  Entraînement de {nom}...")
    modele.fit(X_train_scaled, y_train)
    modeles_entraines[nom] = modele
    print(f"    {nom} entraîné avec succès")


print("\n" + "="*80)
print("question 6 : Évaluation des modèles - Matrices de confusion")
print("="*80)

for idx, (nom, modele) in enumerate(modeles_entraines.items()):
    y_pred = modele.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraude', 'Fraude'],
                yticklabels=['Non-Fraude', 'Fraude'],
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title(f'Matrice de confusion - {nom}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    
    nom_fichier = f'static/images/matrice_confusion/{nom.replace(" ", "_")}.png'
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Matrice de confusion sauvegardée : {nom_fichier}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")


print("\n" + "="*80)
print("question 7 : Calcul des métriques de performance")
print("="*80)

resultats = []
courbes_roc = {}
courbes_pr = {}

for nom, modele in modeles_entraines.items():
    y_pred = modele.predict(X_test_scaled)
    
    if hasattr(modele, 'predict_proba'):
        y_pred_proba = modele.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(modele, 'decision_function'):
        y_pred_proba = modele.decision_function(X_test_scaled)
    else:
        y_pred_proba = y_pred  
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        courbes_roc[nom] = (fpr, tpr, roc_auc)
    except:
        roc_auc = np.nan
        courbes_roc[nom] = None
    
    try:
        avg_precision = average_precision_score(y_test, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        courbes_pr[nom] = (recall_curve, precision_curve, avg_precision)
    except:
        avg_precision = np.nan
        courbes_pr[nom] = None
    
    resultats.append({
        'Modèle': nom,
        'Accuracy': acc,
        'Précision': prec,
        'Rappel': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Avg Precision': avg_precision
    })
    
    print(f"\n{nom} :")
    print(f"  Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Précision       : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Rappel (Recall) : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score        : {f1:.4f} ({f1*100:.2f}%)")
    print(f"  ROC-AUC         : {roc_auc:.4f}" if not np.isnan(roc_auc) else "  ROC-AUC         : N/A")
    print(f"  Avg Precision   : {avg_precision:.4f}" if not np.isnan(avg_precision) else "  Avg Precision   : N/A")


print("\n" + "="*80)
print("question 8 : Comparaison des modèles")
print("="*80)

df_resultats = pd.DataFrame(resultats)
df_resultats = df_resultats.sort_values('F1-Score', ascending=False)

print("\nTableau comparatif des performances :")
print(df_resultats.to_string(index=False))

metriques = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'ROC-AUC']
numero_base = 15

for idx, metrique in enumerate(metriques):
    plt.figure(figsize=(12, 6))
    
    df_plot = df_resultats[['Modèle', metrique]].dropna()
    df_plot_sorted = df_plot.sort_values(metrique, ascending=True)
    
    bars = plt.barh(df_plot_sorted['Modèle'], df_plot_sorted[metrique], color='steelblue')
    plt.title(f'Comparaison des modèles - {metrique}', fontsize=14, fontweight='bold')
    plt.xlabel(metrique, fontsize=12)
    plt.ylabel('Modèle', fontsize=12)
    plt.xlim([0, 1.1])
    plt.grid(axis='x', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, df_plot_sorted[metrique])):
        plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    numero = numero_base + idx
    plt.savefig(f'static/images/comparaison/{metrique.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {numero:02d}_comparaison_{metrique.lower().replace('-', '_')}.png")


print("\n" + "="*80)
print("COURBES ROC (Receiver Operating Characteristic)")
print("="*80)

plt.figure(figsize=(10, 8))
for nom, data in courbes_roc.items():
    if data is not None:
        fpr, tpr, roc_auc = data
        plt.plot(fpr, tpr, lw=2, label=f'{nom} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aléatoire (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
plt.title('Courbes ROC - Tous les modèles', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/comparaison/modeles.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Graphique sauvegardé : modeles.png")

for nom, data in courbes_roc.items():
    if data is not None:
        fpr, tpr, roc_auc = data
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aléatoire (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
        plt.title(f'Courbe ROC - {nom}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'static/images/comparaison/roc/{nom.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Courbe ROC sauvegardée : roc/{nom.replace(' ', '_')}.png")
        numero += 1


print("\n" + "="*80)
print("COURBES PRECISION-RECALL")
print("="*80)

plt.figure(figsize=(10, 8))
for nom, data in courbes_pr.items():
    if data is not None:
        recall_curve, precision_curve, avg_precision = data
        plt.plot(recall_curve, precision_curve, lw=2, 
                label=f'{nom} (AP = {avg_precision:.3f})')

plt.xlabel('Rappel (Recall)', fontsize=12)
plt.ylabel('Précision', fontsize=12)
plt.title('Courbes Précision-Rappel - Tous les modèles', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=9)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('static/images/comparaison/precision_recall_modeles.png', dpi=300, bbox_inches='tight')
plt.close()

for nom, data in courbes_pr.items():
    if data is not None:
        recall_curve, precision_curve, avg_precision = data
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                label=f'PR (AP = {avg_precision:.3f})')
        plt.xlabel('Rappel (Recall)', fontsize=12)
        plt.ylabel('Précision', fontsize=12)
        plt.title(f'Courbe Précision-Rappel - {nom}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(f'static/images/comparaison/precision_recall/{nom.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Courbe PR sauvegardée : precision_recall/{nom.replace(' ', '_')}.png")

meilleur_modele_nom = df_resultats.iloc[0]['Modèle']
meilleur_f1 = df_resultats.iloc[0]['F1-Score']
meilleur_roc_auc = df_resultats.iloc[0]['ROC-AUC']

print(f"\n{'='*80}")
print(f" MEILLEUR MODÈLE : {meilleur_modele_nom}")
print(f"   F1-Score  : {meilleur_f1:.4f} ({meilleur_f1*100:.2f}%)")
print(f"   ROC-AUC   : {meilleur_roc_auc:.4f}" if not np.isnan(meilleur_roc_auc) else "   ROC-AUC   : N/A")
print(f"{'='*80}")


print("\n" + "="*80)
print("question 9 : Sauvegarde du meilleur modèle")
print("="*80)

meilleur_modele = modeles_entraines[meilleur_modele_nom]

joblib.dump(meilleur_modele, 'ml/model.pkl')
joblib.dump(scaler, 'ml/scaler.pkl')

train_stats = {
    'percentile_99': df['TransactionAmount'].quantile(0.99),
    'mean': df['TransactionAmount'].mean(),
    'std': df['TransactionAmount'].std()
}
joblib.dump(train_stats, 'ml/train_stats.pkl')

modele_charge = joblib.load('ml/model.pkl')

print("\nUtilisation du modèle sauvegardé :")
print("  model = joblib.load('ml/model.pkl')")
print("  scaler = joblib.load('ml/scaler.pkl')")
print("  train_stats = joblib.load('ml/train_stats.pkl')")






