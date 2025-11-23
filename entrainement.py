# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc, 
                             precision_recall_curve, average_precision_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Modèles de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Pour la gestion du déséquilibre
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Pour sauvegarder le modèle
import joblib

print("="*80)
print("DÉTECTION DE FRAUDE BANCAIRE - MACHINE LEARNING")
print("="*80)

# ============================================================================
# TÂCHE 1 : CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[TÂCHE 1] Chargement des données...")

df = pd.read_csv('./static/data/creditcarddata.csv')
print(f"✓ Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nAperçu des données :")
print(df.head())
print("\nInformations sur les données :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())

# ============================================================================
# TÂCHE 2 : LISTE DES MODÈLES POUR RÉSOUDRE LE PROBLÈME
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 2] Modèles de classification pour la détection de fraude")
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

# ============================================================================
# TÂCHE 3 : PRÉPARATION DES DONNÉES
# ============================================================================
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
#plt.savefig('03_distribution_originale.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 03_distribution_originale.png")

# Graphique 2 : Distribution log-transformée
plt.figure(figsize=(10, 6))
plt.hist(df['TransactionAmount_log'], bins=50, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution log-transformée de TransactionAmount', fontsize=14, fontweight='bold')
plt.xlabel('log(Montant + 1)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.savefig('04_distribution_log.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 04_distribution_log.png")

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
#plt.savefig('05_distribution_zscore.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 05_distribution_zscore.png")

print("\n✓ Traitement des valeurs aberrantes terminé")
print(f"  Dimensions finales : {df.shape}")

# 3.4 Analyse du déséquilibre de classes
print("\n3.4 Analyse du déséquilibre de classes :")
distribution_classe = df['PotentialFraud'].value_counts()
print(distribution_classe)

taux_fraude = distribution_classe[1]/len(df)*100
taux_non_fraude = distribution_classe[0]/len(df)*100

print(f"\nProportion de fraudes : {taux_fraude:.2f}%")
print(f"Proportion de non-fraudes : {taux_non_fraude:.2f}%")
print(f"Ratio : 1 fraude pour {distribution_classe[0]/distribution_classe[1]:.1f} non-fraudes")

# Visualisation du déséquilibre - Graphique 1 : Bar plot
plt.figure(figsize=(8, 6))
distribution_classe.plot(kind='bar', color=['green', 'red'])
plt.title('Distribution des classes (AVANT rééquilibrage)', fontsize=14, fontweight='bold')
plt.xlabel('PotentialFraud', fontsize=12)
plt.ylabel('Nombre d\'observations', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
#plt.savefig('01_distribution_classes_barplot_AVANT.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 01_distribution_classes_barplot_AVANT.png")

# Visualisation du déséquilibre - Graphique 2 : Pie chart
plt.figure(figsize=(8, 8))
plt.pie(distribution_classe, labels=['Non-Fraude', 'Fraude'], autopct='%1.1f%%', 
        colors=['green', 'red'], startangle=90, textprops={'fontsize': 12},
        explode=(0, 0.1))
plt.title('Proportion des classes (AVANT rééquilibrage)', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig('02_distribution_classes_piechart_AVANT.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 02_distribution_classes_piechart_AVANT.png")

# Détection du déséquilibre
if taux_fraude < 30:
    print(f"\n⚠️ DÉSÉQUILIBRE IMPORTANT DÉTECTÉ : Seulement {taux_fraude:.2f}% de fraudes")
    print("   → Le rééquilibrage sera appliqué après la division train/test")
    desequilibre = True
else:
    desequilibre = False

# ============================================================================
# TÂCHE 4 : DIVISION DES DONNÉES (70% train, 30% test)
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 4] Division des données")
print("="*80)

# Séparation des features et de la cible
X = df.drop('PotentialFraud', axis=1)
y = df['PotentialFraud']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✓ Données d'entraînement : {X_train.shape[0]} observations ({70}%)")
print(f"✓ Données de test : {X_test.shape[0]} observations ({30}%)")
print(f"\nDistribution dans le train AVANT rééquilibrage :")
print(y_train.value_counts())
print(f"\nDistribution dans le test :")
print(y_test.value_counts())

# ============================================================================
# RÉÉQUILIBRAGE DU DATASET (CRITIQUE POUR LA DÉTECTION DE FRAUDE)
# ============================================================================
print("\n" + "="*80)
print("RÉÉQUILIBRAGE DU DATASET")
print("="*80)

if desequilibre:
    print("\n⚠️ Application du rééquilibrage sur les données d'entraînement...")
    print("   (Les données de test ne sont PAS modifiées pour une évaluation réaliste)")
    
    # Méthode 1: SMOTE (Synthetic Minority Over-sampling Technique)
    print("\n📊 Méthode utilisée : SMOTE + Tomek Links (SMOTETomek)")
    print("   - SMOTE : Crée des échantillons synthétiques de la classe minoritaire")
    print("   - Tomek Links : Supprime les échantillons ambigus à la frontière")
    
    # Appliquer SMOTETomek (combinaison de sur-échantillonnage et nettoyage)
    smote_tomek = SMOTETomek(
        smote=SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5),
        random_state=42
    )
    
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    print(f"\n✅ Rééquilibrage terminé !")
    print(f"\n   AVANT rééquilibrage : {len(X_train)} échantillons")
    print(f"   APRÈS rééquilibrage : {len(X_train_resampled)} échantillons")
    
    print(f"\n   Distribution AVANT :")
    print(f"      Non-fraude : {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"      Fraude     : {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    
    print(f"\n   Distribution APRÈS :")
    print(f"      Non-fraude : {(y_train_resampled == 0).sum()} ({(y_train_resampled == 0).sum()/len(y_train_resampled)*100:.1f}%)")
    print(f"      Fraude     : {(y_train_resampled == 1).sum()} ({(y_train_resampled == 1).sum()/len(y_train_resampled)*100:.1f}%)")
    
    # Utiliser les données rééquilibrées pour l'entraînement
    X_train = X_train_resampled
    y_train = y_train_resampled
    
    # Visualisation APRÈS rééquilibrage
    plt.figure(figsize=(8, 6))
    y_train.value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Distribution des classes (APRÈS rééquilibrage)', fontsize=14, fontweight='bold')
    plt.xlabel('PotentialFraud', fontsize=12)
    plt.ylabel('Nombre d\'observations', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    #plt.savefig('01b_distribution_classes_barplot_APRES.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Graphique sauvegardé : 01b_distribution_classes_barplot_APRES.png")
    
    # Comparaison avant/après
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Avant
    distribution_classe.plot(kind='bar', color=['green', 'red'], ax=axes[0])
    axes[0].set_title('AVANT rééquilibrage', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('PotentialFraud')
    axes[0].set_ylabel('Nombre')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Après
    y_train.value_counts().plot(kind='bar', color=['green', 'red'], ax=axes[1])
    axes[1].set_title('APRÈS rééquilibrage (Train)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('PotentialFraud')
    axes[1].set_ylabel('Nombre')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.suptitle('Impact du rééquilibrage SMOTETomek', fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig('01c_comparaison_reequilibrage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Graphique sauvegardé : 01c_comparaison_reequilibrage.png")

else:
    print("\n✅ Dataset suffisamment équilibré, pas de rééquilibrage nécessaire")

# ============================================================================
# NORMALISATION DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("NORMALISATION DES DONNÉES")
print("="*80)

print("\nPlusieurs méthodes de normalisation disponibles :")
print("  1. StandardScaler : (X - mean) / std → Distribution N(0,1)")
print("  2. MinMaxScaler : (X - min) / (max - min) → Échelle [0,1]")
print("  3. RobustScaler : Utilise la médiane et IQR → Robuste aux outliers")

# Afficher les statistiques avant normalisation
print("\nStatistiques AVANT normalisation (échantillon) :")
print(X_train[['Age', 'TransactionAmount', 'TransactionAmount_log']].describe())

# Méthode 1 : StandardScaler (recommandé pour la plupart des modèles)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

print("\n→ Application de StandardScaler (méthode principale)")
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)
print("  ✓ StandardScaler appliqué")

# Méthode 2 : RobustScaler (pour données avec outliers)
print("\n→ Application de RobustScaler (robuste aux outliers)")
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)
print("  ✓ RobustScaler appliqué")

# Méthode 3 : MinMaxScaler (pour réseaux de neurones)
print("\n→ Application de MinMaxScaler (échelle 0-1)")
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)
print("  ✓ MinMaxScaler appliqué")

# Utiliser StandardScaler par défaut
X_train_scaled = X_train_standard
X_test_scaled = X_test_standard
scaler = scaler_standard

print("\n✓ Normalisation principale : StandardScaler sélectionné")

# Afficher les statistiques après normalisation
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, 
    columns=X_train.columns
)
print("\nStatistiques APRÈS normalisation (échantillon) :")
print(X_train_scaled_df[['Age', 'TransactionAmount', 'TransactionAmount_log']].describe())

# Visualisation de l'effet de la normalisation
colonnes_viz = ['Age', 'TransactionAmount', 'TransactionAmount_log']

for idx, col in enumerate(colonnes_viz):
    # Graphique AVANT normalisation
    plt.figure(figsize=(10, 6))
    plt.hist(X_train[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.title(f'{col} - AVANT normalisation', fontsize=14, fontweight='bold')
    plt.xlabel('Valeur', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.savefig(f'06_normalisation_avant_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : 06_normalisation_avant_{col}.png")
    
    # Graphique APRÈS normalisation
    plt.figure(figsize=(10, 6))
    col_index = list(X_train.columns).index(col)
    plt.hist(X_train_scaled[:, col_index], bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.title(f'{col} - APRÈS normalisation (StandardScaler)', fontsize=14, fontweight='bold')
    plt.xlabel('Valeur normalisée', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.savefig(f'07_normalisation_apres_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : 07_normalisation_apres_{col}.png")

print("\n" + "="*80)

# ============================================================================
# TÂCHE 5 : CRÉATION ET ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 5] Entraînement des modèles")
print("="*80)

# Dictionnaire des modèles (avec class_weight pour gérer le déséquilibre)
print("\n💡 Note: Les modèles utilisent class_weight='balanced' pour mieux détecter les fraudes")

modeles = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight='balanced'  # Pénalise plus les erreurs sur la classe minoritaire
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, 
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42, 
        n_estimators=100, 
        class_weight='balanced'  # Important pour la fraude
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42, 
        n_estimators=100
        # Note: GradientBoosting n'a pas class_weight, mais SMOTE compense
    ),
    'SVM': SVC(
        random_state=42, 
        kernel='rbf', 
        class_weight='balanced',
        probability=True  # Nécessaire pour predict_proba
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5
        # Note: KNN n'a pas class_weight, mais SMOTE compense
    ),
    'Naive Bayes': GaussianNB()
    # Note: GaussianNB n'a pas class_weight, mais SMOTE compense
}

# Entraînement des modèles
modeles_entraines = {}
for nom, modele in modeles.items():
    print(f"\n→ Entraînement de {nom}...")
    modele.fit(X_train_scaled, y_train)
    modeles_entraines[nom] = modele
    print(f"  ✓ {nom} entraîné avec succès")

# ============================================================================
# TÂCHE 6 : ÉVALUATION DES MODÈLES (Matrices de confusion)
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 6] Évaluation des modèles - Matrices de confusion")
print("="*80)

# Création des matrices de confusion individuelles
for idx, (nom, modele) in enumerate(modeles_entraines.items()):
    y_pred = modele.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Créer une figure pour chaque matrice
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraude', 'Fraude'],
                yticklabels=['Non-Fraude', 'Fraude'],
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title(f'Matrice de confusion - {nom}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    
    # Sauvegarder avec un nom numéroté
    numero = 8 + idx  # Commence à 08 après les graphiques de normalisation
    nom_fichier = f'{numero:02d}_matrice_confusion_{nom.replace(" ", "_")}.png'
    #plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Matrice de confusion sauvegardée : {nom_fichier}")
    
    # Analyse de la matrice
    tn, fp, fn, tp = cm.ravel()
    print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ============================================================================
# TÂCHE 7 : CALCUL DES MÉTRIQUES (Accuracy, Précision, Rappel, F1, ROC-AUC)
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 7] Calcul des métriques de performance")
print("="*80)

resultats = []
courbes_roc = {}
courbes_pr = {}

for nom, modele in modeles_entraines.items():
    y_pred = modele.predict(X_test_scaled)
    
    # Calculer les probabilités si le modèle le supporte
    if hasattr(modele, 'predict_proba'):
        y_pred_proba = modele.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(modele, 'decision_function'):
        y_pred_proba = modele.decision_function(X_test_scaled)
    else:
        y_pred_proba = y_pred  # Fallback
    
    # Calcul des métriques de base
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calcul du ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        # Calculer la courbe ROC pour visualisation
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        courbes_roc[nom] = (fpr, tpr, roc_auc)
    except:
        roc_auc = np.nan
        courbes_roc[nom] = None
    
    # Calcul de l'Average Precision (pour courbe Precision-Recall)
    try:
        avg_precision = average_precision_score(y_test, y_pred_proba)
        # Calculer la courbe Precision-Recall
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

# ============================================================================
# TÂCHE 8 : COMPARAISON DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 8] Comparaison des modèles")
print("="*80)

df_resultats = pd.DataFrame(resultats)

# Pour la détection de fraude, on trie par RAPPEL (détecter un max de fraudes)
# puis par F1-Score pour l'équilibre
df_resultats = df_resultats.sort_values(['Rappel', 'F1-Score'], ascending=[False, False])

print("\nTableau comparatif des performances (trié par Rappel) :")
print(df_resultats.to_string(index=False))

print("\n💡 Pour la détection de FRAUDE :")
print("   - RAPPEL élevé = Détecte un maximum de vraies fraudes (priorité #1)")
print("   - PRÉCISION élevée = Peu de fausses alertes")
print("   - F1-Score = Équilibre entre les deux")

# Visualisation comparative - Graphiques individuels
metriques = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'ROC-AUC']
numero_base = 15

for idx, metrique in enumerate(metriques):
    plt.figure(figsize=(12, 6))
    
    # Filtrer les valeurs NaN pour ROC-AUC
    df_plot = df_resultats[['Modèle', metrique]].dropna()
    df_plot_sorted = df_plot.sort_values(metrique, ascending=True)
    
    bars = plt.barh(df_plot_sorted['Modèle'], df_plot_sorted[metrique], color='steelblue')
    plt.title(f'Comparaison des modèles - {metrique}', fontsize=14, fontweight='bold')
    plt.xlabel(metrique, fontsize=12)
    plt.ylabel('Modèle', fontsize=12)
    plt.xlim([0, 1.1])
    plt.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, df_plot_sorted[metrique])):
        plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    numero = numero_base + idx
    #plt.savefig(f'{numero:02d}_comparaison_{metrique.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : {numero:02d}_comparaison_{metrique.lower().replace('-', '_')}.png")

# ============================================================================
# COURBES ROC (Receiver Operating Characteristic)
# ============================================================================
print("\n" + "="*80)
print("COURBES ROC (Receiver Operating Characteristic)")
print("="*80)

# Graphique combiné de toutes les courbes ROC
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
#plt.savefig('20_courbes_roc_tous_modeles.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 20_courbes_roc_tous_modeles.png")

# Courbes ROC individuelles
numero = 21
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
        #plt.savefig(f'{numero:02d}_courbe_roc_{nom.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Courbe ROC sauvegardée : {numero:02d}_courbe_roc_{nom.replace(' ', '_')}.png")
        numero += 1

# ============================================================================
# COURBES PRECISION-RECALL
# ============================================================================
print("\n" + "="*80)
print("COURBES PRECISION-RECALL")
print("="*80)

# Graphique combiné de toutes les courbes Precision-Recall
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
#plt.savefig('28_courbes_precision_recall_tous_modeles.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graphique sauvegardé : 28_courbes_precision_recall_tous_modeles.png")

# Courbes Precision-Recall individuelles
numero = 29
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
        #plt.savefig(f'{numero:02d}_courbe_pr_{nom.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Courbe PR sauvegardée : {numero:02d}_courbe_pr_{nom.replace(' ', '_')}.png")
        numero += 1

# Identification du meilleur modèle (basé sur le Rappel pour la fraude)
meilleur_modele_nom = df_resultats.iloc[0]['Modèle']
meilleur_rappel = df_resultats.iloc[0]['Rappel']
meilleur_f1 = df_resultats.iloc[0]['F1-Score']
meilleur_roc_auc = df_resultats.iloc[0]['ROC-AUC']

print(f"\n{'='*80}")
print(f"🏆 MEILLEUR MODÈLE POUR LA DÉTECTION DE FRAUDE : {meilleur_modele_nom}")
print(f"   Rappel (Recall) : {meilleur_rappel:.4f} ({meilleur_rappel*100:.2f}%) - Détecte {meilleur_rappel*100:.0f}% des fraudes")
print(f"   F1-Score        : {meilleur_f1:.4f} ({meilleur_f1*100:.2f}%)")
print(f"   ROC-AUC         : {meilleur_roc_auc:.4f}" if not np.isnan(meilleur_roc_auc) else "   ROC-AUC         : N/A")
print(f"{'='*80}")

# ============================================================================
# TÂCHE 9 : SAUVEGARDE DU MEILLEUR MODÈLE
# ============================================================================
print("\n" + "="*80)
print("[TÂCHE 9] Sauvegarde du meilleur modèle")
print("="*80)

meilleur_modele = modeles_entraines[meilleur_modele_nom]

# Sauvegarder le modèle, le scaler et les statistiques d'entraînement
joblib.dump(meilleur_modele, 'ml/model.pkl')
joblib.dump(scaler, 'ml/scaler.pkl')

# Sauvegarder les statistiques pour garantir la cohérence lors des prédictions
train_stats = {
    'percentile_99': df['TransactionAmount'].quantile(0.99),
    'mean': df['TransactionAmount'].mean(),
    'std': df['TransactionAmount'].std()
}
joblib.dump(train_stats, 'ml/train_stats.pkl')

print(f"✓ Modèle sauvegardé : ml/model.pkl")
print(f"✓ Scaler sauvegardé : ml/scaler.pkl")
print(f"✓ Statistiques sauvegardées : ml/train_stats.pkl")
# Test de chargement
modele_charge = joblib.load('ml/model.pkl')
print(f"✓ Test de chargement réussi")

print("\n" + "="*80)
print("TRAITEMENT TERMINÉ AVEC SUCCÈS !")
print("="*80)
print("\nFichiers générés :")
print("\n📊 GRAPHIQUES DE DISTRIBUTION :")
print("  01. distribution_classes_barplot.png - Distribution en barres")
print("  02. distribution_classes_piechart.png - Distribution en camembert")

print("\n🔍 GRAPHIQUES D'ANALYSE DES OUTLIERS :")
print("  03. distribution_originale.png - Distribution originale de TransactionAmount")
print("  04. distribution_log.png - Distribution log-transformée")
print("  05. distribution_zscore.png - Distribution des Z-scores")

print("\n⚖️ GRAPHIQUES DE NORMALISATION (Avant/Après) :")
print("  06. normalisation_avant_Age.png")
print("  07. normalisation_apres_Age.png")
print("  06. normalisation_avant_TransactionAmount.png")
print("  07. normalisation_apres_TransactionAmount.png")
print("  06. normalisation_avant_TransactionAmount_log.png")
print("  07. normalisation_apres_TransactionAmount_log.png")

print("\n📈 MATRICES DE CONFUSION (par modèle) :")
print("  08. matrice_confusion_Logistic_Regression.png")
print("  09. matrice_confusion_Decision_Tree.png")
print("  10. matrice_confusion_Random_Forest.png")
print("  11. matrice_confusion_Gradient_Boosting.png")
print("  12. matrice_confusion_SVM.png")
print("  13. matrice_confusion_KNN.png")
print("  14. matrice_confusion_Naive_Bayes.png")

print("\n📊 COMPARAISONS DES MODÈLES (Métriques) :")
print("  15. comparaison_accuracy.png")
print("  16. comparaison_précision.png")
print("  17. comparaison_rappel.png")
print("  18. comparaison_f1_score.png")
print("  19. comparaison_roc_auc.png")

print("\n📈 COURBES ROC :")
print("  20. courbes_roc_tous_modeles.png - Toutes les courbes ROC")
print("  21-27. courbe_roc_[nom_modele].png - Courbes ROC individuelles")

print("\n📉 COURBES PRÉCISION-RAPPEL :")
print("  28. courbes_precision_recall_tous_modeles.png - Toutes les courbes PR")
print("  29-35. courbe_pr_[nom_modele].png - Courbes PR individuelles")

print("\n💾 FICHIERS DU MODÈLE :")
print("  • meilleur_modele_fraud_detection.pkl - Meilleur modèle entraîné")
print("  • scaler.pkl - Scaler pour la normalisation")
print("  • train_stats.pkl - Statistiques d'entraînement pour prédictions")

print("\n📋 RÉSUMÉ DES MÉTRIQUES :")
print("  ✓ Accuracy    : Précision globale du modèle")
print("  ✓ Précision   : Taux de vrais positifs parmi les prédictions positives")
print("  ✓ Rappel      : Taux de fraudes détectées (sensibilité)")
print("  ✓ F1-Score    : Moyenne harmonique entre précision et rappel")
print("  ✓ ROC-AUC     : Aire sous la courbe ROC (capacité de discrimination)")
print("  ✓ Avg Prec    : Average Precision (qualité de la courbe PR)")

print("\n💡 INTERPRÉTATION :")
print("  → Pour la détection de fraude, privilégier:")
print("     • F1-Score élevé (équilibre précision/rappel)")
print("     • Rappel élevé (détecter un maximum de fraudes)")
print("     • ROC-AUC élevé (bonne séparation fraude/non-fraude)")
print("\nUtilisation du modèle sauvegardé :")
print("  model = joblib.load('meilleur_modele_fraud_detection.pkl')")
print("  scaler = joblib.load('scaler.pkl')")
print("  train_stats = joblib.load('train_stats.pkl')")
print("  # Voir le script 'prediction_script.py' pour des exemples complets")