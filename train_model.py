import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



print("Random Forest")

#Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques (features) que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Créer et entraîner le modèle de forêt aléatoire
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

print(f"Accuracy : {accuracy}")
print(f"\nMatrice de confusion :\n{conf_matrix}")
print(f"\nClassification Report :\n{classification_report_str}")

print("#####################################################""")
print("modèle de machine à vecteurs de support (SVM) ")
# Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques (features) que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Normalisation des données (important pour les SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle SVM
model_svm = SVC(random_state=42)
model_svm.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
predictions_svm = model_svm.predict(X_test_scaled)

# Évaluer la performance du modèle SVM
accuracy_svm = accuracy_score(y_test, predictions_svm)
conf_matrix_svm = confusion_matrix(y_test, predictions_svm)
classification_report_str_svm = classification_report(y_test, predictions_svm)

print(f"Accuracy (SVM) : {accuracy_svm}")
print(f"\nMatrice de confusion (SVM) :\n{conf_matrix_svm}")
print(f"\nClassification Report (SVM) :\n{classification_report_str_svm}")
# Matrice de confusion sur un graphique
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matrice de Confusion (SVM)")
plt.xlabel("Prédiction")
plt.ylabel("Vraie Valeur")
plt.show()

# Courbe ROC avec l'AUC
fpr, tpr, thresholds = roc_curve(y_test, model_svm.decision_function(X_test_scaled))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Référence aléatoire')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - SVM')
plt.legend(loc='lower right')
plt.show()

#Dans l'évaluation du modèle de machine à vecteurs de support (SVM) appliqué aux données du Titanic, l'accuracy obtenue est d'environ 81%. La matrice de confusion révèle que le modèle a correctement identifié la majorité des passagers non survivants (True Negatives: 77), mais a eu des difficultés à capturer tous les cas de survie (True Positives: 36). Cette observation se reflète dans le rapport de classification, où la précision pour la classe "Survie" est élevée (88%), suggérant que le modèle est précis lorsqu'il prédit la survie. Cependant, le rappel associé à cette classe est relativement plus faible (63%), indiquant que le modèle peut manquer certains cas réels de survie. Globalement, le modèle présente un équilibre entre la précision et le rappel, avec un F1-score moyen de 0.80, témoignant d'une performance modérée.





import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques (features) que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Normalisation des données (important pour certains algorithmes, y compris la régression logistique)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle de régression logistique
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
predictions_logistic = model_logistic.predict(X_test_scaled)

# Évaluer la performance du modèle de régression logistique
accuracy_logistic = accuracy_score(y_test, predictions_logistic)
conf_matrix_logistic = confusion_matrix(y_test, predictions_logistic)
classification_report_str_logistic = classification_report(y_test, predictions_logistic)

print(f"Accuracy (Régression Logistique) : {accuracy_logistic}")
print(f"\nMatrice de confusion (Régression Logistique) :\n{conf_matrix_logistic}")
print(f"\nClassification Report (Régression Logistique) :\n{classification_report_str_logistic}")
# Matrice de confusion sur un graphique
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logistic, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matrice de Confusion (Régression Logistique)")
plt.xlabel("Prédiction")
plt.ylabel("Vraie Valeur")
plt.show()

# Courbe ROC avec l'AUC
fpr, tpr, thresholds = roc_curve(y_test, model_logistic.decision_function(X_test_scaled))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Référence aléatoire')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - Régression Logistique')
plt.legend(loc='lower right')
plt.show()

#Dans l'évaluation du modèle de régression logistique appliqué aux données du Titanic, l'accuracy obtenue est d'environ 82%. La matrice de confusion indique que le modèle a correctement classifié la majorité des passagers non survivants (True Negatives: 74) et a également bien identifié les cas de survie (True Positives: 40). Cependant, il a également produit quelques faux positifs (Predicted Survived, Actual Non-Survived: 8) et faux négatifs (Predicted Non-Survived, Actual Survived: 17).

# Le rapport de classification fournit une analyse plus détaillée de la performance du modèle. La précision pour la classe "Non-Survie" est de 81%, indiquant que parmi les prédictions positives pour cette classe, 81% étaient correctes. Pour la classe "Survie", la précision est de 83%, montrant une précision élevée dans les prédictions positives pour cette classe.

# Le rappel (recall) est de 90% pour la classe "Non-Survie", indiquant que le modèle a capturé une grande proportion des cas réels de non-survie. Cependant, pour la classe "Survie", le rappel est de 70%, suggérant que le modèle a manqué certains cas réels de survie.

# Le F1-score, qui prend en compte la précision et le rappel, est de 0.86 pour la classe "Non-Survie" et de 0.76 pour la classe "Survie". La moyenne pondérée des F1-scores donne un aperçu global de la performance du modèle, avec une valeur de 0.82.

# En conclusion, le modèle de régression logistique présente une performance globale solide avec une accuracy de 82%

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques (features) que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Normalisation des données (important pour les algorithmes basés sur la distance, y compris KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle KNN
k_neighbors = 5  # Vous pouvez ajuster ce nombre selon votre préférence
model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
model_knn.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
predictions_knn = model_knn.predict(X_test_scaled)

# Évaluer la performance du modèle KNN
accuracy_knn = accuracy_score(y_test, predictions_knn)
conf_matrix_knn = confusion_matrix(y_test, predictions_knn)
classification_report_str_knn = classification_report(y_test, predictions_knn)

print(f"Accuracy (KNN) : {accuracy_knn}")
print(f"\nMatrice de confusion (KNN) :\n{conf_matrix_knn}")
print(f"\nClassification Report (KNN) :\n{classification_report_str_knn}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculer les probabilités de classe 1 pour le modèle KNN
y_scores_knn = model_knn.predict_proba(X_test_scaled)[:, 1]

# Calculer la courbe ROC et l'AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores_knn)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC avec des annotations pour l'AUC
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.fill_between(fpr, 0, tpr, alpha=0.2, color='darkorange', label='AUC Confidence Interval')

# Ajouter des annotations
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matrice de Confusion (KNN)")
plt.xlabel("Prédiction")
plt.ylabel("Vraie Valeur")
plt.show()

#Dans l'évaluation du modèle KNN (k plus proches voisins) appliqué aux données du Titanic, l'accuracy obtenue est d'environ 81%. La matrice de confusion révèle que le modèle a correctement classifié la majorité des passagers non survivants (True Negatives: 73) et a également bien identifié les cas de survie (True Positives: 40). Cependant, il a également produit quelques faux positifs (Predicted Survived, Actual Non-Survived: 9) et faux négatifs (Predicted Non-Survived, Actual Survived: 17).

# Le rapport de classification fournit une analyse plus détaillée de la performance du modèle. La précision pour la classe "Non-Survie" est de 81%, indiquant que parmi les prédictions positives pour cette classe, 81% étaient correctes. Pour la classe "Survie", la précision est de 82%, montrant une précision relativement élevée dans les prédictions positives pour cette classe.

# Le rappel (recall) est de 89% pour la classe "Non-Survie" et de 70% pour la classe "Survie". Cela suggère que le modèle a bien identifié une grande proportion des cas réels de non-survie, mais a manqué certains cas de survie.

# Le F1-score, qui prend en compte la précision et le rappel, est de 0.85 pour la classe "Non-Survie" et de 0.75 pour la classe "Survie". La moyenne pondérée des F1-scores donne un aperçu global de la performance du modèle, avec une valeur de 0.81.

# Globalement, le modèle KNN présente une performance solide avec une accuracy de 81%. 
print("################")

from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à ajuster
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Créer le modèle KNN
model_knn = KNeighborsClassifier()

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(model_knn, param_grid, cv=5, scoring='accuracy')

# Exécuter la recherche en grille sur les données d'entraînement
grid_search.fit(X_train_scaled, y_train)

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres :", grid_search.best_params_)

# Utiliser le modèle avec les meilleurs hyperparamètres
best_model_knn = grid_search.best_estimator_

# Prédictions sur l'ensemble de test avec le meilleur modèle
predictions_best_knn = best_model_knn.predict(X_test_scaled)

# Évaluer la performance du meilleur modèle
accuracy_best_knn = accuracy_score(y_test, predictions_best_knn)
conf_matrix_best_knn = confusion_matrix(y_test, predictions_best_knn)
classification_report_str_best_knn = classification_report(y_test, predictions_best_knn)

print(f"Accuracy (Meilleur modèle KNN) : {accuracy_best_knn}")
print(f"\nMatrice de confusion (Meilleur modèle KNN) :\n{conf_matrix_best_knn}")
print(f"\nClassification Report (Meilleur modèle KNN) :\n{classification_report_str_best_knn}")



import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Normalisation des données (important pour certains algorithmes, y compris RandomForest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer le modèle RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Entraîner le modèle sur les données d'entraînement
model_rf.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
predictions_rf = model_rf.predict(X_test_scaled)

# Évaluer la performance du modèle RandomForest
accuracy_rf = accuracy_score(y_test, predictions_rf)
conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
classification_report_str_rf = classification_report(y_test, predictions_rf)

# Afficher la matrice de confusion sous forme d'image avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit 0', 'Prédit 1'], yticklabels=['Réel 0', 'Réel 1'])
plt.title('Matrice de Confusion - Random Forest')
plt.xlabel('Prédictions')
plt.ylabel('Réelles')
plt.show()

print(f"Précision (Random Forest) : {accuracy_rf}")
print(f"\nMatrice de confusion (Random Forest) :\n{conf_matrix_rf}")
print(f"\nRapport de Classification (Random Forest) :\n{classification_report_str_rf}")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données du Titanic
titanic_data = sns.load_dataset('titanic')

# Diviser les données en ensembles d'entraînement et de test avec la même proportion de survie
train_data, test_data = train_test_split(titanic_data, test_size=0.2, stratify=titanic_data['survived'], random_state=42)

# Sélectionner les caractéristiques que vous souhaitez utiliser pour la prédiction
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# Prétraitement des données (remplacement des valeurs manquantes, encodage des variables catégorielles)
train_data = train_data[features + ['survived']].dropna()
test_data = test_data[features + ['survived']].dropna()

# Encodage des variables catégorielles
train_data_encoded = pd.get_dummies(train_data, columns=['sex', 'embarked'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['sex', 'embarked'], drop_first=True)

# Séparer les variables indépendantes (X) et la variable cible (y)
X_train = train_data_encoded.drop('survived', axis=1)
y_train = train_data_encoded['survived']

X_test = test_data_encoded.drop('survived', axis=1)
y_test = test_data_encoded['survived']

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer le modèle RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Définir les hyperparamètres à ajuster
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(model_rf, param_grid, cv=5, scoring='accuracy')

# Exécuter la recherche en grille sur les données d'entraînement
grid_search.fit(X_train_scaled, y_train)

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres :", grid_search.best_params_)

# Utiliser le modèle avec les meilleurs hyperparamètres
best_model_rf = grid_search.best_estimator_

# Prédictions sur l'ensemble de test avec le meilleur modèle
predictions_best_rf = best_model_rf.predict(X_test_scaled)

# Évaluer la performance du meilleur modèle
accuracy_best_rf = accuracy_score(y_test, predictions_best_rf)
conf_matrix_best_rf = confusion_matrix(y_test, predictions_best_rf)
classification_report_str_best_rf = classification_report(y_test, predictions_best_rf)

# Afficher la matrice de confusion sous forme d'image avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit 0', 'Prédit 1'], yticklabels=['Réel 0', 'Réel 1'])
plt.title('Matrice de Confusion - Random Forest (Meilleur modèle)')
plt.xlabel('Prédictions')
plt.ylabel('Réelles')
plt.show()

print(f"Accuracy (Random Forest - Meilleur modèle) : {accuracy_best_rf}")
print(f"\nMatrice de confusion (Random Forest - Meilleur modèle) :\n{conf_matrix_best_rf}")
print(f"\nRapport de Classification (Random Forest - Meilleur modèle) :\n{classification_report_str_best_rf}")


