# Prédicteur d’Attrition Client (Customer Churn) – Checklist Complète

## Étape 1 : Analyse Exploratoire, Nettoyage et Préparation des Données

- [ ] Créer le dossier du projet et l'environnement virtuel (`venv`)
- [ ] Télécharger le dataset **WA_Fn-UseC_-Telco-Customer-Churn.csv**
- [ ] Charger les données avec pandas (`pd.read_csv`)
- [ ] Vérifier les informations de base : `shape`, `info()`, `describe()`, `head()`
- [ ] Analyser les valeurs manquantes et les types de données
- [ ] Convertir `TotalCharges` en type numérique (`pd.to_numeric`, `errors='coerce'`)
- [ ] Traiter les valeurs manquantes dans `TotalCharges` (remplacer par 0 ou la médiane)
- [ ] Convertir la cible `Churn` en binaire (`Yes` → 1, `No` → 0)
- [ ] Supprimer la colonne inutile (`customerID`)
- [ ] **Analyse Exploratoire (EDA)** :
  - Calculer le taux global de churn
  - Analyse univariée (histogrammes, countplots) pour toutes les variables
  - Analyse bivariée (churn vs chaque feature : Contract, Tenure, MonthlyCharges, InternetService, PaymentMethod, SeniorCitizen, etc.)
  - Heatmap de corrélation des variables numériques
  - Créer de nouvelles features (ex. : `TenureGroup`, `TotalServices`, `ChargePerMonth`, etc.)
- [ ] Encoder les variables catégorielles (One-Hot Encoding avec `pd.get_dummies` ou `ColumnTransformer`)
- [ ] Séparer les features (`X`) et la cible (`y`)
- [ ] Diviser les données : Train/Test (80/20, `stratify=y`, `random_state=42`)
- [ ] Sauvegarder le dataset nettoyé et les features préparées

**Notebook recommandé** : `01_eda_preprocessing.ipynb`

---

## Étape 2 : Entraînement et Comparaison des Modèles

- [ ] Créer le notebook `02_modeling.ipynb`
- [ ] Préparer les données (`X_train`, `X_test`, `y_train`, `y_test`)
- [ ] Entraîner 3 modèles avec paramètres par défaut :
  - Random Forest
  - LightGBM
  - XGBoost
- [ ] Pour chaque modèle :
  - Faire les prédictions sur le test set
  - Calculer les métriques : **AUC-ROC**, Accuracy, Precision, Recall, F1-score, Confusion Matrix
- [ ] Comparer les 3 modèles dans un tableau (pandas DataFrame)
- [ ] Identifier le meilleur modèle (généralement XGBoost)

---

## Étape 3 : Optimisation des Hyperparamètres + Validation Croisée

- [ ] Utiliser `GridSearchCV`, `RandomizedSearchCV` ou **Optuna** sur XGBoost
- [ ] Paramètres à optimiser :
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- [ ] Utiliser `StratifiedKFold` (5 ou 10 folds)
- [ ] Optimiser selon la métrique **AUC-ROC**
- [ ] Entraîner le modèle final avec les meilleurs hyperparamètres
- [ ] Évaluer sur le test set → **Objectif : AUC-ROC ≥ 0.89**
- [ ] Sauvegarder le meilleur modèle (`joblib` ou `mlflow`)

---

## Étape 4 : Suivi des Expérimentations avec MLflow

- [ ] Installer et configurer MLflow
- [ ] Pour chaque expérience (modèles par défaut + modèles optimisés) :
  - Lancer un run MLflow (`mlflow.start_run()`)
  - Logger les paramètres (`mlflow.log_params`)
  - Logger les métriques (`mlflow.log_metrics`)
  - Logger le modèle (`mlflow.xgboost.log_model` ou `mlflow.sklearn`)
  - Logger les artefacts (matrice de confusion, graphiques, etc.)
- [ ] Lancer l’interface MLflow (`mlflow ui`)
- [ ] Enregistrer le meilleur modèle dans le Model Registry

---

## Étape 5 : Interprétation du Modèle avec SHAP

- [ ] Installer SHAP
- [ ] Créer un `TreeExplainer` pour le modèle XGBoost
- [ ] Calculer les valeurs SHAP sur le jeu de test
- [ ] Générer les visualisations suivantes :
  - SHAP Summary Plot (beeswarm)
  - SHAP Feature Importance (bar plot)
  - SHAP Dependence Plots pour les features les plus importantes (ex. tenure, Contract, MonthlyCharges)
  - Force Plot pour quelques clients individuels
- [ ] Rédiger les insights business à partir des résultats SHAP

**Notebook recommandé** : `03_interpretation.ipynb`

---

## Étape 6 : Déploiement sur Azure ML

- [ ] Créer un Workspace Azure ML
- [ ] Connecter le projet à Azure ML (via SDK ou MLflow)
- [ ] Enregistrer le meilleur modèle depuis MLflow vers Azure ML
- [ ] Créer un **Endpoint** (real-time ou batch inference)
- [ ] Créer un scoring script (`score.py`) si nécessaire
- [ ] Déployer le modèle en tant que web service
- [ ] Tester l’endpoint avec des données de test
- [ ] Documenter l’URL de l’endpoint et le processus de déploiement

---

## Étape 7 : Tableau de Bord Power BI

- [ ] Exporter les prédictions, valeurs SHAP et données originales en CSV
- [ ] Dans Power BI :
  - Importer les données
  - Créer des KPIs (Taux de churn global, Nombre de clients à risque élevé, etc.)
  - Créer des slicers (Contract, Tenure, MonthlyCharges, etc.)
  - Réaliser les visualisations :
    - Taux de churn par Contract, PaymentMethod, InternetService
    - Distribution de Tenure avec churn
    - Top clients à risque
    - Feature Importance (SHAP)
    - Métriques de performance du modèle
- [ ] Ajouter une page de recommandations de rétention
- [ ] Publier le rapport sur Power BI Service (optionnel)

**Deliverable** : Fichier `.pbix`

---

## Étape 8 : Documentation et Finalisation du Projet

- [ ] Créer un `README.md` professionnel contenant :
  - Objectif du projet
  - Description du dataset
  - Résultats clés (AUC-ROC = 0.89)
  - Tableau de comparaison des modèles
  - Insights SHAP
  - Informations sur le déploiement Azure ML
  - Captures d’écran du dashboard Power BI
  - Instructions pour exécuter le projet
- [ ] Organiser la structure des dossiers
- [ ] Générer `requirements.txt`
- [ ] (Optionnel) Pousser le projet sur GitHub avec un bon historique de commits