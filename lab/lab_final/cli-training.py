import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pprint import pprint # For clean printing of dictionaries

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, silhouette_score

# Models strictly from original scripts
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    BaggingClassifier, BaggingRegressor, 
    VotingClassifier, VotingRegressor
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

def clean_data(df):
    """Handles missing values and duplicates."""
    if df.isnull().sum().sum() > 0:
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df = df.dropna()
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df

def get_preprocessor(X):
    """Builds a processing pipeline for numeric and categorical features."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
        ]
    )

def run_classification(X, y, no_vis):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = get_preprocessor(X)
    
    # Tracking dictionaries
    phase1_params = {}
    phase2_params = {}
    
    print("\n--- PHASE 1: Optimizing Base Models ---")
    base_configs = {
        'LogisticRegression': (LogisticRegression(max_iter=5000), {
            'clf__C': [0.1, 1, 10, 100],
            'poly__degree': [1, 2]
        }),
        'DecisionTree': (DecisionTreeClassifier(random_state=42), {
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_leaf': [1, 5]
        }),
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, None]
        }),
        'KNN': (KNeighborsClassifier(), {
            'clf__n_neighbors': [3, 5, 11],
            'clf__weights': ['uniform', 'distance']
        })
    }

    best_base_models = {}
    base_scores = {}
    
    for name, (model, grid) in base_configs.items():
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('clf', model)])
        gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        score = gs.score(X_test, y_test)
        best_base_models[name] = gs.best_estimator_.named_steps['clf']
        base_scores[name] = score
        phase1_params[name] = gs.best_params_
        print(f"Best {name} Accuracy: {score:.4f}")

    print("\n--- PHASE 2: Optimizing Ensembles ---")
    ensemble_configs = {
        'Bagging': (BaggingClassifier(estimator=best_base_models['DecisionTree'], random_state=42), {
            'clf__n_estimators': [50, 100],
            'clf__max_samples': [0.8, 1.0]
        }),
        'Voting': (VotingClassifier(estimators=[
            ('lr', best_base_models['LogisticRegression']),
            ('dt', best_base_models['DecisionTree']),
            ('rf', best_base_models['RandomForest']),
            ('kn', best_base_models['KNN'])
        ]), {
            'clf__voting': ['hard', 'soft'],
            'clf__weights': [[1,1,1,1], [2,1,2,1], [1,1,2,1]]
        })
    }

    final_results = base_scores.copy()
    model_objects = {name: Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('clf', best_base_models[name])]).fit(X_train, y_train) for name in best_base_models}

    for name, (model, grid) in ensemble_configs.items():
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('clf', model)])
        gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        score = gs.score(X_test, y_test)
        final_results[name] = score
        model_objects[name] = gs.best_estimator_
        phase2_params[name] = gs.best_params_
        print(f"Best {name} Ensemble Accuracy: {score:.4f}")

    winner = max(final_results, key=final_results.get)
    print(f"\n{'='*50}\nFINAL WINNER: {winner} | SCORE: {final_results[winner]:.4f}\n{'='*50}")

    # Logic to print hyperparams based on which phase won
    if winner in phase2_params:
        print(f"--- {winner} Ensemble Hyperparameters ---")
        pprint(phase2_params[winner])
        print(f"\n--- Underlying Base Model Hyperparameters (from Phase 1) ---")
        if winner == 'Bagging':
             pprint({ "DecisionTree": phase1_params['DecisionTree'] })
        elif winner == 'Voting':
             pprint(phase1_params)
    else:
        print(f"--- {winner} Hyperparameters ---")
        pprint(phase1_params[winner])
    print(f"{'='*50}\n")

    if not no_vis:
        y_pred = model_objects[winner].predict(X_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {winner}")
        plt.show()

def run_regression(X, y, no_vis):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = get_preprocessor(X)
    
    phase1_params = {}
    phase2_params = {}

    print("\n--- PHASE 1: Optimizing Base Regression Models ---")
    base_configs = {
        'LinearRegression': (LinearRegression(), {'poly__degree': [1, 2, 3]}),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {
            'reg__max_depth': [5, 10, None],
            'reg__min_samples_leaf': [1, 5]
        }),
        'RandomForest': (RandomForestRegressor(random_state=42), {
            'reg__n_estimators': [100, 200]
        })
    }

    best_base_regs = {}
    reg_scores = {}

    for name, (model, grid) in base_configs.items():
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('reg', model)])
        gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        r2 = gs.score(X_test, y_test)
        best_base_regs[name] = gs.best_estimator_.named_steps['reg']
        reg_scores[name] = r2
        phase1_params[name] = gs.best_params_
        print(f"Best {name} R2 Score: {r2:.4f}")

    print("\n--- PHASE 2: Optimizing Regression Ensembles ---")
    ensemble_configs = {
        'Bagging': (BaggingRegressor(estimator=best_base_regs['DecisionTree'], random_state=42), {
            'reg__n_estimators': [50, 100]
        }),
        'Voting': (VotingRegressor(estimators=[
            ('lr', best_base_regs['LinearRegression']),
            ('dt', best_base_regs['DecisionTree']),
            ('rf', best_base_regs['RandomForest'])
        ]), {
            'reg__weights': [[1,1,1], [2,1,1], [1,1,2]]
        })
    }

    final_results = reg_scores.copy()

    for name, (model, grid) in ensemble_configs.items():
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('reg', model)])
        gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        r2 = gs.score(X_test, y_test)
        final_results[name] = r2
        phase2_params[name] = gs.best_params_
        print(f"Best {name} Ensemble R2: {r2:.4f}")

    winner = max(final_results, key=final_results.get)
    print(f"\n{'='*50}\nFINAL WINNER: {winner} | R2 SCORE: {final_results[winner]:.4f}\n{'='*50}")

    if winner in phase2_params:
        print(f"--- {winner} Ensemble Hyperparameters ---")
        pprint(phase2_params[winner])
        print("\n--- Underlying Base Model Hyperparameters (from Phase 1) ---")
        if winner == 'Bagging':
             pprint({ "DecisionTree": phase1_params['DecisionTree'] })
        elif winner == 'Voting':
             pprint(phase1_params)
    else:
        print(f"--- {winner} Hyperparameters ---")
        pprint(phase1_params[winner])
    print(f"{'='*50}\n")

def run_clustering(X, no_vis):
    pre = get_preprocessor(X)
    X_scaled = pre.fit_transform(X).astype(float)
    max_silhouette, best_k = -1, 2

    print("\n--- Exhaustively checking K Clusters (2-10) ---")
    for k in range(2, 11):
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20)
        score = silhouette_score(X_scaled, model.fit_predict(X_scaled))
        print(f"K={k} | Silhouette Score: {score:.4f}")
        if score > max_silhouette:
            max_silhouette, best_k = score, k
    print(f"\n{'='*50}\nOPTIMAL CLUSTERS: {best_k} (Silhouette: {max_silhouette:.4f})\n{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Phase Maximum Performance ML Tool")
    parser.add_argument("--mode", choices=['regression', 'classification', 'clustering'], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    
    df = clean_data(pd.read_csv(args.dataset))
    
    if args.mode == 'clustering':
        run_clustering(df.select_dtypes(include='number'), args.no_vis)
    elif args.target:
        X, y = df.drop(columns=[args.target]), df[args.target]
        if args.mode == 'classification':
            run_classification(X, y, args.no_vis)
        else:
            run_regression(X, y, args.no_vis)
    else:
        print("Error: --target column is required for supervised learning.")