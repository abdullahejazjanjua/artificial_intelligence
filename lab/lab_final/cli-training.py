import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    root_mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score
)

# Strict Model Imports from your shared scripts
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    BaggingClassifier, BaggingRegressor, 
    VotingClassifier, VotingRegressor
)
from sklearn.cluster import KMeans

# Suppress warnings for exhaustive search
warnings.filterwarnings("ignore")

def clean_data(df):
    """Cleaning logic from provided scripts."""
    if df.isnull().sum().sum() > 0:
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df = df.dropna()
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df

def get_preprocessor(X):
    """Preprocessing logic (StandardScaler + OneHotEncoder)."""
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
    
    # Exhaustive grids for classification.py models
    # n_jobs=-1 enables full CPU acceleration for the grid search.
    models = {
        'LogisticRegression': (LogisticRegression(max_iter=2000), {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'poly__degree': [1, 2]
        }),
        'DecisionTree': (DecisionTreeClassifier(random_state=42), {
            'clf__max_depth': [3, 5, 10, 20, None],
            'clf__min_samples_leaf': [1, 5, 10],
            'clf__criterion': ['gini', 'entropy']
        }),
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5]
        }),
        'KNN': (KNeighborsClassifier(), {
            'clf__n_neighbors': [3, 5, 7, 11, 21],
            'clf__weights': ['uniform', 'distance']
        }),
        'Bagging': (BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42), {
            'clf__n_estimators': [10, 50, 100],
            'clf__max_samples': [0.5, 0.8, 1.0]
        }),
        'Voting': (VotingClassifier(estimators=[
            ('m1', LogisticRegression(max_iter=2000)),
            ('m2', DecisionTreeClassifier(max_depth=5)),
            ('m3', RandomForestClassifier(n_estimators=100)),
            ('m4', KNeighborsClassifier())
        ]), {
            'clf__voting': ['hard', 'soft'],
            'clf__weights': [[1,1,1,1], [2,1,2,1], [1,2,1,2]]
        })
    }

    best_score = -1
    best_results = None

    for name, (model, param_grid) in models.items():
        print(f"Exhaustive Search: {name}...")
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('clf', model)])
        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        score = gs.score(X_test, y_test)
        print(f"-> Best {name} Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_results = (name, gs.best_params_, gs.best_estimator_)

    print(f"\n--- WINNER: {best_results[0]} ---")
    print(f"Accuracy: {best_score:.4f}")
    print(f"Best Parameters: {best_results[1]}")
    
    y_pred = best_results[2].predict(X_test)
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if not no_vis:
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {best_results[0]}")
        plt.show()

def run_regression(X, y, no_vis):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = get_preprocessor(X)
    
    # Exhaustive grids for regression.py models
    models = {
        'LinearRegression': (LinearRegression(), {
            'poly__degree': [1, 2, 3]
        }),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {
            'reg__max_depth': [3, 5, 10, None],
            'reg__min_samples_leaf': [1, 2, 5],
            'reg__criterion': ['squared_error', 'absolute_error']
        }),
        'RandomForest': (RandomForestRegressor(random_state=42), {
            'reg__n_estimators': [50, 100, 200],
            'reg__max_depth': [10, 20, None]
        }),
        'Bagging': (BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=42), {
            'reg__n_estimators': [10, 50, 100],
            'reg__max_samples': [0.5, 0.8, 1.0]
        }),
        'Voting': (VotingRegressor(estimators=[
            ('m1', LinearRegression()),
            ('m2', DecisionTreeRegressor(max_depth=3)),
            ('m3', RandomForestRegressor(n_estimators=100))
        ]), {
            'reg__weights': [[1, 1, 1], [2, 1, 1], [1, 1, 2]]
        })
    }

    best_r2 = -np.inf
    best_results = None

    for name, (model, param_grid) in models.items():
        print(f"Exhaustive Search: {name}...")
        pipe = Pipeline([('pre', pre), ('poly', PolynomialFeatures()), ('reg', model)])
        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        y_pred = gs.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"-> {name} R2 Score: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_results = (name, gs.best_params_, gs.best_estimator_)

    print(f"\n--- WINNER: {best_results[0]} ---")
    print(f"R2 Score: {best_r2:.4f}")
    print(f"Best Parameters: {best_results[1]}")
    print(f"RMSE: {root_mean_squared_error(y_test, best_results[2].predict(X_test)):.2f}")

def run_clustering(X):
    """Clustering logic from clustering.py with optimal K search."""
    pre = get_preprocessor(X)
    X_scaled = pre.fit_transform(X).astype(float)
    
    max_silhouette = -1
    best_k = 2

    print("Checking K from 2 to 10...")
    for k in range(2, 11):
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K={k} | Silhouette Score: {score:.4f}")
        if score > max_silhouette:
            max_silhouette = score
            best_k = k
            
    print(f"\nOptimal Clusters: {best_k} (Silhouette: {max_silhouette:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exhaustive AutoML CLI Tool")
    parser.add_argument("--mode", choices=['regression', 'classification', 'clustering'], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", help="Target column for supervised learning")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    df = clean_data(pd.read_csv(args.dataset))
    
    if args.mode == 'clustering':
        run_clustering(df.select_dtypes(include='number'), args.no_vis)
    else:
        if not args.target:
            print("Error: --target is required for regression and classification.")
        else:
            X, y = df.drop(columns=[args.target]), df[args.target]
            if args.mode == 'classification':
                run_classification(X, y, args.no_vis)
            else:
                run_regression(X, y, args.no_vis)