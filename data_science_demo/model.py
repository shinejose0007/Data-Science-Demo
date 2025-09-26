import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

def load_data_from_sqlite(db_path='outputs/data.db', table='observations'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn, parse_dates=['date'])
    conn.close()
    return df

def create_features(df):
    df = df.copy()
    # time features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    # rolling features
    df['numeric_1_roll7'] = df['numeric_1'].rolling(7, min_periods=1).mean()
    # text -> tfidf (we'll vectorize later)
    return df

def train_model(df, out_dir='outputs', n_splits=5):
    os.makedirs(out_dir, exist_ok=True)
    df = create_features(df)
    # Sort by date for time-series split
    df = df.sort_values('date').reset_index(drop=True)
    # target & basic numeric features
    X_num = df[['numeric_1','numeric_2','numeric_1_roll7','dayofweek','day','month']].fillna(0)
    # text features via TF-IDF
    tfidf = TfidfVectorizer(max_features=50)
    X_text = tfidf.fit_transform(df['text'].astype(str))
    # Combine numeric and text by converting numeric to dense and hstack
    from scipy import sparse
    X = sparse.hstack([sparse.csr_matrix(X_num.values), X_text]).tocsr()
    y = df['target'].values
    # TimeSeriesSplit for backtesting-like evaluation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {'rmse': [], 'mae': []}
    models = []
    split_idx = 0
    for train_index, test_index in tscv.split(X):
        split_idx += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # simple pipeline: scaler (for numeric part) isn't applied to sparse matrices; GradientBoosting handles raw
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        models.append(model)
    # Train a final model on full data
    final_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    final_model.fit(X, y)
    # save artifacts
    with open(os.path.join(out_dir, 'tfidf.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(out_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
    # aggregate metrics
    metrics_summary = {'rmse_mean': float(np.mean(metrics['rmse'])), 'mae_mean': float(np.mean(metrics['mae']))}
    # feature importance for numeric features only (approx)
    try:
        import numpy as np
        numeric_feature_names = ['numeric_1','numeric_2','numeric_1_roll7','dayofweek','day','month']
        importances = final_model.feature_importances_[:len(numeric_feature_names)]
        fi = dict(zip(numeric_feature_names, map(float, importances)))
    except Exception as e:
        fi = {}
    metrics_summary['feature_importance'] = fi
    return metrics_summary

if __name__ == '__main__':
    df = load_data_from_sqlite('outputs/data.db')
    metrics = train_model(df, 'outputs', n_splits=4)
    print('Training abgeschlossen. Metrics:', metrics)
