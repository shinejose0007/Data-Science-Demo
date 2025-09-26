import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os

def generate_sample_data(path='sample_data.csv', n=500, seed=42):
    np.random.seed(seed)
    start = datetime(2022,1,1)
    dates = [start + timedelta(days=i) for i in range(n)]
    trend = np.linspace(0, 5, n)
    seasonality = 2 * np.sin(np.linspace(0, 3.14*4, n))
    noise = np.random.normal(0, 0.8, n)
    numeric_1 = 10 + trend + seasonality + noise
    numeric_2 = np.random.randint(0,100, n)
    # simple text field with a few categories + noise
    texts = np.random.choice(['report', 'error', 'update', 'maintenance', 'alert'], size=n)
    texts = [t + ' ' + ('high' if x%7==0 else 'low') for x,t in enumerate(texts)]
    # target is a continuous value
    target = numeric_1 * 1.5 + numeric_2 * 0.05 + np.random.normal(0,1,n)
    df = pd.DataFrame({
        'date': dates,
        'numeric_1': numeric_1,
        'numeric_2': numeric_2,
        'text': texts,
        'target': target
    })
    df.to_csv(path, index=False)
    return df

def load_and_clean(csv_path='sample_data.csv'):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # basic validation
    assert 'target' in df.columns, "Zielspalte fehlt"
    # drop duplicates, sort by date
    df = df.drop_duplicates().sort_values('date').reset_index(drop=True)
    # simple missing value handling
    df['numeric_1'] = df['numeric_1'].fillna(df['numeric_1'].median())
    df['numeric_2'] = df['numeric_2'].fillna(df['numeric_2'].median())
    df['text'] = df['text'].fillna('missing')
    return df

def save_to_sqlite(df, db_path='data.db', table='observations'):
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists='replace', index=False)
    conn.close()

def run_etl(out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'sample_data.csv')
    # generate sample data
    df = generate_sample_data(path=csv_path, n=600)
    # load & clean
    df = load_and_clean(csv_path)
    # save to sqlite database
    db_path = os.path.join(out_dir, 'data.db')
    save_to_sqlite(df, db_path)
    return df, db_path

if __name__ == '__main__':
    df, db = run_etl('outputs')
    print('ETL fertig, date sample:', df['date'].min(), '->', df['date'].max())
