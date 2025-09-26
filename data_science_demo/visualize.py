import pandas as pd
import matplotlib.pyplot as plt
import os

def timeseries_plot(df, out_dir='outputs'):
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(df['date'], df['target'], label='target')
    plt.title('Zielvariable über Zeit')
    plt.xlabel('Datum')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots', 'timeseries_target.png'))
    plt.close()

def feature_importance_plot(fi, out_dir='outputs'):
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    if not fi:
        return
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names = [i[0] for i in items]
    values = [i[1] for i in items]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.title('Feature Importances (numeric features approx.)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots', 'feature_importance.png'))
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv('outputs/sample_data.csv', parse_dates=['date'])
    timeseries_plot(df)
    # try to read feature importances from metrics
    import json
    try:
        with open('outputs/metrics.json','r') as f:
            metrics = json.load(f)
            fi = metrics.get('feature_importance', {})
    except Exception:
        fi = {}
    feature_importance_plot(fi)
