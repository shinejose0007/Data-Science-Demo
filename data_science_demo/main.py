from etl import run_etl
from model import train_model, load_data_from_sqlite
from visualize import timeseries_plot, feature_importance_plot
import json, os

def main():
    out = 'outputs'
    # ETL
    df, db_path = run_etl(out_dir=out)
    # Train & evaluate
    metrics = train_model(df, out_dir=out, n_splits=4)
    # Save metrics
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # Visualize
    timeseries_plot(df, out_dir=out)
    feature_importance_plot(metrics.get('feature_importance', {}), out_dir=out)
    print('Pipeline abgeschlossen. Outputs im Ordner:', out)

if __name__ == '__main__':
    main()
