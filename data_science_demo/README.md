Data Science Demo Project
=========================

Zweck
-----
Dieses kleine Demo-Projekt zeigt eine kompakte, reproduzierbare Pipeline, die viele Anforderungen aus der ausgeschriebenen
Booz Allen Data Scientist-Stelle abdeckt: ETL (CSV <-> SQLite), Datenbereinigung, Feature-Engineering (inkl. Zeitreihen-Features
und einfache Text-Vektorisierung), Modelltraining (Gradient Boosting), Backtesting/Time-based Split, Evaluation (RMSE, MAE),
und Visualisierung (matplotlib).

Inhalte
-------
- `main.py`        : Orchestrator (ETL -> Training -> Evaluation -> Save outputs)
- `etl.py`         : Erzeugt Beispieldaten, läd und säubert sie, speichert in SQLite (simuliert strukturierte Speicherung)
- `model.py`       : Trainiert ein Modell, macht Vorhersagen und Evaluationsmetriken
- `visualize.py`   : Erzeugt und speichert Plots (Zeitreihe, Feature-Importances)
- `requirements.txt`: Empfohlene Bibliotheken
- `sample_data.csv`: Generierte Beispieldaten (Datum, numerische Features, Text, Zielvariable)
- `README.md`      : Diese Datei
- `demo_report.txt`: Kurzer Abgleich: wie dieses Projekt Anforderungen in der Stellenausschreibung erfüllt

Ausführen
--------
Python 3.8+ (empfohlen). Installation (optional):
```
pip install -r requirements.txt
```
Dann:
```
python main.py
```
Outputs werden im Ordner `outputs/` erzeugt:
- `model.pkl` (gespeichertes Modell)
- `metrics.json` (RMSE / MAE)
- `plots/*.png` (Visualisierungen)

Mapping zur Stellenbeschreibung
-------------------------------
- **ETL / SQL**: `etl.py` zeigt das Laden und Speichern in SQLite sowie grundlegende Datenvalidierungschecks.
- **Python / SQL / Feature Engineering**: Kernlogik in `etl.py` und `model.py` nutzt pandas, sklearn & SQL (sqlite3).
- **Machine Learning / Evaluation**: `model.py` trainiert ein GradientBoostingRegressor und erstellt Backtesting-artige Splits.
- **Unstructured Data / NLP**: Der Text-Feature-Workflow zeigt einfache TF-IDF Vektorisierung als Beispiel für Text-Mining.
- **Visualisierung**: `visualize.py` erstellt Plots für Stakeholder-Reporting.
