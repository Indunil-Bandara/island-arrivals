# Island Arrivals
A one-month-ahead tourism forecasting dashboard for Machine Learning & Pattern Recognition. Compare six methods on the same 36 historical test months and explore how one December observation changes a January estimate.

## Quick start on Windows
1. Extract the ZIP to a new folder. Open the folder containing app.py in VS Code.
2. In Terminal → New Terminal, run the following command with Python 3.12 installed.

~~~powershell
py -3.12 app.py
~~~

3. Open http://127.0.0.1:8501 in your browser. Keep the terminal open while using the dashboard. Press Ctrl+C to stop it.

The supplied backend runs on the Python standard library. It loads the fitted model parameters from models/arrivals.pkl. The pickle contains plain Python numbers, lists and dictionaries, including learned tree splits and leaves. The backend evaluates the fitted models on each prediction request. This format removes the dependency on native XGBoost snapshots when starting the dashboard on Windows.

## Retrain and run the notebook
Use Python 3.12, the VS Code Python extension and the Jupyter extension. In the project folder run:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe app.py
~~~

Alternatively open ensemble_study.ipynb, select the .venv kernel and choose Run All. The notebook imports the same project training code, fits models on all three chronological folds, exports the final parameters and displays the scores and research comparison. Restart the dashboard after retraining to load the new files.

## Demo controls
- Use the sun or moon icon in the header to switch themes. Your selection is retained on this browser when local storage is available.
- Enter 200000 in the December arrivals field and press Enter to compare the observed-input forecast with a lower December scenario. Reset to observed restores the reported December 2025 count.
- Select a year or All years to inspect the chart and error table. Inspect one month compares all methods with the selected month's observed arrivals.

## Data and experimental design
Source: Sri Lanka Tourism Development Authority, https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports . The CSV includes a source URL and audit note for each month. There are 132 consecutive national counts, January 2015–December 2025. For April–November 2020, the report displays dashes. The four numerical months sum to the published annual total; those eight records are represented as zero.

Features are lags 1, 2 and 12, plus a month-of-year sine and cosine. The initial year supplies lag history. Train on 84 rows for the 2023 test, 96 for 2024 and 108 for 2025. Forecast parameters stay fixed throughout each test year. As each month ends its newly observed count becomes available for the next prediction. Each annual test contains 12 one-step forecasts. Settings and seed (42 for the forests and XGBoost) are fixed. After evaluation, all 120 eligible rows train the January 2026 replay model. This retrospective replay has a December 2025 information cutoff.

## Stacking
Random Forest and XGBoost are the two base estimators. Four expanding chronological splits inside each outer training window generate out-of-fold forecasts. An ordinary least-squares linear regression learns an intercept and two weights from these forecasts. Both base models then refit on the full training window. At inference the fitted linear equation combines their outputs. The annual test observations never fit that year’s stack.

## Results
| Model | 2023 MAE | 2024 MAE | 2025 MAE | Pooled MAE | Pooled RMSE |
|---|---:|---:|---:|---:|---:|
| Seasonal naive | 63,944 | 47,180 | 25,755 | 45,626 | 55,563 |
| AR(1) | 21,767 | 31,381 | 26,941 | 26,696 | 34,593 |
| Random Forest | 25,114 | 31,585 | 14,896 | 23,865 | 30,990 |
| XGBoost | 27,481 | 29,882 | 17,662 | 25,008 | 31,728 |
| AR + Forest | 16,937 | 22,580 | 10,612 | 16,710 | 22,816 |
| Stacking | 55,844 | 48,987 | 38,200 | 47,677 | 54,800 |

Stacking had higher pooled MAE than either base model on these 36 months. MAE is average absolute error in arrivals. Pooled RMSE is the square root of the mean squared error over all 36 months. The forest correction reduced pooled AR(1) MAE by 37.4%. Evaluation covers three adjacent years. Changing one lag input measures model sensitivity; it does not establish a causal demand effect.

## Submission contents
- ensemble_study.ipynb: executed analysis, plots, benchmark comparisons and cited research on Random Forest, XGBoost, LightGBM, CatBoost, tabular, image and text features.
- data/arrivals.csv: national monthly counts and original source links.
- train.py and requirements.txt: reproducible training and evaluation with scikit-learn and XGBoost.
- model.py and models/: portable model inference, fitted parameters and full evaluation results.
- app.py and index.html: Python HTTP backend and interactive frontend.
- export_host.py and worker-template.js: rebuild a self-contained hosted Worker module with python export_host.py dist/server/index.js.
- presentation/Island_Arrivals.pptx: five editable slides.
- presentation/Presentation_Script.txt: separate English speaker script with timing and demo cues.

## Publish the dashboard
Publish this folder to GitHub with app.py at the repository root. In Render create a Python web service connected to the repository. Set HOST to 0.0.0.0, use python -m compileall -q app.py model.py as the build command, and python app.py as the start command. Render supplies PORT. The training dependencies are needed only for the notebook and retraining.
