"""Train and evaluate one-step-ahead national arrivals forecasts."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
from model import NAMES, predict_one
FEATURES = ["lag1", "lag2", "lag12", "season_sin", "season_cos"]

def features(month, previous, two_back, year_back):
    angle = 2 * np.pi * month / 12
    return [previous, two_back, year_back, np.sin(angle), np.cos(angle)]

def dataset():
    data = pd.read_csv(ROOT / "data/arrivals.csv", parse_dates=["month"])
    assert len(data) == 132 and data.month.tolist() == list(pd.date_range("2015-01-01", periods=132, freq="MS"))
    assert data.arrivals.notna().all() and (data.arrivals >= 0).all()
    for lag in [1, 2, 12]:
        data[f"lag{lag}"] = data.arrivals.shift(lag)
    angle = 2 * np.pi * data.month.dt.month / 12
    return data.assign(season_sin=np.sin(angle), season_cos=np.cos(angle)).dropna()

def forest():
    return RandomForestRegressor(n_estimators=48, max_depth=5, min_samples_leaf=2, random_state=42, n_jobs=1)

def fit_base(x, y):
    rf = forest().fit(x, y)
    boost = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=.075, objective="reg:squarederror", n_jobs=1, random_state=42).fit(x, y)
    return rf, boost

def fit_stacker(x, y):
    # Every row seen by the meta-model was forecast by earlier-only base models.
    meta_inputs, targets = [], []
    for earlier, later in TimeSeriesSplit(n_splits=4).split(x):
        rf, boost = fit_base(x[earlier], y[earlier])
        meta_inputs.append(np.column_stack([rf.predict(x[later]), boost.predict(x[later])]))
        targets.extend(y[later])
    return LinearRegression().fit(np.vstack(meta_inputs), targets)

def fit(frame):
    x, y = frame[FEATURES].to_numpy(), frame.arrivals.to_numpy()
    stacker = fit_stacker(x, y)
    rf, boost = fit_base(x, y)
    slope, intercept = np.polyfit(x[:, 0], y, 1)
    residuals = y - (intercept + slope * x[:, 0])
    correction = forest().fit(x, residuals)
    return rf, boost, float(slope), float(intercept), correction, stacker


def compact_sklearn(estimator, node=0):
    t = estimator.tree_
    if t.children_left[node] < 0:
        return float(t.value[node, 0, 0])
    return [int(t.feature[node]), float(t.threshold[node]), compact_sklearn(estimator, t.children_left[node]), compact_sklearn(estimator, t.children_right[node])]

def compact_boost(node):
    if "leaf" in node:
        return node["leaf"]
    children = {child["nodeid"]: child for child in node["children"]}
    return [int(node["split"][1:]), node["split_condition"], compact_boost(children[node["yes"]]), compact_boost(children[node["no"]])]

def portable(fitted):
    rf, boost, slope, intercept, correction, stacker = fitted
    b = boost.get_booster()
    params = json.loads(b.save_config())["learner"]["learner_model_param"]
    base = json.loads(params["base_score"])[0]
    return dict(stack_weights=stacker.coef_.tolist(), stack_bias=float(stacker.intercept_), base=base, slope=slope, intercept=intercept,
                forest=[compact_sklearn(t) for t in rf.estimators_],
                residual=[compact_sklearn(t) for t in correction.estimators_],
                boosted=[compact_boost(json.loads(t)) for t in b.get_dump(dump_format="json")])

def score(rows):
    actual = [r["actual"] for r in rows]
    result = {}
    for name in NAMES:
        predicted = [r["pred"][name] for r in rows]
        result[name] = dict(MAE=float(mean_absolute_error(actual, predicted)),
                            RMSE=float(root_mean_squared_error(actual, predicted)))
    return result

def evaluate(frame):
    rows, scores = [], {}
    for year in [2023, 2024, 2025]:
        history = frame[frame.month.dt.year < year]
        test = frame[frame.month.dt.year == year]
        fitted = fit(history)
        model = portable(fitted)
        fold = []
        for _, item in test.iterrows():
            row = item[FEATURES].astype(float).tolist()
            values = predict_one(model, row)
            fold.append(dict(month=item.month.strftime("%Y-%m"), actual=int(item.arrivals), pred=values))
        rf, booster, slope, intercept, correction, stacker = fitted
        x = test[FEATURES].to_numpy()
        ar = slope*x[:,0] + intercept
        native = np.maximum(0, np.column_stack([x[:,2], ar, rf.predict(x), booster.predict(x), ar+correction.predict(x),
            stacker.predict(np.column_stack([rf.predict(x), booster.predict(x)]))]))
        exported = [[r["pred"][n] for n in NAMES] for r in fold]
        np.testing.assert_allclose(exported, native, rtol=0, atol=0.1)
        scores[str(year)] = score(fold)
        rows.extend(fold)
    scores["All years"] = score(rows)
    return rows, scores

def train():
    frame = dataset()
    rows, scores = evaluate(frame)
    model = portable(fit(frame))
    result = dict(rows=rows, scores=scores, names=NAMES, prior=int(frame.iloc[-1].arrivals),
                  two_back=int(frame.iloc[-2].arrivals), year_back=int(frame.iloc[-12].arrivals))
    from model import features
    result["observed"] = predict_one(model, features(1, result["prior"], result["two_back"], result["year_back"]))
    path = ROOT / "models"
    path.mkdir(exist_ok=True)
    # Store only primitive Python data, avoiding platform-specific estimator snapshots.
    (path / "arrivals.pkl").write_bytes(pickle.dumps(model, protocol=4))
    (path / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    for period, scores in train()["scores"].items():
        print(period, pd.DataFrame(scores).T.round(1), sep="\n")
