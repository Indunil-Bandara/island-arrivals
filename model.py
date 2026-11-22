"""Portable inference for the fitted trees. Only Python's standard library is needed."""
from pathlib import Path
import math
import struct

ROOT = Path(__file__).resolve().parent
NAMES = ["Seasonal naive", "AR(1)", "Random Forest", "XGBoost", "AR + Forest", "Stacking"]

def features(month, previous, two_back, year_back):
    angle = 2 * math.pi * month / 12
    return [previous, two_back, year_back, math.sin(angle), math.cos(angle)]

def float32(number):
    return struct.unpack("f", struct.pack("f", number))[0]

def leaf(tree, x, strict=False):
    while isinstance(tree, list):
        feature, threshold, left, right = tree
        below = x[feature] < float32(threshold) if strict else x[feature] <= threshold
        tree = left if below else right
    return tree

def predict_one(model, row):
    x = list(map(float32, row))
    ar = model["intercept"] + model["slope"] * row[0]
    def forest(key):
        trees = model[key]
        return sum(leaf(t, x) for t in trees) / len(trees)
    boost = float32(model["base"])
    for tree in model["boosted"]:
        boost = float32(boost + float32(leaf(tree, x, strict=True)))
    rf = forest("forest")
    w_rf, w_boost = model["stack_weights"]
    stack = model["stack_bias"] + w_rf * rf + w_boost * boost
    values = [row[2], ar, rf, boost, ar + forest("residual"), stack]
    return dict(zip(NAMES, (max(0, v) for v in values)))
