import pickle
import numpy as np

# Load preprocessing.pkl generated during training
with open("models/preprocessing.pkl", "rb") as f:
    p = pickle.load(f)

scaler = p["scaler"]
lower = p["lower"]
upper = p["upper"]

print("=== LOWER (as python list) ===")
print(lower.values.tolist())

print("\n=== UPPER (as python list) ===")
print(upper.values.tolist())

print("\n=== SCALER MEAN_ (python list) ===")
print(scaler.mean_.tolist())

print("\n=== SCALER SCALE_ (python list) ===")
print(scaler.scale_.tolist())
