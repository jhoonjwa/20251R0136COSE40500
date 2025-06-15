# ------------------------------------------------------------
# 1.  Setup
# ------------------------------------------------------------
# pip install -q sentence-transformers scikit-learn pot

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import ot                                  # POT: Python Optimal Transport

### use it well

# ---------------- user data -------------------------------------------------
import pandas as pd

# Load the Excel file
file_path = "/home/ubuntu/jonghoon/AI-TA/datasets/2번.xlsx"
df = pd.read_excel(file_path)

# Extract the relevant columns
kai_pred = df["agent debate"].dropna().tolist()
gpt_pred = df["단일 프롬프팅"].dropna().tolist()
gt_actual = df["학생 실 질문-agent"].dropna().tolist()
# ---------------------------------------------------------------------------

# ------------------------------------------------------------
# 2.  Embed sentences
# ------------------------------------------------------------
encoder = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

A = encoder.encode(kai_pred,   batch_size=64, show_progress_bar=True)  # (n_a, d)
B = encoder.encode(gpt_pred,   batch_size=64, show_progress_bar=True)  # (n_b, d)
C = encoder.encode(gt_actual,  batch_size=64, show_progress_bar=True)  # (n_c, d)

# L2-normalise so dot == cosine
A = normalize(A)
B = normalize(B)
C = normalize(C)

# ------------------------------------------------------------
# 3.  Helper metrics
# ------------------------------------------------------------
def set_f1(X, Y) -> float:
    """
    Symmetric max-cosine F1 between two embedding sets.
    Larger ⇒ more similar.
    """
    sim = X @ Y.T                           # (|X|, |Y|)
    recall    = sim.max(axis=1).mean()      # X → Y
    precision = sim.max(axis=0).mean()      # Y → X
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def set_mover_distance(X, Y) -> float:
    """
    Earth/Set-Mover distance (optimal transport) between two point clouds
    on the unit hypersphere.  Smaller ⇒ more similar.
    """
    # cosine → distance in [0, 2]
    M = 1.0 - X @ Y.T                       # cost matrix             (|X|, |Y|)
    a = np.ones(len(X)) / len(X)            # uniform weights
    b = np.ones(len(Y)) / len(Y)
    return ot.emd2(a, b, M)                 # quadratic cost (Wasserstein-2)

# ------------------------------------------------------------
# 4.  Score KAI and GPT-4o against ground-truth
# ------------------------------------------------------------
f1_kai  = set_f1(A, C)
f1_gpt  = set_f1(B, C)

emd_kai = set_mover_distance(A, C)
emd_gpt = set_mover_distance(B, C)

print("\n-----   Results   -----")
print(f"Set-F1   KAI vs GT : {f1_kai:.4f}")
print(f"Set-F1   GPT vs GT : {f1_gpt:.4f}")
print(f"EMD      KAI vs GT : {emd_kai:.4f}")
print(f"EMD      GPT vs GT : {emd_gpt:.4f}")

winner_f1  = "KAI" if f1_kai  > f1_gpt  else "GPT-4o"
winner_emd = "KAI" if emd_kai < emd_gpt else "GPT-4o"
print(f"\nF1  winner : {winner_f1}")
print(f"EMD winner : {winner_emd}")
