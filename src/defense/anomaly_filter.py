"""
Semantic Anomaly Filter (Defense Layer 1)
-----------------------------------------
Uses sentence embeddings (all-MiniLM-L6-v2) + IsolationForest to detect
poisoned observations. Target: <15ms per check on CPU.

NOVEL: ROC-optimal threshold auto-tuning via auto_tune_threshold().
Instead of fixed contamination parameter, finds the threshold that
minimizes FPR while keeping catch rate > 85%.
"""

import os
import time
import pickle
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "sentence-transformers not installed. SemanticAnomalyFilter will use "
        "a simple character-frequency fallback encoder.",
        RuntimeWarning,
    )


class _FallbackEncoder:
    """Simple character n-gram frequency encoder as fallback."""

    def encode(self, texts, show_progress_bar=False, batch_size=32):
        import re
        vectors = []
        for text in texts:
            text = text.lower()
            # 384-dim vector via character frequency + simple features
            vec = np.zeros(384, dtype=np.float32)
            words = text.split()
            vec[0] = len(words) / 100.0
            vec[1] = text.count("compromised") / 10.0
            vec[2] = text.count("operational") / 10.0
            vec[3] = text.count("critical") / 10.0
            vec[4] = text.count("anomaly") / 10.0
            vec[5] = text.count("remove") / 5.0
            vec[6] = text.count("restore") / 5.0
            vec[7] = text.count("monitor") / 5.0
            # Fill with character frequencies
            for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
                vec[8 + i] = text.count(c) / (len(text) + 1)
            vectors.append(vec)
        return np.array(vectors)


class SemanticAnomalyFilter:
    """
    Semantic anomaly detector for CAGE 4 observation texts.

    Defense Layer 1: Fast unsupervised detection of poisoned observations.
    Combines sentence embeddings with IsolationForest anomaly scoring.

    NOVEL: ROC-optimal auto_tune_threshold() finds the threshold that
    minimizes FPR while maintaining >85% true positive rate.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        sim_threshold: float = 0.82,
    ):
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.encoder = _FallbackEncoder()

        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
        )
        self.centroid: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.threshold: float = sim_threshold
        self._check_latencies: List[float] = []

    def fit(self, clean_observations: List[str], save_path: str = "data/filter_fitted.pkl"):
        """
        Fit the filter on clean (unattacked) observation texts.

        Encodes all clean observations, fits IsolationForest,
        computes centroid for similarity scoring, and saves state.
        """
        print(f"[SemanticAnomalyFilter] Fitting on {len(clean_observations)} clean observations...")
        embeddings = self.encoder.encode(
            clean_observations,
            show_progress_bar=True,
            batch_size=32,
        )
        self.detector.fit(embeddings)
        self.centroid = np.mean(embeddings, axis=0)
        self.is_fitted = True

        # Save fitted state
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        try:
            with open(save_path, "wb") as f:
                pickle.dump({
                    "detector": self.detector,
                    "centroid": self.centroid,
                    "threshold": self.threshold,
                }, f)
            print(f"[SemanticAnomalyFilter] Saved fitted filter to {save_path}")
        except Exception as e:
            print(f"[SemanticAnomalyFilter] Warning: could not save filter: {e}")

    def check(self, obs_text: str) -> Tuple[bool, float]:
        """
        Check if an observation is anomalous.

        Returns:
            (is_anomaly, confidence)
            confidence = cosine similarity to centroid (higher = more trusted)
        """
        t0 = time.time()

        if not self.is_fitted:
            # Not fitted: trust everything (no false positives)
            latency_ms = (time.time() - t0) * 1000
            self._check_latencies.append(latency_ms)
            return False, 1.0

        emb = self.encoder.encode([obs_text], show_progress_bar=False)[0]

        # IsolationForest decision: negative score = anomaly
        iso_score = float(self.detector.decision_function([emb])[0])

        # Cosine similarity to centroid
        centroid_norm = np.linalg.norm(self.centroid)
        emb_norm = np.linalg.norm(emb)
        if centroid_norm > 0 and emb_norm > 0:
            cosine_sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
        else:
            cosine_sim = 0.0

        # Anomaly if IsolationForest says anomaly OR similarity too low
        is_anomaly = (iso_score < 0) or (cosine_sim < self.threshold)
        confidence = float(cosine_sim)

        latency_ms = (time.time() - t0) * 1000
        self._check_latencies.append(latency_ms)

        return is_anomaly, confidence

    def check_batch(self, obs_list: List[str]) -> List[Tuple[bool, float]]:
        """Batch check for efficiency — encodes all at once."""
        if not self.is_fitted:
            return [(False, 1.0)] * len(obs_list)

        t0 = time.time()
        embeddings = self.encoder.encode(obs_list, show_progress_bar=False, batch_size=32)

        results = []
        for emb in embeddings:
            iso_score = float(self.detector.decision_function([emb])[0])
            centroid_norm = np.linalg.norm(self.centroid)
            emb_norm = np.linalg.norm(emb)
            if centroid_norm > 0 and emb_norm > 0:
                cosine_sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
            else:
                cosine_sim = 0.0
            is_anomaly = (iso_score < 0) or (cosine_sim < self.threshold)
            results.append((is_anomaly, cosine_sim))

        total_ms = (time.time() - t0) * 1000
        per_ms = total_ms / len(obs_list)
        self._check_latencies.extend([per_ms] * len(obs_list))
        return results

    def auto_tune_threshold(
        self,
        val_clean: List[str],
        val_poisoned: List[str],
    ) -> Dict:
        """
        NOVEL: ROC-optimal threshold selection.

        Finds the cosine similarity threshold that minimizes FPR
        while keeping TPR >= 0.85.

        Args:
            val_clean: Clean validation observations (ground truth: not anomaly)
            val_poisoned: Poisoned validation observations (ground truth: anomaly)

        Returns:
            {'threshold': float, 'TPR': float, 'FPR': float}
        """
        if not self.is_fitted:
            raise RuntimeError("Filter must be fitted before auto_tune_threshold.")

        print(f"[SemanticAnomalyFilter] Auto-tuning threshold on "
              f"{len(val_clean)} clean / {len(val_poisoned)} poisoned samples...")

        all_texts = val_clean + val_poisoned
        labels = [0] * len(val_clean) + [1] * len(val_poisoned)  # 1 = anomaly

        embeddings = self.encoder.encode(all_texts, show_progress_bar=True, batch_size=32)

        # Compute cosine similarity scores (lower = more anomalous)
        scores = []
        for emb in embeddings:
            centroid_norm = np.linalg.norm(self.centroid)
            emb_norm = np.linalg.norm(emb)
            if centroid_norm > 0 and emb_norm > 0:
                sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
            else:
                sim = 0.0
            scores.append(sim)

        # ROC curve: higher threshold = more strict = higher TPR but also higher FPR
        # We negate scores so that "higher score → more likely anomaly"
        neg_scores = [-s for s in scores]
        labels_arr = np.array(labels)
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(labels_arr, neg_scores)

        # Find threshold with TPR >= 0.85 and minimum FPR
        best_threshold = self.threshold
        best_fpr = 1.0
        best_tpr = 0.0

        for fpr, tpr, thresh in zip(fpr_arr, tpr_arr, thresholds_arr):
            if tpr >= 0.85 and fpr < best_fpr:
                best_fpr = float(fpr)
                best_tpr = float(tpr)
                # Convert back from neg_score threshold to similarity threshold
                best_threshold = float(-thresh)

        self.threshold = best_threshold
        print(f"[SemanticAnomalyFilter] Auto-tuned threshold: {best_threshold:.4f} "
              f"(TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")

        return {
            "threshold": best_threshold,
            "TPR": best_tpr,
            "FPR": best_fpr,
        }

    def load(self, path: str = "data/filter_fitted.pkl"):
        """Load fitted state from pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.detector = state["detector"]
        self.centroid = state["centroid"]
        self.threshold = state.get("threshold", self.threshold)
        self.is_fitted = True
        print(f"[SemanticAnomalyFilter] Loaded filter from {path}")

    def get_stats(self) -> Dict:
        """Return latency statistics."""
        lats = self._check_latencies
        if not lats:
            return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "total_checks": 0}
        return {
            "mean_latency_ms": round(float(np.mean(lats)), 3),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
            "total_checks": len(lats),
            "is_fitted": self.is_fitted,
            "threshold": self.threshold,
        }
