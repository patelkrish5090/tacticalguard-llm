"""
Defense Layer 1: Semantic Anomaly Filter.

Uses sentence embeddings (all-MiniLM-L6-v2) + IsolationForest to detect
poisoned observations before they reach the LLM.

Target latency: <15ms per check on CPU.
Novel feature: ROC-optimal threshold auto-tuning (auto_tune_threshold).
"""

import os
import pickle
import time
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. SemanticAnomalyFilter disabled.", stacklevel=2)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    warnings.warn("sentence-transformers not available. SemanticAnomalyFilter disabled.", stacklevel=2)


class SemanticAnomalyFilter:
    """
    Detects poisoned observations via semantic embedding anomaly detection.

    Pipeline:
      obs_text -> SentenceTransformer embedding -> IsolationForest + cosine sim
                 -> (is_anomaly: bool, confidence: float)

    The confidence score (cosine similarity to clean centroid) is passed to
    ProvenancePromptBuilder to update agent reliability scores.
    """

    ENCODER_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        contamination: float = 0.05,
        sim_threshold: float = 0.82,
    ):
        if not (SKLEARN_AVAILABLE and SBERT_AVAILABLE):
            raise RuntimeError(
                "scikit-learn and sentence-transformers are required for "
                "SemanticAnomalyFilter. pip install scikit-learn sentence-transformers"
            )

        self.contamination = contamination
        self.threshold = sim_threshold

        logger.info(f"[AnomalyFilter] Loading encoder: {self.ENCODER_MODEL}")
        self.encoder = SentenceTransformer(self.ENCODER_MODEL)
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
        )

        self.centroid: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self._check_latencies: list[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        clean_observations: list[str],
        save_path: str = "data/filter_fitted.pkl",
    ) -> None:
        """
        Fit the IsolationForest on clean observation embeddings.
        Computes centroid for cosine similarity threshold check.
        """
        logger.info(f"[AnomalyFilter] Fitting on {len(clean_observations)} clean obs…")

        embeddings = self.encoder.encode(
            clean_observations,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        )

        self.detector.fit(embeddings)
        self.centroid = embeddings.mean(axis=0)
        self.is_fitted = True

        # Persist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "detector": self.detector,
                "centroid": self.centroid,
                "threshold": self.threshold,
                "contamination": self.contamination,
            }, f)
        logger.info(f"[AnomalyFilter] Saved to {save_path}")

    def load(self, path: str) -> None:
        """Load a previously fitted filter state."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.detector = state["detector"]
        self.centroid = state["centroid"]
        self.threshold = state.get("threshold", self.threshold)
        self.contamination = state.get("contamination", self.contamination)
        self.is_fitted = True
        logger.info(f"[AnomalyFilter] Loaded from {path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def check(self, obs_text: str) -> tuple[bool, float]:
        """
        Check a single observation for anomaly.

        Returns:
            (is_anomaly, confidence)
            confidence = cosine similarity to clean centroid (higher = more trusted)
        """
        if not self.is_fitted:
            # Not fitted yet — pass everything through
            return False, 1.0

        t0 = time.perf_counter()

        emb = self.encoder.encode([obs_text], convert_to_numpy=True)[0]

        iso_score = self.detector.decision_function([emb])[0]

        # Cosine similarity to centroid
        centroid_norm = np.linalg.norm(self.centroid)
        emb_norm = np.linalg.norm(emb)
        if centroid_norm > 0 and emb_norm > 0:
            cosine_sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
        else:
            cosine_sim = 0.0

        is_anomaly = (iso_score < 0) or (cosine_sim < self.threshold)
        confidence = cosine_sim  # higher = more trusted

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._check_latencies.append(elapsed_ms)

        return is_anomaly, confidence

    def check_batch(self, obs_list: list[str]) -> list[tuple[bool, float]]:
        """Batch check for efficiency (encodes all at once)."""
        if not self.is_fitted or not obs_list:
            return [(False, 1.0)] * len(obs_list)

        t0 = time.perf_counter()
        embeddings = self.encoder.encode(obs_list, convert_to_numpy=True, batch_size=64)

        iso_scores = self.detector.decision_function(embeddings)
        centroid_norm = np.linalg.norm(self.centroid)

        results = []
        for emb, iso_score in zip(embeddings, iso_scores):
            emb_norm = np.linalg.norm(emb)
            if centroid_norm > 0 and emb_norm > 0:
                cosine_sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
            else:
                cosine_sim = 0.0
            is_anomaly = (iso_score < 0) or (cosine_sim < self.threshold)
            results.append((is_anomaly, cosine_sim))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Amortize latency per item
        per_item = elapsed_ms / len(obs_list)
        self._check_latencies.extend([per_item] * len(obs_list))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # NOVEL: ROC-optimal threshold auto-tuning
    # ──────────────────────────────────────────────────────────────────────────

    def auto_tune_threshold(
        self,
        val_clean: list[str],
        val_poisoned: list[str],
        min_tpr: float = 0.85,
    ) -> dict:
        """
        Find optimal cosine-similarity threshold via ROC curve.

        Minimizes FPR while keeping TPR >= min_tpr.
        Updates self.threshold in place.

        Returns:
            {'threshold': float, 'TPR': float, 'FPR': float}
        """
        if not self.is_fitted:
            raise RuntimeError("Filter must be fitted before auto-tuning.")

        all_obs = val_clean + val_poisoned
        labels = [0] * len(val_clean) + [1] * len(val_poisoned)  # 1 = poisoned

        embeddings = self.encoder.encode(all_obs, convert_to_numpy=True, batch_size=64)
        centroid_norm = np.linalg.norm(self.centroid)

        # Score = cosine similarity (higher -> more normal)
        # We want to classify low similarity as anomaly -> score = 1 - cosine_sim
        scores = []
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if centroid_norm > 0 and emb_norm > 0:
                cosine_sim = float(np.dot(emb, self.centroid) / (emb_norm * centroid_norm))
            else:
                cosine_sim = 0.0
            scores.append(1.0 - cosine_sim)  # higher score = more anomalous

        fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)

        # Find threshold that achieves TPR >= min_tpr with minimum FPR
        # (thresholds here are on 1-cosine_sim scale)
        best_threshold = self.threshold  # default
        best_fpr = 1.0
        best_tpr = 0.0

        for fpr, tpr, thresh in zip(fpr_arr, tpr_arr, thresholds):
            if tpr >= min_tpr and fpr < best_fpr:
                best_fpr = float(fpr)
                best_tpr = float(tpr)
                best_threshold = float(1.0 - thresh)  # convert back to cosine_sim threshold

        self.threshold = max(0.0, min(1.0, best_threshold))
        logger.info(
            f"[AnomalyFilter] Auto-tuned threshold={self.threshold:.4f} "
            f"TPR={best_tpr:.3f} FPR={best_fpr:.3f}"
        )

        return {
            "threshold": self.threshold,
            "TPR": best_tpr,
            "FPR": best_fpr,
        }

    def get_stats(self) -> dict:
        """Return latency and check statistics."""
        if not self._check_latencies:
            return {
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_checks": 0,
                "threshold": self.threshold,
                "is_fitted": self.is_fitted,
            }
        lats = np.array(self._check_latencies)
        return {
            "mean_latency_ms": float(lats.mean()),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "total_checks": len(self._check_latencies),
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
        }
