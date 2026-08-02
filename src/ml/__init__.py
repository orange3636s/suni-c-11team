"""제조 공정 회귀 모델 학습 도구."""

import os

# Configure joblib before NumPy/scikit-learn imports. Newer Windows versions
# can lack WMIC, which otherwise produces a harmless but noisy core-count
# warning. Training sections also enforce explicit threadpool limits.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.ml.dataset import ALLOWED_TARGETS, RANDOM_STATE

__all__ = ["ALLOWED_TARGETS", "RANDOM_STATE"]
