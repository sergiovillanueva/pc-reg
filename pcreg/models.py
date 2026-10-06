"""PC-Reg and the baselines of the paper, on cached patch features (N, P, D) and CLS attention (N, P).

Every fitted parameter is obtained in closed form from the normal training images (PCA, ridge regression,
Ledoit-Wolf covariances, means and standard deviations); there is no gradient-based optimization.

Section 3 of the paper:
  * context     c_i = attention-weighted (or uniform) mean of the neighbors within Chebyshev radius R   (Eq. 1)
  * regression  W_i = argmin sum ||f_i - W c_i||^2 + lambda ||W||_F^2, one per grid position           (Eq. 2)
  * distance    d_i = Mahalanobis distance of the residual f_i - W_i c_i (Ledoit-Wolf)                   (Eq. 3)
  * positional  S_pos = P95_i of (d_i - m_i) / s_i, with m_i, s_i from the training maps                  (Eq. 4)
  * compositional  S_comp = squared Mahalanobis distance of the concatenated quadrant mean features
  * fusion      S = alpha z_pos + (1 - alpha) z_comp, z-scores from the training scores                  (Eq. 5)
"""
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from .config import ALPHA, LAMBDA, PCA_DIM, PCA_SEED, PERCENTILE, RADIUS


# ----------------------------------------------------------------------------- features
def fit_pca(train_raw: np.ndarray, dim: int = PCA_DIM) -> PCA:
    """PCA fitted on all training patches of the category."""
    return PCA(n_components=dim, random_state=PCA_SEED).fit(train_raw.reshape(-1, train_raw.shape[-1]))


def project(pca: PCA, arr: np.ndarray) -> np.ndarray:
    n, p, d = arr.shape
    return pca.transform(arr.reshape(-1, d)).reshape(n, p, -1).astype(np.float32)


def neighbors(grid: int, radius: int = RADIUS) -> list[np.ndarray]:
    """Positions within Chebyshev distance `radius`, excluding the position itself."""
    out = []
    for py in range(grid):
        for px in range(grid):
            out.append(np.array([qy * grid + qx
                                 for qy in range(max(0, py - radius), min(grid, py + radius + 1))
                                 for qx in range(max(0, px - radius), min(grid, px + radius + 1))
                                 if (qy, qx) != (py, px)]))
    return out


def _lw(x: np.ndarray):
    lw = LedoitWolf().fit(x)
    return lw.location_, lw.precision_


def _maha(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    c = x - mu
    return np.einsum("nd,de,ne->n", c, prec, c)


def standardized_p95(maps: np.ndarray, train_maps: np.ndarray) -> np.ndarray:
    """Per-position standardization with the training maps, then the 95th percentile over positions (Eq. 4)."""
    m, s = train_maps.mean(0), train_maps.std(0) + 1e-6
    return np.percentile((maps - m) / s, PERCENTILE, axis=1)


def zscore(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (x - ref.mean()) / (ref.std() + 1e-8)


# ----------------------------------------------------------------------------- PC-Reg
class PCReg:
    """Per-position ridge regression of the patch feature on its neighbor context, Mahalanobis residual distance.

    weighting = "cls" uses the CLS-to-patch attention of each image as neighbor weights (PC-Reg_CLS);
    weighting = "uniform" uses the plain neighbor mean (PC-Reg)."""

    def __init__(self, grid: int, weighting: str = "cls", radius: int = RADIUS, lam: float = LAMBDA):
        self.grid, self.weighting, self.lam = grid, weighting, lam
        self.nbrs = neighbors(grid, radius)

    def _context(self, f: np.ndarray, p: int, attn: np.ndarray | None) -> np.ndarray:
        nf = f[:, self.nbrs[p], :]
        if self.weighting == "uniform":
            return nf.mean(axis=1)
        w = attn[:, self.nbrs[p]]
        w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
        return (w[:, :, None] * nf).sum(axis=1)

    def fit(self, train: np.ndarray, attn: np.ndarray | None = None) -> "PCReg":
        n, P, D = train.shape
        eye = np.eye(D)
        self.W, self.mu, self.prec = [], [], []
        for p in range(P):
            X = self._context(train, p, attn)
            Y = train[:, p, :]
            W = np.linalg.solve(X.T @ X + self.lam * eye, X.T @ Y)  # closed-form ridge
            r = Y - X @ W
            lw = LedoitWolf().fit(r)
            self.W.append(W); self.mu.append(r.mean(axis=0)); self.prec.append(lw.precision_)
        return self

    def distance_maps(self, feats: np.ndarray, attn: np.ndarray | None = None) -> np.ndarray:
        """d_i = sqrt of the squared Mahalanobis distance of the residual, shape (N, P)."""
        n, P, _ = feats.shape
        out = np.zeros((n, P))
        for p in range(P):
            c = (feats[:, p, :] - self._context(feats, p, attn) @ self.W[p]) - self.mu[p]
            out[:, p] = np.sqrt(np.maximum(0, np.sum(c @ self.prec[p] * c, axis=1)))
        return out


class QuadrantScore:
    """Compositional branch: Ledoit-Wolf Gaussian on the concatenated mean features of the four quadrants."""

    def __init__(self, grid: int):
        self.grid = grid

    def descriptor(self, f: np.ndarray) -> np.ndarray:
        n, _, d = f.shape
        h = self.grid // 2
        return f.reshape(n, 2, h, 2, h, d).mean(axis=(2, 4)).reshape(n, 4 * d).astype(np.float64)

    def fit(self, train: np.ndarray) -> "QuadrantScore":
        self.mu, self.prec = _lw(self.descriptor(train))
        return self

    def score(self, f: np.ndarray) -> np.ndarray:
        return _maha(self.descriptor(f), self.mu, self.prec)  # squared distance


class GlobalMeanScore:
    """Ledoit-Wolf Mahalanobis distance of the global mean feature (the 'global mean' rows of Tables 2 and 3)."""

    def fit(self, train: np.ndarray) -> "GlobalMeanScore":
        self.mu, self.prec = _lw(train.mean(axis=1).astype(np.float64))
        return self

    def score(self, f: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(0, _maha(f.mean(axis=1).astype(np.float64), self.mu, self.prec)))


class PCRegDual:
    """Full detector: PC-Reg_CLS positional score + quadrant compositional score, z-fused with training statistics."""

    def __init__(self, grid: int, alpha: float = ALPHA):
        self.reg, self.quad, self.alpha = PCReg(grid, "cls"), QuadrantScore(grid), alpha

    def fit(self, train: np.ndarray, attn: np.ndarray) -> "PCRegDual":
        self.reg.fit(train, attn)
        self.quad.fit(train)
        self.train_maps = self.reg.distance_maps(train, attn)              # in-sample, as in the paper
        self.train_pos = standardized_p95(self.train_maps, self.train_maps)
        self.train_comp = self.quad.score(train)
        return self

    def branch_scores(self, feats: np.ndarray, attn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return standardized_p95(self.reg.distance_maps(feats, attn), self.train_maps), self.quad.score(feats)

    def score(self, feats: np.ndarray, attn: np.ndarray) -> np.ndarray:
        pos, comp = self.branch_scores(feats, attn)
        return self.alpha * zscore(pos, self.train_pos) + (1 - self.alpha) * zscore(comp, self.train_comp)


# ----------------------------------------------------------------------------- baselines on the same features
class PerPositionGaussian:
    """PaDiM (residual = feature) or MeanSub (residual = feature - uniform neighbor mean): Ledoit-Wolf per position."""

    def __init__(self, grid: int, mean_subtraction: bool = False, radius: int = RADIUS):
        self.mean_subtraction = mean_subtraction
        self.nbrs = neighbors(grid, radius)

    def _residual(self, f: np.ndarray) -> np.ndarray:
        if not self.mean_subtraction:
            return f.astype(np.float64)
        return np.stack([f[:, p, :] - f[:, self.nbrs[p], :].mean(axis=1) for p in range(f.shape[1])], axis=1).astype(np.float64)

    def fit(self, train: np.ndarray) -> "PerPositionGaussian":
        r = self._residual(train)
        self.mu, self.prec = zip(*[_lw(r[:, p, :]) for p in range(r.shape[1])])
        self.train_maps = self.distance_maps(train)
        return self

    def distance_maps(self, f: np.ndarray) -> np.ndarray:
        r = self._residual(f)
        return np.sqrt(np.maximum(0, np.stack([_maha(r[:, p, :], self.mu[p], self.prec[p]) for p in range(r.shape[1])], axis=1)))


class PatchCore:
    """Exact k = 1 nearest neighbor in the memory bank of all training patches (no coreset), Euclidean distance.
    Image score: maximum over patches (original rule, without the reweighting step) or P95."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train: np.ndarray) -> "PatchCore":
        self.bank = torch.from_numpy(train.reshape(-1, train.shape[-1]).astype(np.float32)).to(self.device)
        self.bank_sq = (self.bank * self.bank).sum(1)
        return self

    @torch.no_grad()
    def distance_maps(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros(f.shape[:2], dtype=np.float32)
        for i in range(f.shape[0]):
            q = torch.from_numpy(f[i].astype(np.float32)).to(self.device)
            d2 = (q * q).sum(1, keepdim=True) - 2 * q @ self.bank.T + self.bank_sq[None, :]
            out[i] = torch.sqrt(torch.clamp(d2.min(1).values, min=0)).cpu().numpy()
        return out


# ----------------------------------------------------------------------------- metrics
def auroc(good: np.ndarray, anomalous: np.ndarray) -> float:
    y = np.r_[np.zeros(len(good)), np.ones(len(anomalous))]
    return float(roc_auc_score(y, np.r_[good, anomalous]))


def evaluate(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Image AUROC; on MVTec LOCO AD also logical and structural (each against all normal test images)."""
    g = scores["test_good"]
    if "test_logical" in scores:
        return {"auroc": auroc(g, np.r_[scores["test_logical"], scores["test_structural"]]),
                "logical": auroc(g, scores["test_logical"]), "structural": auroc(g, scores["test_structural"])}
    return {"auroc": auroc(g, scores["test_anomaly"]), "logical": np.nan, "structural": np.nan}
