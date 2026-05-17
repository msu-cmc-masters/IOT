from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


RANDOM_STATE = 42
DATASET_SLUG = "agungpambudi/network-malware-detection-connection-analysis"
LABEL_COLUMN = "label"
TARGET_COLUMN = "target"

RAW_NUMERIC_COLUMNS = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]
STRING_NUMERIC_COLUMNS = ["duration", "orig_bytes", "resp_bytes"]
IDENTIFIER_COLUMNS = [
    "uid",
    "id.orig_h",
    "id.resp_h",
    "tunnel_parents",
    "detailed-label",
]
EXPECTED_CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
]


@dataclass
class EvaluationResult:
    name: str
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_score: Optional[np.ndarray]


@dataclass
class AttackResult:
    epsilon: float
    success_rate: float
    success_count: int
    total_count: int
    average_l2: float
    adversarial_samples: np.ndarray


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], output_dim: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible wrapper around a PyTorch MLP."""

    def __init__(
        self,
        hidden_layer_sizes_grid: Sequence[Tuple[int, ...]] = ((100,), (100, 50), (200, 100)),
        learning_rates: Sequence[float] = (0.001, 0.01),
        max_epochs: int = 50,
        batch_size: int = 512,
        patience: int = 5,
        validation_size: float = 0.2,
        random_state: int = RANDOM_STATE,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.hidden_layer_sizes_grid = hidden_layer_sizes_grid
        self.learning_rates = learning_rates
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_size = validation_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _as_float32_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    @staticmethod
    def _load_state(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
        model.load_state_dict({key: value.to(device) for key, value in state.items()})

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = self._as_float32_array(X)
        y_np = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = X_np.shape[1]
        self.device_ = self._resolve_device()

        class_counts = Counter(y_np)
        stratify = y_np if len(class_counts) == 2 and min(class_counts.values()) >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_np,
            y_np,
            test_size=self.validation_size,
            stratify=stratify,
            random_state=self.random_state,
        )

        if self.verbose:
            print(f"[1.4] PyTorch MLP device: {self.device_}")

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        X_val_tensor = torch.from_numpy(X_val).to(self.device_)
        y_val_tensor = torch.from_numpy(y_val).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        best_overall_state: Optional[Dict[str, torch.Tensor]] = None
        best_overall_score = -np.inf
        best_overall_loss = np.inf
        best_overall_params: Dict[str, object] = {}

        for hidden_layers in self.hidden_layer_sizes_grid:
            for learning_rate in self.learning_rates:
                model = TorchMLP(self.n_features_in_, hidden_layers).to(self.device_)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                best_candidate_state: Optional[Dict[str, torch.Tensor]] = None
                best_candidate_loss = np.inf
                best_candidate_f1 = -np.inf
                best_candidate_epoch = 0
                stale_epochs = 0

                for epoch in range(1, self.max_epochs + 1):
                    model.train()
                    running_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(self.device_)
                        batch_y = batch_y.to(self.device_)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * batch_x.size(0)

                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_val_tensor)
                        val_loss = criterion(val_logits, y_val_tensor).item()
                        val_pred = val_logits.argmax(dim=1).cpu().numpy()
                    val_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

                    if val_loss < best_candidate_loss - 1e-5:
                        best_candidate_loss = val_loss
                        best_candidate_f1 = val_f1
                        best_candidate_epoch = epoch
                        best_candidate_state = self._state_to_cpu(model)
                        stale_epochs = 0
                    else:
                        stale_epochs += 1

                    if stale_epochs >= self.patience:
                        break

                if self.verbose:
                    print(
                        "[1.4] MLP candidate "
                        f"hidden_layers={hidden_layers}, lr={learning_rate}, "
                        f"best_epoch={best_candidate_epoch}, "
                        f"val_loss={best_candidate_loss:.4f}, val_f1={best_candidate_f1:.4f}"
                    )

                is_better = (
                    best_candidate_f1 > best_overall_score
                    or (
                        np.isclose(best_candidate_f1, best_overall_score)
                        and best_candidate_loss < best_overall_loss
                    )
                )
                if is_better and best_candidate_state is not None:
                    best_overall_state = best_candidate_state
                    best_overall_score = best_candidate_f1
                    best_overall_loss = best_candidate_loss
                    best_overall_params = {
                        "hidden_layers": hidden_layers,
                        "learning_rate": learning_rate,
                        "epochs": best_candidate_epoch,
                        "validation_f1_weighted": best_candidate_f1,
                    }

        if best_overall_state is None:
            raise RuntimeError("MLP training failed to produce a fitted model.")

        self.model_ = TorchMLP(
            self.n_features_in_,
            best_overall_params["hidden_layers"],  # type: ignore[arg-type]
        ).to(self.device_)
        self._load_state(self.model_, best_overall_state, self.device_)
        self.best_params_ = best_overall_params
        if self.verbose:
            print(f"[1.4] MLP best hyperparams: {self.best_params_}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["model_", "device_"])
        X_np = self._as_float32_array(X)
        if X_np.shape[0] == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float32)

        self.model_.eval()
        probabilities: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X_np.shape[0], self.batch_size * 8):
                batch = torch.from_numpy(X_np[start : start + self.batch_size * 8]).to(self.device_)
                logits = self.model_(batch)
                batch_prob = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(batch_prob)
        return np.vstack(probabilities)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def print_section(title: str, subtitle: Optional[str] = None) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if subtitle:
        print(subtitle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Network traffic malware detection: MLP/RF, permutation importance, FGSM."
    )
    parser.add_argument("--data_path", default="data", help="Directory containing pipe-delimited CSV files.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="FGSM perturbation strength. If passed explicitly, epsilon sweep is skipped.",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=100_000,
        help="Number of Benign and Malware rows to keep for modeling.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=250_000,
        help="Chunk size used while scanning large CSV files.",
    )
    parser.add_argument(
        "--permutation_repeats",
        type=int,
        default=10,
        help="Number of repeats for sklearn permutation_importance.",
    )
    parser.add_argument(
        "--fgsm_batch_size",
        type=int,
        default=8192,
        help="Batch size for FGSM gradient computation.",
    )
    args = parser.parse_args()
    args.epsilon_was_explicit = any(
        item == "--epsilon" or item.startswith("--epsilon=") for item in sys.argv[1:]
    )
    return args


def ensure_dataset(data_path: Path) -> List[Path]:
    data_path.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(data_path.glob("*.csv"))
    if csv_files:
        print(f"[1.1] Found {len(csv_files)} CSV files under {data_path}.")
        return csv_files

    print(f"[1.1] No CSV files found under {data_path}; downloading with kagglehub...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    copied = 0
    for source in downloaded_path.iterdir():
        if source.suffix.lower() == ".csv":
            destination = data_path / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
                copied += 1
                print(f"[1.1] Copied {source.name} -> {destination}")

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found after downloading {DATASET_SLUG}.")
    print(f"[1.1] Download complete; {copied} CSV files copied.")
    return csv_files


def update_reservoir(
    reservoir: Optional[pd.DataFrame],
    new_rows: pd.DataFrame,
    max_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if new_rows.empty:
        return reservoir if reservoir is not None else new_rows
    new_rows = new_rows.copy()
    new_rows["_sample_key"] = rng.random(new_rows.shape[0])
    combined = new_rows if reservoir is None else pd.concat([reservoir, new_rows], ignore_index=True)
    if combined.shape[0] > max_rows:
        combined = combined.nsmallest(max_rows, "_sample_key")
    return combined


def print_counter_distribution(counter: Counter, total: int, prefix: str) -> None:
    print(prefix)
    if total == 0:
        print("  <empty>")
        return
    for label, count in counter.most_common():
        print(f"  {label}: {count:,} ({count / total:.2%})")


def load_and_explore_data(data_path: str, sample_per_class: int, chunk_size: int) -> pd.DataFrame:
    data_dir = Path(data_path)
    csv_files = ensure_dataset(data_dir)
    rng = np.random.default_rng(RANDOM_STATE)

    total_rows = 0
    column_names: Optional[List[str]] = None
    dtype_map: Optional[pd.Series] = None
    missing_counts: Optional[pd.Series] = None
    label_counts: Counter = Counter()
    binary_counts: Counter = Counter()
    reservoirs: Dict[int, Optional[pd.DataFrame]] = {0: None, 1: None}

    print(f"[1.1] Reading {len(csv_files)} files in chunks of {chunk_size:,} rows...")
    for csv_file in csv_files:
        print(f"[1.1] Scanning {csv_file.name}")
        for chunk in pd.read_csv(csv_file, sep="|", chunksize=chunk_size, low_memory=False):
            if LABEL_COLUMN not in chunk.columns:
                raise ValueError(f"Column '{LABEL_COLUMN}' is missing in {csv_file}.")
            if column_names is None:
                column_names = list(chunk.columns)
                dtype_map = chunk.dtypes.astype(str)

            total_rows += chunk.shape[0]
            chunk_missing = chunk.isna().sum()
            missing_counts = (
                chunk_missing
                if missing_counts is None
                else missing_counts.add(chunk_missing, fill_value=0).astype(int)
            )

            raw_labels = chunk[LABEL_COLUMN].fillna("<missing>").astype(str)
            label_counts.update(raw_labels.value_counts().to_dict())
            binary_target = np.where(raw_labels.str.strip().eq("Benign"), 0, 1)
            binary_counts.update(Counter(binary_target))

            chunk[TARGET_COLUMN] = binary_target.astype(np.int64)
            for class_value in (0, 1):
                class_rows = chunk.loc[chunk[TARGET_COLUMN] == class_value]
                reservoirs[class_value] = update_reservoir(
                    reservoirs[class_value], class_rows, sample_per_class, rng
                )

    if column_names is None or dtype_map is None or missing_counts is None:
        raise ValueError("No data could be read from the CSV files.")

    print(f"[1.1] Merged {len(csv_files)} files, logical data shape: ({total_rows:,}, {len(column_names)})")
    print("[1.1] Column dtypes:")
    print(dtype_map.to_string())

    nonzero_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print("[1.1] Missing value counts:")
    if nonzero_missing.empty:
        print("  No missing values detected by pandas NA markers.")
    else:
        print(nonzero_missing.to_string())

    print_counter_distribution(
        label_counts,
        total_rows,
        "[1.1] Original label distribution (Benign vs. detailed malicious labels):",
    )
    print_counter_distribution(
        binary_counts,
        total_rows,
        "[1.1] Binarized class distribution: 0=Benign, 1=Malware",
    )

    sampled_parts = []
    for class_value, reservoir in reservoirs.items():
        if reservoir is None or reservoir.empty:
            continue
        sampled_parts.append(reservoir.drop(columns=["_sample_key"]))
        print(f"[1.2] Sampled class {class_value}: {reservoir.shape[0]:,} rows")

    if len(sampled_parts) < 2:
        raise ValueError("Both Benign and Malware samples are required for binary classification.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[1.2] After sampling, modeling data shape: {sampled_df.shape}")
    print_counter_distribution(
        Counter(sampled_df[TARGET_COLUMN].astype(int)),
        sampled_df.shape[0],
        "[1.2] Sampled class distribution:",
    )
    return sampled_df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    df = df.copy()
    print("[1.2] Starting preprocessing...")

    protected_columns = {LABEL_COLUMN, TARGET_COLUMN}
    missing_rate = df.isna().mean()
    high_missing_columns = [
        column for column, rate in missing_rate.items() if rate > 0.8 and column not in protected_columns
    ]
    if high_missing_columns:
        print(f"[1.2] Dropping high-missing columns (>80%): {high_missing_columns}")
    else:
        print("[1.2] No columns exceeded the 80% missing-rate threshold.")

    drop_columns = [
        column
        for column in high_missing_columns + IDENTIFIER_COLUMNS + [LABEL_COLUMN]
        if column in df.columns
    ]
    if drop_columns:
        print(f"[1.2] Dropping identifier/non-predictive columns: {drop_columns}")
        df = df.drop(columns=drop_columns)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'.")
    y = df.pop(TARGET_COLUMN).astype(np.int64).to_numpy()

    numeric_candidates = [column for column in RAW_NUMERIC_COLUMNS + STRING_NUMERIC_COLUMNS if column in df.columns]
    for column in numeric_candidates:
        df[column] = pd.to_numeric(df[column].replace("-", np.nan), errors="coerce")

    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for column in df.columns:
        if column in numeric_candidates or pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    for column in numeric_features:
        median = df[column].median()
        if pd.isna(median):
            median = 0.0
        df[column] = df[column].fillna(median)

    for column in categorical_features:
        mode_values = df[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
        df[column] = df[column].fillna(fill_value).astype(str)

    for column in categorical_features:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    print(f"[1.2] Numerical features: {numeric_features}")
    print(f"[1.2] Categorical features label-encoded: {categorical_features}")

    print("[1.2] IQR outlier clipping for numerical columns:")
    for column in numeric_features:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            print(f"  {column}: skipped (IQR=0)")
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[column] < lower) | (df[column] > upper)
        outlier_ratio = float(outlier_mask.mean())
        print(f"  {column}: outlier ratio={outlier_ratio:.2%}, clip=[{lower:.4f}, {upper:.4f}]")
        df[column] = df[column].clip(lower=lower, upper=upper)

    X = df.astype(np.float32)
    return X, y, numeric_features, categorical_features


def split_scale_and_balance(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_features: Sequence[str],
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    print("[1.4] Creating stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    numeric_columns = [column for column in numeric_features if column in X_train_scaled.columns]
    if numeric_columns:
        X_train_scaled.loc[:, numeric_columns] = scaler.fit_transform(X_train_scaled[numeric_columns])
        X_test_scaled.loc[:, numeric_columns] = scaler.transform(X_test_scaled[numeric_columns])
        print(f"[1.2] StandardScaler fitted on training numerical features: {numeric_columns}")

    class_counts = Counter(y_train)
    print_counter_distribution(class_counts, len(y_train), "[1.2] Training distribution before SMOTE:")
    if len(class_counts) == 2 and class_counts[0] != class_counts[1] and min(class_counts.values()) >= 2:
        print("[1.2] Applying SMOTE to the training split.")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        X_train_scaled = pd.DataFrame(X_resampled, columns=X_train_scaled.columns)
        y_train = np.asarray(y_resampled, dtype=np.int64)
    else:
        print("[1.2] SMOTE skipped: training split is already balanced or too small for resampling.")

    print_counter_distribution(Counter(y_train), len(y_train), "[1.2] Training distribution after SMOTE:")
    return (
        X_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        y_train,
        y_test,
        scaler,
    )


def target_correlations(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    values = {}
    for column in X.columns:
        series = X[column].to_numpy(dtype=np.float64)
        if np.std(series) == 0:
            values[column] = 0.0
            continue
        corr = np.corrcoef(series, y)[0, 1]
        values[column] = 0.0 if np.isnan(corr) else abs(float(corr))
    return pd.Series(values)


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    print("[1.3] Computing feature correlation matrix...")
    absolute_corr = X_train.corr(numeric_only=True).abs()
    target_corr = target_correlations(X_train, y_train)
    columns = list(absolute_corr.columns)
    to_drop = set()
    high_corr_pairs = []

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            corr_value = absolute_corr.loc[left, right]
            if corr_value > correlation_threshold:
                drop_column = left if target_corr[left] < target_corr[right] else right
                to_drop.add(drop_column)
                high_corr_pairs.append((left, right, corr_value, drop_column))

    if high_corr_pairs:
        print(f"[1.3] Highly correlated feature pairs (|corr|>{correlation_threshold}):")
        for left, right, corr_value, drop_column in high_corr_pairs:
            print(f"  {left} vs {right}: corr={corr_value:.4f}; drop={drop_column}")
    else:
        print(f"[1.3] No feature pairs exceeded |corr|>{correlation_threshold}.")

    if to_drop:
        X_train = X_train.drop(columns=sorted(to_drop))
        X_test = X_test.drop(columns=sorted(to_drop))
        print(f"[1.3] Dropped correlated features: {sorted(to_drop)}")

    print("[1.3] Running Random-Forest-based SelectFromModel...")
    selector_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector_forest.fit(X_train, y_train)
    mean_importance = float(np.mean(selector_forest.feature_importances_))
    threshold = 0.5 * mean_importance
    selector = SelectFromModel(selector_forest, threshold=threshold, prefit=True)
    support_mask = selector.get_support()
    if not support_mask.any():
        best_index = int(np.argmax(selector_forest.feature_importances_))
        support_mask[best_index] = True
        print("[1.3] SelectFromModel selected no features; keeping the strongest RF feature.")

    selected_features = X_train.columns[support_mask].tolist()
    dropped_by_selector = X_train.columns[~support_mask].tolist()
    print(f"[1.3] RF importance threshold: {threshold:.6f}")
    print(f"[1.3] Selected features ({len(selected_features)}): {selected_features}")
    if dropped_by_selector:
        print(f"[1.3] Dropped by SelectFromModel: {dropped_by_selector}")

    return X_train[selected_features], X_test[selected_features], selected_features


def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray) -> GridSearchCV:
    print("[1.4] Training Random Forest with GridSearchCV (3-fold)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"[1.4] RF best params: {grid.best_params_}")
    print(f"[1.4] RF best CV weighted F1: {grid.best_score_:.4f}")
    return grid


def safe_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_class1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_class1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def plot_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = cm.max() / 2.0 if cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[1.5] Saved confusion matrix: {output_path}")


def evaluate_classifier(name: str, model, X_test: pd.DataFrame, y_test: np.ndarray, plots_dir: Path) -> EvaluationResult:
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] >= 2:
            y_score = probabilities[:, 1]

    metrics = compute_metrics(y_test, y_pred)
    print(
        f"[1.5] {name} -> "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision(macro)={metrics['precision_macro']:.4f}, Recall(macro)={metrics['recall_macro']:.4f}, "
        f"F1(macro)={metrics['f1_macro']:.4f}, "
        f"Precision(weighted)={metrics['precision_weighted']:.4f}, "
        f"Recall(weighted)={metrics['recall_weighted']:.4f}, F1(weighted)={metrics['f1_weighted']:.4f}"
    )
    print(f"[1.5] {name} classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            zero_division=0,
        )
    )
    plot_confusion_matrix_png(
        y_test,
        y_pred,
        f"Confusion Matrix - {name}",
        plots_dir / f"confusion_matrix_{safe_slug(name)}.png",
    )
    return EvaluationResult(name=name, metrics=metrics, y_pred=y_pred, y_score=y_score)


def plot_roc_curves(results: Sequence[EvaluationResult], y_test: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results:
        if result.y_score is None or len(np.unique(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, result.y_score)
        roc_auc = auc(fpr, tpr)
        result.metrics["auc"] = roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"{result.name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[1.5] Saved ROC curve comparison: {output_path}")


def compute_permutation(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    repeats: int,
    n_jobs: int,
) -> object:
    print(f"[2] Computing permutation importance for {name} with n_repeats={repeats}...")
    return permutation_importance(
        estimator,
        X,
        y,
        n_repeats=repeats,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def top_importance_rows(importance_result, feature_names: Sequence[str], top_n: int = 10) -> List[Tuple[str, float, float]]:
    means = importance_result.importances_mean
    stds = importance_result.importances_std
    order = np.argsort(means)[::-1][:top_n]
    return [(feature_names[index], float(means[index]), float(stds[index])) for index in order]


def print_top_importances(title: str, rows: Sequence[Tuple[str, float, float]]) -> None:
    print(title)
    for index, (feature, mean_value, std_value) in enumerate(rows, start=1):
        print(f"  {index:2d}. {feature:<24} {mean_value:.6f} +/- {std_value:.6f}")


def plot_permutation_importance(
    importance_result,
    feature_names: Sequence[str],
    title: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    rows = top_importance_rows(importance_result, feature_names, top_n=top_n)
    labels = [row[0] for row in rows][::-1]
    means = [row[1] for row in rows][::-1]
    stds = [row[2] for row in rows][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.42 * len(labels) + 1.5)))
    ax.barh(labels, means, xerr=stds, color="#4472C4", alpha=0.88)
    ax.set_xlabel("Accuracy decrease after permutation")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[2] Saved permutation importance plot: {output_path}")


def plot_importance_comparison(
    mlp_rows: Sequence[Tuple[str, float, float]],
    rf_rows: Sequence[Tuple[str, float, float]],
    output_path: Path,
) -> None:
    ordered_features = []
    for feature, _, _ in list(mlp_rows) + list(rf_rows):
        if feature not in ordered_features:
            ordered_features.append(feature)

    mlp_map = {feature: mean for feature, mean, _ in mlp_rows}
    rf_map = {feature: mean for feature, mean, _ in rf_rows}
    y_positions = np.arange(len(ordered_features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(ordered_features) + 1.5)))
    ax.barh(y_positions - width / 2, [mlp_map.get(f, 0.0) for f in ordered_features], width, label="MLP")
    ax.barh(y_positions + width / 2, [rf_map.get(f, 0.0) for f in ordered_features], width, label="Random Forest")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_features)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance")
    ax.set_title("MLP vs Random Forest Permutation Importance Top Features")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[2.4] Saved feature importance comparison: {output_path}")


def fgsm_attack_malware(
    mlp: TorchMLPClassifier,
    X_malware: np.ndarray,
    epsilon: float,
    batch_size: int,
) -> AttackResult:
    check_is_fitted(mlp, ["model_", "device_"])
    if X_malware.shape[0] == 0:
        return AttackResult(epsilon, 0.0, 0, 0, 0.0, X_malware.copy())

    model = mlp.model_
    device = mlp.device_
    model.eval()
    criterion = nn.CrossEntropyLoss()

    adversarial_batches = []
    successful = 0
    total = 0
    l2_values = []

    for start in range(0, X_malware.shape[0], batch_size):
        batch_np = X_malware[start : start + batch_size].astype(np.float32, copy=False)
        batch_x = torch.tensor(batch_np, dtype=torch.float32, device=device, requires_grad=True)
        target_benign = torch.zeros(batch_x.shape[0], dtype=torch.long, device=device)

        model.zero_grad(set_to_none=True)
        logits = model(batch_x)
        target_loss = criterion(logits, target_benign)
        target_loss.backward()

        # This is targeted FGSM: minimize class-0 loss to push malware toward Benign.
        adversarial_x = batch_x - epsilon * batch_x.grad.sign()
        with torch.no_grad():
            adversarial_logits = model(adversarial_x)
            adversarial_pred = adversarial_logits.argmax(dim=1)
            perturbation = adversarial_x - batch_x
            batch_l2 = torch.sqrt(torch.sum(perturbation * perturbation, dim=1))

        successful += int((adversarial_pred == 0).sum().item())
        total += batch_x.shape[0]
        l2_values.append(batch_l2.detach().cpu().numpy())
        adversarial_batches.append(adversarial_x.detach().cpu().numpy())

    adversarial_samples = np.vstack(adversarial_batches)
    l2_array = np.concatenate(l2_values)
    success_rate = successful / total if total else 0.0
    average_l2 = float(np.mean(l2_array)) if l2_array.size else 0.0
    return AttackResult(epsilon, success_rate, successful, total, average_l2, adversarial_samples)


def choose_best_attack(results: Sequence[AttackResult]) -> AttackResult:
    above_half = [result for result in results if result.success_rate > 0.5]
    if above_half:
        return sorted(above_half, key=lambda item: item.epsilon)[0]
    return max(results, key=lambda item: item.success_rate)


def plot_attack_success(results: Sequence[AttackResult], output_path: Path) -> None:
    epsilons = [result.epsilon for result in results]
    success_rates = [result.success_rate * 100 for result in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epsilons, success_rates, marker="o", color="#C00000")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title("FGSM Attack Success Rate vs epsilon")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[3.4] Saved FGSM success-rate chart: {output_path}")


def evaluate_attack(
    mlp: TorchMLPClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    epsilon_values: Sequence[float],
    batch_size: int,
    plots_dir: Path,
) -> Tuple[List[AttackResult], AttackResult, Dict[str, float], np.ndarray]:
    X_test_np = X_test.to_numpy(dtype=np.float32)
    malware_indices = np.where(y_test == 1)[0]
    X_malware = X_test_np[malware_indices]

    results = []
    print("[3.3] FGSM epsilon results:")
    for epsilon in epsilon_values:
        result = fgsm_attack_malware(mlp, X_malware, epsilon, batch_size)
        results.append(result)
        print(
            f"  epsilon={epsilon:.4f} -> Attack Success: {result.success_rate:.2%} "
            f"({result.success_count:,}/{result.total_count:,}), avg L2={result.average_l2:.6f}"
        )

    best_result = choose_best_attack(results)
    plot_attack_success(results, plots_dir / "fgsm_success_rate.png")
    print(
        f"[3.4] Best epsilon={best_result.epsilon:.4f}, "
        f"Attack Success={best_result.success_rate:.2%} "
        f"({best_result.success_count:,}/{best_result.total_count:,})"
    )
    print(f"[3.4] Average L2 perturbation: {best_result.average_l2:.6f}")

    X_after_attack = X_test_np.copy()
    X_after_attack[malware_indices] = best_result.adversarial_samples
    y_pred_after = mlp.predict(X_after_attack)
    after_metrics = compute_metrics(y_test, y_pred_after)
    print(
        "[3.4] Post-attack MLP metrics -> "
        f"Accuracy={after_metrics['accuracy']:.4f}, "
        f"Precision(class=1)={after_metrics['precision_class1']:.4f}, "
        f"Recall(class=1)={after_metrics['recall_class1']:.4f}, "
        f"F1(class=1)={after_metrics['f1_class1']:.4f}"
    )
    plot_confusion_matrix_png(
        y_test,
        y_pred_after,
        "Confusion Matrix - MLP After FGSM",
        plots_dir / "confusion_matrix_after_fgsm.png",
    )
    return results, best_result, after_metrics, y_pred_after


def print_before_after_comparison(before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> float:
    before_f1 = before_metrics["f1_class1"]
    after_f1 = after_metrics["f1_class1"]
    f1_drop_pct = ((before_f1 - after_f1) / before_f1 * 100.0) if before_f1 > 0 else 0.0
    print("[3.5] Before-vs-after comparison for MLP:")
    print("         Before Attack    After Attack")
    print(f"  Acc    {before_metrics['accuracy']:.4f}           {after_metrics['accuracy']:.4f}")
    print(f"  Prec   {before_metrics['precision_class1']:.4f}           {after_metrics['precision_class1']:.4f}")
    print(f"  Recall {before_metrics['recall_class1']:.4f}           {after_metrics['recall_class1']:.4f}")
    print(f"  F1     {before_f1:.4f}           {after_f1:.4f}")
    print(
        f"[3.5] Conclusion: FGSM reduced class-1 MLP F1 from {before_f1:.4f} "
        f"to {after_f1:.4f}, a drop of {f1_drop_pct:.2f}%."
    )
    return f1_drop_pct


def fmt_metric(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def plot_summary_table(
    mlp_result: EvaluationResult,
    rf_result: EvaluationResult,
    after_metrics: Dict[str, float],
    best_attack: AttackResult,
    f1_drop_pct: float,
    mlp_top_rows: Sequence[Tuple[str, float, float]],
    rf_top_rows: Sequence[Tuple[str, float, float]],
    plots_dir: Path,
) -> None:
    mlp_top = [row[0] for row in mlp_top_rows[:10]]
    rf_top = [row[0] for row in rf_top_rows[:10]]
    intersection_count = len(set(mlp_top).intersection(rf_top))

    best_model = (
        mlp_result
        if mlp_result.metrics["f1_weighted"] >= rf_result.metrics["f1_weighted"]
        else rf_result
    )
    rows = [
        ["Model Performance Comparison", "", "", ""],
        [
            "Accuracy",
            fmt_metric(mlp_result.metrics["accuracy"]),
            fmt_metric(rf_result.metrics["accuracy"]),
            fmt_metric(after_metrics["accuracy"]),
        ],
        [
            "Precision (weighted)",
            fmt_metric(mlp_result.metrics["precision_weighted"]),
            fmt_metric(rf_result.metrics["precision_weighted"]),
            fmt_metric(after_metrics["precision_weighted"]),
        ],
        [
            "Recall (weighted)",
            fmt_metric(mlp_result.metrics["recall_weighted"]),
            fmt_metric(rf_result.metrics["recall_weighted"]),
            fmt_metric(after_metrics["recall_weighted"]),
        ],
        [
            "F1-score (weighted)",
            fmt_metric(mlp_result.metrics["f1_weighted"]),
            fmt_metric(rf_result.metrics["f1_weighted"]),
            fmt_metric(after_metrics["f1_weighted"]),
        ],
        ["Adversarial Attack Impact", "", "", ""],
        ["Best epsilon", f"{best_attack.epsilon:.4f}", "", f"{best_attack.epsilon:.4f}"],
        ["Attack success rate", f"{best_attack.success_rate:.2%}", "", f"{best_attack.success_rate:.2%}"],
        ["MLP F1 drop percentage", f"{f1_drop_pct:.2f}%", "", f"{f1_drop_pct:.2f}%"],
        ["Feature Importance Top-10", "", "", ""],
        ["MLP/RF intersection count", f"{intersection_count}/10", f"{intersection_count}/10", ""],
        ["MLP Top-3 feature names", ", ".join(mlp_top[:3]), "", ""],
        ["RF Top-3 feature names", "", ", ".join(rf_top[:3]), ""],
        ["Conclusion", "", "", ""],
        [
            "Best model and F1",
            best_model.name if best_model.name == "MLP" else "",
            best_model.name if best_model.name != "MLP" else "",
            fmt_metric(best_model.metrics["f1_weighted"]),
        ],
        ["MLP robustness assessment", f"F1 drop {f1_drop_pct:.2f}%", "", "FGSM vulnerable"],
    ]

    fig_height = max(8, 0.42 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "MLP", "Random Forest", "MLP (after attack)"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(col=list(range(4)))

    header_color = "#4472C4"
    section_color = "#D9E2F3"
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if row_index == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif rows[row_index - 1][1:] == ["", "", ""]:
            cell.set_facecolor(section_color)
            cell.get_text().set_weight("bold")

    ax.set_title(
        "Network Traffic Malware Detection — Comprehensive Analysis Summary",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    fig.tight_layout()
    output_path = plots_dir / "summary_table.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"[Summary] Summary table saved: {output_path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print_section(
        "Part 1: Classifier (MLP vs Random Forest)",
        "Binary labels are Benign=0 and Malware=1. The MLP is differentiable; RF is the control baseline.",
    )
    df = load_and_explore_data(args.data_path, args.sample_per_class, args.chunk_size)
    X, y, numeric_features, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test, _ = split_scale_and_balance(
        X, y, numeric_features, args.test_size
    )
    X_train, X_test, selected_features = select_features(X_train, X_test, y_train)

    mlp = TorchMLPClassifier(verbose=True)
    mlp.fit(X_train, y_train)
    rf_grid = train_random_forest(X_train, y_train)
    rf = rf_grid.best_estimator_

    mlp_result = evaluate_classifier("MLP", mlp, X_test, y_test, plots_dir)
    rf_result = evaluate_classifier("Random Forest", rf, X_test, y_test, plots_dir)
    plot_roc_curves([mlp_result, rf_result], y_test, plots_dir / "roc_curves.png")

    best_model = (
        mlp_result
        if mlp_result.metrics["f1_weighted"] >= rf_result.metrics["f1_weighted"]
        else rf_result
    )
    f1_gap = abs(mlp_result.metrics["f1_weighted"] - rf_result.metrics["f1_weighted"])
    print(f"[1.5] Best model: {best_model.name} (weighted F1 gap={f1_gap:.4f})")

    print_section(
        "Part 2: Permutation Importance Explanation",
        "Permutation Importance shuffles one feature at a time and measures the accuracy drop.",
    )
    mlp_perm_train = compute_permutation(
        "MLP train set", mlp, X_train, y_train, args.permutation_repeats, n_jobs=1
    )
    mlp_perm_test = compute_permutation(
        "MLP test set", mlp, X_test, y_test, args.permutation_repeats, n_jobs=1
    )
    rf_perm_test = compute_permutation(
        "Random Forest test set", rf, X_test, y_test, args.permutation_repeats, n_jobs=-1
    )

    plot_permutation_importance(
        mlp_perm_train,
        selected_features,
        "MLP Permutation Importance - Train Set",
        plots_dir / "permutation_importance_mlp_train.png",
    )
    plot_permutation_importance(
        mlp_perm_test,
        selected_features,
        "MLP Permutation Importance - Test Set",
        plots_dir / "permutation_importance_mlp_test.png",
    )
    plot_permutation_importance(
        rf_perm_test,
        selected_features,
        "Random Forest Permutation Importance - Test Set",
        plots_dir / "permutation_importance_rf_test.png",
    )

    mlp_top_rows = top_importance_rows(mlp_perm_test, selected_features, top_n=10)
    rf_top_rows = top_importance_rows(rf_perm_test, selected_features, top_n=10)
    print_top_importances("[2.2] MLP Top-10 key features (test permutation importance):", mlp_top_rows)
    print_top_importances("[2.3] RF Top-10 key features (test permutation importance):", rf_top_rows)

    mlp_top_features = [row[0] for row in mlp_top_rows]
    rf_top_features = [row[0] for row in rf_top_rows]
    intersection = sorted(set(mlp_top_features).intersection(rf_top_features))
    only_mlp = [feature for feature in mlp_top_features if feature not in rf_top_features]
    only_rf = [feature for feature in rf_top_features if feature not in mlp_top_features]
    print(f"[2.4] MLP and RF Top-10 intersection: {len(intersection)}/10 -> {intersection}")
    print(f"[2.4] MLP-only Top-10 features: {only_mlp}")
    print(f"[2.4] RF-only Top-10 features: {only_rf}")
    if len(intersection) == 10:
        print("[2.4] The two models rely on highly consistent feature rankings.")
    else:
        print(
            "[2.4] The models overlap but are not identical, suggesting the neural network and "
            "tree ensemble use partly different decision surfaces."
        )
    plot_importance_comparison(mlp_top_rows, rf_top_rows, plots_dir / "permutation_importance_comparison.png")

    print_section(
        "Part 3: FGSM White-Box Adversarial Attack",
        "FGSM uses input gradients from the MLP. For a targeted Benign attack, the code minimizes class-0 loss.",
    )
    epsilon_values = [args.epsilon] if args.epsilon_was_explicit else [0.01, 0.05, 0.10, 0.20]
    _, best_attack, after_metrics, _ = evaluate_attack(
        mlp,
        X_test,
        y_test,
        epsilon_values,
        args.fgsm_batch_size,
        plots_dir,
    )
    f1_drop_pct = print_before_after_comparison(mlp_result.metrics, after_metrics)

    print_section("Summary Comparison Table")
    plot_summary_table(
        mlp_result,
        rf_result,
        after_metrics,
        best_attack,
        f1_drop_pct,
        mlp_top_rows,
        rf_top_rows,
        plots_dir,
    )
    print(f"[Done] All plots are saved in: {plots_dir}")


if __name__ == "__main__":
    main()
