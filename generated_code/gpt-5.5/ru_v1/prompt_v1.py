from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from textwrap import fill
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
_cache_dir = os.path.join(tempfile.gettempdir(), "iot_malware_detection_cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_dir, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_cache_dir, "xdg"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from matplotlib import font_manager
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
from torch.utils.data import DataLoader, TensorDataset


DATASET_SLUG = "agungpambudi/network-malware-detection-connection-analysis"

LABEL_COLUMN = "label"
ID_COLUMNS = ["uid", "id.orig_h", "id.resp_h", "tunnel_parents", "detailed-label"]
TEXT_NUMERIC_COLUMNS = ["duration", "orig_bytes", "resp_bytes"]
KNOWN_NUMERIC_COLUMNS = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]
KNOWN_CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
]


@dataclass
class DataProfile:
    file_count: int
    total_rows: int
    columns: list[str]
    dtypes: pd.Series
    missing_counts: pd.Series
    label_counts: pd.Series


@dataclass
class AttackResult:
    epsilon: float
    success_rate: float
    successful: int
    total: int
    mean_l2: float
    x_adv_malicious: np.ndarray


def configure_matplotlib() -> None:
    preferred_fonts = [
        "PingFang SC",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    available_fonts = [font for font in preferred_fonts if font in installed_fonts]
    plt.rcParams["font.sans-serif"] = available_fonts or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(PLOTS_DIR, exist_ok=True)


def set_global_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def find_csv_files(data_path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))


def ensure_dataset(data_path: str) -> list[str]:
    os.makedirs(data_path, exist_ok=True)
    csv_files = find_csv_files(data_path)
    if csv_files:
        return csv_files

    print("[1.1] CSV-файлы не найдены, загружаем датасет через kagglehub...")
    import kagglehub

    downloaded_path = kagglehub.dataset_download(DATASET_SLUG)
    copied = 0
    for root, _, files in os.walk(downloaded_path):
        for file_name in files:
            if not file_name.endswith(".csv"):
                continue
            src = os.path.join(root, file_name)
            dst = os.path.join(data_path, file_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
                print(f"[1.1] Скопирован {file_name} -> {data_path}/")

    csv_files = find_csv_files(data_path)
    if not csv_files:
        raise FileNotFoundError(
            f"После загрузки не найдено CSV-файлов в директории {data_path!r}."
        )
    print(f"[1.1] Скопировано CSV-файлов: {copied}")
    return csv_files


def add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.copy()
    return left.add(right, fill_value=0)


def load_profile_and_balanced_sample(
    csv_files: list[str],
    sample_per_class: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, DataProfile]:
    rng = np.random.default_rng(random_state)
    reservoirs: dict[int, pd.DataFrame] = {0: pd.DataFrame(), 1: pd.DataFrame()}
    total_rows = 0
    missing_counts: pd.Series | None = None
    label_counts: pd.Series | None = None
    first_dtypes: pd.Series | None = None
    first_columns: list[str] | None = None

    print(
        "[1.1] Читаем CSV-файлы потоково: считаем профиль и формируем "
        "сбалансированную выборку."
    )
    for file_index, csv_file in enumerate(csv_files, start=1):
        print(f"[1.1] Файл {file_index}/{len(csv_files)}: {os.path.basename(csv_file)}")
        reader = pd.read_csv(csv_file, sep="|", chunksize=chunksize, low_memory=False)
        for chunk in reader:
            if LABEL_COLUMN not in chunk.columns:
                raise ValueError(f"В файле {csv_file} нет столбца {LABEL_COLUMN!r}.")

            if first_dtypes is None:
                first_dtypes = chunk.dtypes.astype(str)
                first_columns = list(chunk.columns)

            total_rows += len(chunk)
            missing_counts = add_series(missing_counts, chunk.isna().sum())

            label_text = chunk[LABEL_COLUMN].fillna("MISSING").astype(str)
            label_counts = add_series(label_counts, label_text.value_counts())
            binary_label = np.where(label_text.str.strip().eq("Benign"), 0, 1)

            chunk = chunk.copy()
            chunk["binary_label"] = binary_label
            for class_value in (0, 1):
                subset = chunk.loc[chunk["binary_label"] == class_value].copy()
                if subset.empty:
                    continue
                subset["__sample_key__"] = rng.random(len(subset))
                reservoir = pd.concat(
                    [reservoirs[class_value], subset], ignore_index=True
                )
                if len(reservoir) > sample_per_class:
                    reservoir = reservoir.nsmallest(sample_per_class, "__sample_key__")
                reservoirs[class_value] = reservoir.reset_index(drop=True)

    if first_dtypes is None or first_columns is None:
        raise ValueError("CSV-файлы пустые, невозможно сформировать датасет.")

    profile = DataProfile(
        file_count=len(csv_files),
        total_rows=total_rows,
        columns=first_columns,
        dtypes=first_dtypes,
        missing_counts=missing_counts.fillna(0).astype("int64"),
        label_counts=label_counts.fillna(0).astype("int64").sort_values(ascending=False),
    )

    if reservoirs[0].empty or reservoirs[1].empty:
        raise ValueError(
            "Нужны оба класса для бинарной классификации, но один из классов пуст."
        )

    data = pd.concat([reservoirs[0], reservoirs[1]], ignore_index=True)
    data = data.drop(columns=["__sample_key__"], errors="ignore")
    data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return data, profile


def print_profile(profile: DataProfile, sample_df: pd.DataFrame) -> None:
    print(
        f"[1.1] Объединено {profile.file_count} файлов, размер данных: "
        f"({profile.total_rows}, {len(profile.columns)})"
    )
    print("[1.1] Типы столбцов:")
    print(profile.dtypes.to_string())
    print("[1.1] Пропущенные значения:")
    print(profile.missing_counts.sort_values(ascending=False).to_string())

    label_distribution = pd.DataFrame(
        {
            "count": profile.label_counts,
            "share": profile.label_counts / max(profile.total_rows, 1),
        }
    )
    print("[1.1] Исходное распределение label:")
    print(label_distribution.to_string(formatters={"share": "{:.2%}".format}))

    benign_total = int(
        sum(
            count
            for label, count in profile.label_counts.items()
            if str(label).strip() == "Benign"
        )
    )
    malicious_total = int(profile.total_rows - benign_total)
    full_binary_summary = pd.DataFrame(
        {
            "count": pd.Series({0: benign_total, 1: malicious_total}),
            "share": pd.Series(
                {
                    0: benign_total / max(profile.total_rows, 1),
                    1: malicious_total / max(profile.total_rows, 1),
                }
            ),
        }
    )
    print("[1.1] Бинаризованное распределение во всем датасете:")
    print(full_binary_summary.to_string(formatters={"share": "{:.2%}".format}))

    binary_counts = sample_df["binary_label"].value_counts().sort_index()
    binary_shares = sample_df["binary_label"].value_counts(normalize=True).sort_index()
    binary_summary = pd.DataFrame({"count": binary_counts, "share": binary_shares})
    print("[1.1] Бинаризованное распределение в рабочей выборке:")
    print(binary_summary.to_string(formatters={"share": "{:.2%}".format}))


def fill_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    if pd.isna(median):
        median = 0.0
    return numeric.fillna(median)


def fill_categorical(series: pd.Series) -> pd.Series:
    series = series.astype("object")
    mode = series.dropna().mode()
    fill_value = mode.iloc[0] if not mode.empty else "missing"
    return series.fillna(fill_value).astype(str)


def preprocess_data(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    print("[1.2] Предобработка: очистка, кодирование, обработка выбросов.")
    y = sample_df["binary_label"].astype(int).to_numpy()
    data = sample_df.drop(columns=["binary_label"], errors="ignore").copy()

    missing_ratio = data.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.80].index.tolist()
    if high_missing_cols:
        print(f"[1.2] Удалены столбцы с пропусками > 80%: {high_missing_cols}")
        data = data.drop(columns=high_missing_cols)

    drop_cols = [LABEL_COLUMN, *ID_COLUMNS]
    existing_drop_cols = [col for col in drop_cols if col in data.columns]
    if existing_drop_cols:
        print(f"[1.2] Удалены идентификаторы/служебные метки: {existing_drop_cols}")
        data = data.drop(columns=existing_drop_cols)

    for col in TEXT_NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = data[col].replace("-", np.nan)
            data[col] = fill_numeric(data[col])

    for col in KNOWN_NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = fill_numeric(data[col])

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        data[col] = fill_numeric(data[col])

    categorical_cols = data.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        data[col] = fill_categorical(data[col])
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    outlier_cells = 0
    checked_cells = 0
    clipped_cols: list[str] = []
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (data[col] < lower) | (data[col] > upper)
        outlier_cells += int(mask.sum())
        checked_cells += len(data)
        if mask.any():
            clipped_cols.append(col)
            data[col] = data[col].clip(lower=lower, upper=upper)

    outlier_share = outlier_cells / checked_cells if checked_cells else 0.0
    print(
        f"[1.2] Доля экстремальных выбросов по IQR: {outlier_share:.2%}; "
        f"обрезаны столбцы: {clipped_cols or 'нет'}"
    )

    data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    print(f"[1.2] После семплирования и предобработки размер: {data.shape}")
    print(
        "[1.2] Распределение после семплирования: "
        f"0={int((y == 0).sum())}, 1={int((y == 1).sum())}"
    )
    return data.astype("float32"), y, {"encoders": encoders, "outlier_share": outlier_share}


def safe_target_correlation(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    correlations: dict[str, float] = {}
    y_std = np.std(y)
    for col in X.columns:
        values = X[col].to_numpy()
        if np.std(values) == 0 or y_std == 0:
            correlations[col] = 0.0
            continue
        corr = np.corrcoef(values, y)[0, 1]
        correlations[col] = 0.0 if np.isnan(corr) else abs(float(corr))
    return pd.Series(correlations)


def remove_highly_correlated_features(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.95,
) -> pd.DataFrame:
    print("[1.3] Ищем пары признаков с |корреляцией| > 0.95.")
    if X.shape[1] <= 1:
        return X

    corr_matrix = X.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = safe_target_correlation(X, y)
    to_drop: set[str] = set()
    correlated_pairs: list[tuple[str, str, float]] = []

    for col in upper.columns:
        correlated_rows = upper.index[upper[col] > threshold].tolist()
        for row in correlated_rows:
            if row in to_drop or col in to_drop:
                continue
            correlated_pairs.append((row, col, float(upper.loc[row, col])))
            drop_col = row if target_corr[row] < target_corr[col] else col
            to_drop.add(drop_col)

    if correlated_pairs:
        print(f"[1.3] Найдено сильно коррелирующих пар: {len(correlated_pairs)}")
        print(f"[1.3] Удалены признаки: {sorted(to_drop)}")
    else:
        print("[1.3] Сильно коррелирующих пар не найдено.")
    return X.drop(columns=sorted(to_drop), errors="ignore")


def select_features_with_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
) -> pd.DataFrame:
    print("[1.3] SelectFromModel на основе Random Forest.")
    if X.shape[1] <= 1:
        return X

    selector_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector_model.fit(X, y)
    selector = SelectFromModel(selector_model, threshold="0.5*mean", prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    if not selected_features:
        print("[1.3] SelectFromModel не выбрал признаки; оставляем все признаки.")
        selected_features = X.columns.tolist()

    print(
        f"[1.3] Признаков до отбора: {X.shape[1]}, после отбора: "
        f"{len(selected_features)}"
    )
    print(f"[1.3] Выбранные признаки: {selected_features}")
    return X[selected_features]


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (100,),
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        batch_size: int = 512,
        patience: int = 5,
        device_name: str = "cpu",
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device_name = device_name
        self.random_state = random_state
        self.verbose = verbose

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TorchMLPClassifier":
        set_global_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = X.shape[1]

        if X_val is None or y_val is None:
            stratify = y if np.min(np.bincount(y)) >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=stratify,
                random_state=self.random_state,
            )
        else:
            X_train = np.asarray(X, dtype=np.float32)
            y_train = np.asarray(y, dtype=np.int64)
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        self.model_ = TorchMLP(self.n_features_in_, self.hidden_layers).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_val_accuracy = 0.0
        wait = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_pred = val_logits.argmax(dim=1)
                val_accuracy = (val_pred == y_val_tensor).float().mean().item()

            if self.verbose:
                print(
                    f"    epoch={epoch:02d}, val_loss={val_loss:.5f}, "
                    f"val_acc={val_accuracy:.4f}"
                )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch
                best_val_accuracy = val_accuracy
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = best_loss
        self.best_val_accuracy_ = best_val_accuracy
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("MLP-модель еще не обучена.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.as_tensor(
                    X[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = self.model_(batch)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probabilities.append(probs)
        return np.vstack(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


def get_torch_device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_mlp_with_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device_name: str,
    random_state: int,
) -> TorchMLPClassifier:
    hidden_layer_options = [(100,), (100, 50), (200, 100)]
    learning_rates = [0.001, 0.01]
    stratify = y_train if np.min(np.bincount(y_train)) >= 2 else None
    X_core, X_val, y_core, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=stratify,
        random_state=random_state,
    )

    best_candidate: dict | None = None
    print(f"[1.4] Обучаем MLP на устройстве: {device_name}")
    for hidden_layers in hidden_layer_options:
        for learning_rate in learning_rates:
            print(
                f"[1.4] MLP trial: hidden_layers={hidden_layers}, "
                f"lr={learning_rate}"
            )
            candidate = TorchMLPClassifier(
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                num_epochs=50,
                batch_size=512,
                patience=5,
                device_name=device_name,
                random_state=random_state,
                verbose=False,
            )
            candidate.fit(X_core, y_core, X_val=X_val, y_val=y_val)
            score = candidate.best_val_accuracy_
            print(
                f"[1.4]   val_accuracy={score:.4f}, "
                f"best_epoch={candidate.best_epoch_}"
            )
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "score": score,
                    "hidden_layers": hidden_layers,
                    "learning_rate": learning_rate,
                    "best_epoch": candidate.best_epoch_,
                }

    assert best_candidate is not None
    print(
        "[1.4] MLP лучшие гиперпараметры: "
        f"hidden_layers={best_candidate['hidden_layers']}, "
        f"lr={best_candidate['learning_rate']}, "
        f"epochs={best_candidate['best_epoch']}"
    )

    final_model = TorchMLPClassifier(
        hidden_layers=best_candidate["hidden_layers"],
        learning_rate=best_candidate["learning_rate"],
        num_epochs=50,
        batch_size=512,
        patience=5,
        device_name=device_name,
        random_state=random_state,
        verbose=False,
    )
    final_model.fit(X_train, y_train)
    return final_model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> RandomForestClassifier:
    print("[1.4] Обучаем Random Forest с GridSearchCV.")
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    grid = GridSearchCV(
        estimator=base_model,
        param_grid={"n_estimators": [100, 200], "max_depth": [10, 20, None]},
        scoring="f1_weighted",
        cv=3,
        n_jobs=1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"[1.4] RF лучшие параметры: {grid.best_params_}")
    return grid.best_estimator_


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_class_1": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def plot_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    file_name: str,
) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, file_name)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_model(
    name: str,
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    confusion_file: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    y_pred = estimator.predict(X_test)
    metrics = compute_binary_metrics(y_test, y_pred)
    print(
        f"[1.5] {name} -> Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision(macro): {metrics['precision_macro']:.4f}, "
        f"Recall(macro): {metrics['recall_macro']:.4f}, "
        f"F1(macro): {metrics['f1_macro']:.4f}, "
        f"F1(weighted): {metrics['f1_weighted']:.4f}"
    )
    print(f"[1.5] Classification report для {name}:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            zero_division=0,
        )
    )
    plot_path = plot_confusion_matrix_png(
        y_test, y_pred, f"{name}: confusion matrix", confusion_file
    )
    print(f"[1.5] Матрица ошибок сохранена: {plot_path}")

    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    return metrics, y_pred, y_score


def plot_roc_curves(
    roc_items: Iterable[tuple[str, np.ndarray | None]],
    y_test: np.ndarray,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    auc_scores: dict[str, float] = {}
    for name, y_score in roc_items:
        if y_score is None or len(np.unique(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)
        auc_scores[name] = auc_score
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC-кривые: MLP vs Random Forest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[1.5] ROC-кривые сохранены: {path}")
    return auc_scores


def importance_to_frame(result, feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importance_bar(
    importance_df: pd.DataFrame,
    title: str,
    file_name: str,
    top_n: int = 20,
) -> str:
    top = importance_df.head(top_n).iloc[::-1]
    fig_height = max(4.8, 0.34 * len(top) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title(title)
    ax.set_xlabel("Среднее снижение accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, file_name)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_and_plot_permutation_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    split_name: str,
    file_name: str,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    print(
        f"[2] Permutation Importance: {model_name}, выборка={split_name}, "
        f"n_repeats={n_repeats}"
    )
    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
        n_jobs=1,
    )
    frame = importance_to_frame(result, feature_names)
    plot_path = plot_importance_bar(
        frame,
        f"{model_name}: Permutation Importance ({split_name})",
        file_name,
    )
    print(f"[2] График сохранен: {plot_path}")
    print(f"[2] {model_name} Top-10 признаков ({split_name}):")
    for rank, row in frame.head(10).iterrows():
        print(
            f"  {rank + 1}. {row['feature']} "
            f"({row['importance_mean']:.5f} ± {row['importance_std']:.5f})"
        )
    return frame


def plot_importance_comparison(
    mlp_importance: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> str:
    mlp_top = mlp_importance.head(10)["feature"].tolist()
    rf_top = rf_importance.head(10)["feature"].tolist()
    ordered_features = list(dict.fromkeys([*mlp_top, *rf_top]))
    mlp_map = dict(zip(mlp_importance["feature"], mlp_importance["importance_mean"]))
    rf_map = dict(zip(rf_importance["feature"], rf_importance["importance_mean"]))
    x = np.arange(len(ordered_features))
    width = 0.38

    fig_width = max(9.0, 0.42 * len(ordered_features) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    ax.bar(
        x - width / 2,
        [mlp_map.get(feature, 0.0) for feature in ordered_features],
        width,
        label="MLP",
    )
    ax.bar(
        x + width / 2,
        [rf_map.get(feature, 0.0) for feature in ordered_features],
        width,
        label="Random Forest",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_features, rotation=45, ha="right")
    ax.set_ylabel("Среднее снижение accuracy")
    ax.set_title("Сравнение Permutation Importance: MLP vs Random Forest")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "permutation_importance_comparison.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def targeted_fgsm(
    mlp: TorchMLPClassifier,
    X: np.ndarray,
    epsilon: float,
    batch_size: int = 8192,
) -> np.ndarray:
    model = mlp.model_
    model.eval()
    criterion = nn.CrossEntropyLoss()
    adversarial_batches: list[np.ndarray] = []

    for start in range(0, len(X), batch_size):
        batch_np = np.asarray(X[start : start + batch_size], dtype=np.float32)
        batch = torch.as_tensor(batch_np, dtype=torch.float32, device=mlp.device)
        batch.requires_grad_(True)
        target = torch.zeros(len(batch_np), dtype=torch.long, device=mlp.device)

        model.zero_grad(set_to_none=True)
        logits = model(batch)
        target_loss = criterion(logits, target)
        target_loss.backward()

        # Targeted FGSM minimizes loss for class 0, hence the negative gradient sign.
        adv_batch = batch - epsilon * batch.grad.sign()
        adversarial_batches.append(adv_batch.detach().cpu().numpy())

    return np.vstack(adversarial_batches)


def run_fgsm_experiments(
    mlp: TorchMLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: list[float],
) -> tuple[list[AttackResult], AttackResult | None, np.ndarray | None]:
    malicious_mask = y_test == 1
    X_malicious = X_test[malicious_mask]
    if len(X_malicious) == 0:
        print("[3.2] В тестовом наборе нет вредоносных образцов, FGSM пропущен.")
        return [], None, None

    results: list[AttackResult] = []
    print("[3.3] Результаты FGSM:")
    for epsilon in epsilons:
        X_adv = targeted_fgsm(mlp, X_malicious, epsilon=epsilon)
        adv_pred = mlp.predict(X_adv)
        successful = int((adv_pred == 0).sum())
        total = len(X_malicious)
        success_rate = successful / total
        mean_l2 = float(np.linalg.norm(X_adv - X_malicious, axis=1).mean())
        result = AttackResult(
            epsilon=epsilon,
            success_rate=success_rate,
            successful=successful,
            total=total,
            mean_l2=mean_l2,
            x_adv_malicious=X_adv,
        )
        results.append(result)
        print(
            f"  ε={epsilon:.3f} -> Успешность: {success_rate:.2%} "
            f"({successful}/{total}), среднее L2={mean_l2:.4f}"
        )

    above_threshold = [result for result in results if result.success_rate > 0.50]
    if above_threshold:
        best = sorted(above_threshold, key=lambda item: item.epsilon)[0]
    else:
        best = max(results, key=lambda item: item.success_rate)

    X_test_adv = X_test.copy()
    X_test_adv[malicious_mask] = best.x_adv_malicious
    return results, best, X_test_adv


def plot_fgsm_success(results: list[AttackResult]) -> str | None:
    if not results:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(
        [result.epsilon for result in results],
        [result.success_rate * 100 for result in results],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("ε")
    ax.set_ylabel("Успешность атаки, %")
    ax.set_title("FGSM: успешность атаки от ε")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fgsm_success_rate.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[3.4] График успешности FGSM сохранен: {path}")
    return path


def save_summary_table(
    mlp_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    attacked_metrics: dict[str, float] | None,
    best_attack: AttackResult | None,
    mlp_importance: pd.DataFrame,
    rf_importance: pd.DataFrame,
    best_model_name: str,
    best_model_f1: float,
) -> str:
    attacked_metrics = attacked_metrics or {}
    attack_success = best_attack.success_rate if best_attack else 0.0
    best_epsilon = best_attack.epsilon if best_attack else 0.0
    mlp_f1_drop = mlp_metrics["f1_class_1"] - attacked_metrics.get(
        "f1_class_1", mlp_metrics["f1_class_1"]
    )
    mlp_f1_drop_pct = mlp_f1_drop / max(mlp_metrics["f1_class_1"], 1e-12)

    mlp_top10 = mlp_importance.head(10)["feature"].tolist()
    rf_top10 = rf_importance.head(10)["feature"].tolist()
    intersection_count = len(set(mlp_top10) & set(rf_top10))

    rows = [
        ["Сравнение производительности моделей", "", "", ""],
        [
            "Accuracy",
            f"{mlp_metrics['accuracy']:.4f}",
            f"{rf_metrics['accuracy']:.4f}",
            f"{attacked_metrics.get('accuracy', 0.0):.4f}",
        ],
        [
            "Precision",
            f"{mlp_metrics['precision_class_1']:.4f}",
            f"{rf_metrics['precision_class_1']:.4f}",
            f"{attacked_metrics.get('precision_class_1', 0.0):.4f}",
        ],
        [
            "Recall",
            f"{mlp_metrics['recall_class_1']:.4f}",
            f"{rf_metrics['recall_class_1']:.4f}",
            f"{attacked_metrics.get('recall_class_1', 0.0):.4f}",
        ],
        [
            "F1-score",
            f"{mlp_metrics['f1_class_1']:.4f}",
            f"{rf_metrics['f1_class_1']:.4f}",
            f"{attacked_metrics.get('f1_class_1', 0.0):.4f}",
        ],
        ["Влияние состязательной атаки", "", "", ""],
        ["Лучший ε", f"{best_epsilon:.3f}", "", f"{best_epsilon:.3f}"],
        ["Успешность атаки", "", "", f"{attack_success:.2%}"],
        ["Падение F1 MLP", f"{mlp_f1_drop_pct:.2%}", "", f"{mlp_f1_drop_pct:.2%}"],
        ["Важность признаков Top-10", "", "", ""],
        [f"Пересечение MLP ∩ RF", f"{intersection_count}/10", f"{intersection_count}/10", ""],
        ["Top-3 MLP", fill(", ".join(mlp_top10[:3]), width=28), "", ""],
        ["Top-3 RF", "", fill(", ".join(rf_top10[:3]), width=28), ""],
        ["Вывод", "", "", ""],
        [
            "Лучшая модель и устойчивость",
            f"{best_model_name}, F1={best_model_f1:.4f}",
            "",
            f"Падение F1 после FGSM: {mlp_f1_drop_pct:.2%}",
        ],
    ]
    columns = ["Метрика", "MLP", "Random Forest", "MLP (после атаки)"]

    fig_height = max(7.0, 0.42 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(12.6, fig_height))
    ax.axis("off")
    ax.set_title(
        "Network Traffic Malware Detection — Comprehensive Analysis Summary",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.32, 0.22, 0.22, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)

    section_titles = {
        "Сравнение производительности моделей",
        "Влияние состязательной атаки",
        "Важность признаков Top-10",
        "Вывод",
    }

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            row_title = rows[row - 1][0]
            if row_title in section_titles:
                cell.set_facecolor("#D9E2F3")
                cell.set_text_props(fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F7F9FC")

    path = os.path.join(PLOTS_DIR, "summary_table.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Summary] Сводная таблица сохранена: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Network malware detection: MLP, RF, permutation importance, FGSM."
    )
    parser.add_argument(
        "--data_path",
        default="data",
        help="Путь к директории с CSV-файлами датасета.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Доля тестового набора.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Сила возмущения FGSM. Если явно указана, сканирование ε пропускается.",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=100_000,
        help="Количество записей каждого класса в рабочей выборке.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Размер чанка при потоковом чтении CSV.",
    )
    parser.add_argument(
        "--importance_repeats",
        type=int,
        default=10,
        help="Количество повторов для permutation_importance.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Фиксированный seed для воспроизводимости.",
    )
    return parser.parse_args()


def epsilon_was_provided() -> bool:
    return any(arg == "--epsilon" or arg.startswith("--epsilon=") for arg in sys.argv[1:])


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    set_global_seed(args.random_state)

    print("=" * 40)
    print("Часть 1: Классификатор (MLP vs Random Forest)")
    print("=" * 40)
    print(
        "Идея: сравниваем дифференцируемую нейросетевую модель MLP с "
        "древовидным Random Forest для бинарной классификации трафика."
    )
    csv_files = ensure_dataset(args.data_path)
    sample_df, profile = load_profile_and_balanced_sample(
        csv_files=csv_files,
        sample_per_class=args.sample_per_class,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    print_profile(profile, sample_df)

    X, y, _ = preprocess_data(sample_df)
    X = remove_highly_correlated_features(X, y)
    X = select_features_with_random_forest(X, y, random_state=args.random_state)
    feature_names = X.columns.tolist()

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=stratify,
        random_state=args.random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    print("[1.2] StandardScaler обучен на train и применен к train/test.")

    class_counts = np.bincount(y_train)
    if len(class_counts) == 2 and class_counts.min() > 5:
        smote = SMOTE(random_state=args.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(
            "[1.2] После SMOTE train-распределение: "
            f"0={int((y_train_balanced == 0).sum())}, "
            f"1={int((y_train_balanced == 1).sum())}"
        )
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        print("[1.2] SMOTE пропущен: недостаточно образцов миноритарного класса.")

    device_name = get_torch_device_name()
    mlp = train_mlp_with_search(
        X_train_balanced,
        y_train_balanced,
        device_name=device_name,
        random_state=args.random_state,
    )
    rf = train_random_forest(X_train_balanced, y_train_balanced, args.random_state)

    mlp_metrics, _, mlp_score = evaluate_model(
        "MLP", mlp, X_test, y_test, "confusion_matrix_mlp.png"
    )
    rf_metrics, _, rf_score = evaluate_model(
        "Random Forest",
        rf,
        X_test,
        y_test,
        "confusion_matrix_random_forest.png",
    )
    auc_scores = plot_roc_curves(
        [("MLP", mlp_score), ("Random Forest", rf_score)],
        y_test,
    )
    for name, auc_score in auc_scores.items():
        print(f"[1.5] {name} AUC: {auc_score:.4f}")

    if rf_metrics["f1_weighted"] > mlp_metrics["f1_weighted"]:
        best_model_name = "Random Forest"
        best_model_f1 = rf_metrics["f1_weighted"]
    else:
        best_model_name = "MLP"
        best_model_f1 = mlp_metrics["f1_weighted"]
    print(f"[1.5] Лучшая модель: {best_model_name} (F1 weighted={best_model_f1:.4f})")

    print("\n" + "=" * 40)
    print("Часть 2: Объяснение Permutation Importance")
    print("=" * 40)
    print(
        "Идея: перемешиваем один признак и измеряем падение accuracy; "
        "чем сильнее падение, тем важнее признак."
    )
    mlp_importance_train = compute_and_plot_permutation_importance(
        mlp,
        X_train,
        y_train,
        feature_names,
        "MLP",
        "train",
        "permutation_importance_mlp_train.png",
        args.importance_repeats,
        args.random_state,
    )
    mlp_importance_test = compute_and_plot_permutation_importance(
        mlp,
        X_test,
        y_test,
        feature_names,
        "MLP",
        "test",
        "permutation_importance_mlp_test.png",
        args.importance_repeats,
        args.random_state,
    )
    rf_importance_test = compute_and_plot_permutation_importance(
        rf,
        X_test,
        y_test,
        feature_names,
        "Random Forest",
        "test",
        "permutation_importance_rf_test.png",
        args.importance_repeats,
        args.random_state,
    )

    mlp_top10 = set(mlp_importance_test.head(10)["feature"])
    rf_top10 = set(rf_importance_test.head(10)["feature"])
    intersection = sorted(mlp_top10 & rf_top10)
    only_mlp = sorted(mlp_top10 - rf_top10)
    only_rf = sorted(rf_top10 - mlp_top10)
    print(f"[2.4] Пересечение Top-10 MLP и RF: {len(intersection)}/10")
    print(f"[2.4] Общие признаки: {intersection}")
    print(f"[2.4] Только MLP: {only_mlp}")
    print(f"[2.4] Только RF: {only_rf}")
    comparison_path = plot_importance_comparison(mlp_importance_test, rf_importance_test)
    print(f"[2.4] Сравнительный график сохранен: {comparison_path}")
    print(
        "[2.4] Анализ: полное совпадение Top-10 означало бы близкие правила "
        "принятия решений; различия показывают, что MLP и RF могут использовать "
        "разные нелинейные структуры в одних и тех же сетевых признаках."
    )

    print("\n" + "=" * 40)
    print("Часть 3: FGSM белая состязательная атака")
    print("=" * 40)
    print(
        "Идея: FGSM использует градиент MLP по входу и добавляет малое "
        "возмущение, чтобы вредоносный трафик был принят за Benign."
    )
    epsilons = [args.epsilon] if epsilon_was_provided() else [0.01, 0.05, 0.1, 0.2]
    if epsilon_was_provided():
        print(f"[3.3] --epsilon указан явно, используем только ε={args.epsilon:.3f}")
    else:
        print(f"[3.3] Сканируем ε: {epsilons}")

    attack_results, best_attack, X_test_adv = run_fgsm_experiments(mlp, X_test, y_test, epsilons)
    plot_fgsm_success(attack_results)

    attacked_metrics = None
    if best_attack is not None and X_test_adv is not None:
        print(
            f"[3.4] Лучший ε={best_attack.epsilon:.3f}, "
            f"успешность атаки: {best_attack.success_rate:.2%} "
            f"({best_attack.successful}/{best_attack.total})"
        )
        print(f"[3.4] Среднее L2-возмущение: {best_attack.mean_l2:.4f}")
        y_pred_after = mlp.predict(X_test_adv)
        attacked_metrics = compute_binary_metrics(y_test, y_pred_after)
        print(
            f"[3.4] После атаки -> Accuracy: {attacked_metrics['accuracy']:.4f}, "
            f"Precision(class=1): {attacked_metrics['precision_class_1']:.4f}, "
            f"Recall(class=1): {attacked_metrics['recall_class_1']:.4f}, "
            f"F1(class=1): {attacked_metrics['f1_class_1']:.4f}"
        )
        after_cm_path = plot_confusion_matrix_png(
            y_test,
            y_pred_after,
            "MLP после FGSM: confusion matrix",
            "confusion_matrix_after_fgsm.png",
        )
        print(f"[3.5] Матрица ошибок после атаки сохранена: {after_cm_path}")

        f1_drop = mlp_metrics["f1_class_1"] - attacked_metrics["f1_class_1"]
        f1_drop_pct = f1_drop / max(mlp_metrics["f1_class_1"], 1e-12)
        print("[3.5] Сравнение до и после атаки:")
        print(
            f"  До атаки:    Acc={mlp_metrics['accuracy']:.4f}, "
            f"Prec={mlp_metrics['precision_class_1']:.4f}, "
            f"Recall={mlp_metrics['recall_class_1']:.4f}, "
            f"F1={mlp_metrics['f1_class_1']:.4f}"
        )
        print(
            f"  После атаки: Acc={attacked_metrics['accuracy']:.4f}, "
            f"Prec={attacked_metrics['precision_class_1']:.4f}, "
            f"Recall={attacked_metrics['recall_class_1']:.4f}, "
            f"F1={attacked_metrics['f1_class_1']:.4f}"
        )
        print(
            f"[3.5] Вывод: FGSM снизил F1 MLP на {f1_drop_pct:.2%}; "
            "это показывает чувствительность градиентной модели к малым "
            "направленным возмущениям."
        )

    print("\n" + "=" * 40)
    print("Сводная сравнительная таблица")
    print("=" * 40)
    save_summary_table(
        mlp_metrics=mlp_metrics,
        rf_metrics=rf_metrics,
        attacked_metrics=attacked_metrics,
        best_attack=best_attack,
        mlp_importance=mlp_importance_test,
        rf_importance=rf_importance_test,
        best_model_name=best_model_name,
        best_model_f1=best_model_f1,
    )
    print("[Done] Анализ завершен.")
    _ = mlp_importance_train


if __name__ == "__main__":
    main()
