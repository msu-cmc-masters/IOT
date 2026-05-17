#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTU-IoT 网络恶意软件流量检测完整实验程序。

流程：
1. MLP 与 Random Forest 二分类对比
2. Permutation Importance 模型解释
3. FGSM 白盒对抗攻击

图表统一输出到脚本同级 plots/ 目录。
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# 避免某些沙箱/服务器环境中用户级 Matplotlib 缓存目录不可写。
_MPLCONFIGDIR = os.path.join(tempfile.gettempdir(), "iot_prompt_v1_mplconfig")
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
_XDG_CACHE_HOME = os.path.join(_MPLCONFIGDIR, "xdg-cache")
os.makedirs(_XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_HOME)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
DATASET_SLUG = "agungpambudi/network-malware-detection-connection-analysis"

STRING_NUMERIC_COLUMNS = ["duration", "orig_bytes", "resp_bytes"]
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
IDENTIFIER_COLUMNS = [
    "uid",
    "id.orig_h",
    "id.resp_h",
    "tunnel_parents",
    "detailed-label",
]


@dataclass
class ExplorationStats:
    total_rows: int
    column_names: List[str]
    dtypes: pd.Series
    missing_counts: pd.Series
    label_counts: pd.Series
    binary_counts: pd.Series
    source_file_count: int


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    feature_names: List[str]
    label_encoders: Dict[str, LabelEncoder]
    outlier_report: pd.DataFrame


@dataclass
class EvaluationResult:
    name: str
    y_pred: np.ndarray
    y_score: np.ndarray
    metrics: Dict[str, float]


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def plots_dir() -> str:
    path = os.path.join(script_dir(), "plots")
    os.makedirs(path, exist_ok=True)
    return path


def resolve_data_path(data_path: str) -> str:
    if os.path.isabs(data_path):
        return data_path
    return os.path.abspath(os.path.join(os.getcwd(), data_path))


def discover_csv_files(data_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_path, "*.csv")))


def download_dataset_if_needed(data_path: str, no_download: bool = False) -> List[str]:
    os.makedirs(data_path, exist_ok=True)
    csv_files = discover_csv_files(data_path)
    if csv_files:
        return csv_files

    if no_download:
        raise FileNotFoundError(
            f"在 {data_path} 下没有发现 CSV 文件，并且启用了 --no_download。"
        )

    print(f"[1.1] data/ 下未发现 CSV，开始用 kagglehub 下载数据集: {DATASET_SLUG}")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("未安装 kagglehub，请先执行: uv pip install -r requirements.txt") from exc

    downloaded_path = kagglehub.dataset_download(DATASET_SLUG)
    copied = 0
    for file_name in os.listdir(downloaded_path):
        if file_name.endswith(".csv"):
            src = os.path.join(downloaded_path, file_name)
            dst = os.path.join(data_path, file_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
                print(f"  已拷贝 {file_name} -> {data_path}/")

    csv_files = discover_csv_files(data_path)
    if not csv_files:
        raise FileNotFoundError(f"下载完成但没有在 {data_path} 找到 CSV 文件。")

    print(f"[1.1] 下载/拷贝完成，共发现 {len(csv_files)} 个 CSV 文件，新增 {copied} 个。")
    return csv_files


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def compact_series_for_print(series: pd.Series, top_n: int = 12) -> str:
    series = series.sort_values(ascending=False)
    items = []
    for idx, value in series.head(top_n).items():
        items.append(f"{idx}={int(value):,}")
    remaining = len(series) - len(items)
    if remaining > 0:
        items.append(f"... 其余 {remaining} 项")
    return ", ".join(items)


def build_display_label(chunk: pd.DataFrame) -> pd.Series:
    label = chunk["label"].astype(str).str.strip()
    if "detailed-label" not in chunk.columns:
        return label

    detailed = chunk["detailed-label"].astype(str).str.strip()
    has_detail = detailed.notna() & (detailed != "") & (detailed != "-") & (detailed != "nan")
    display = label.copy()
    display.loc[has_detail] = label.loc[has_detail] + "   " + detailed.loc[has_detail]
    return display


def update_reservoir(
    reservoir: Optional[pd.DataFrame],
    incoming: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if incoming.empty:
        return reservoir if reservoir is not None else incoming

    incoming = incoming.copy()
    incoming["_sample_key"] = rng.random(len(incoming))
    combined = incoming if reservoir is None else pd.concat([reservoir, incoming], ignore_index=True)
    if len(combined) > sample_size:
        combined = combined.nsmallest(sample_size, "_sample_key")
    return combined


def read_explore_and_sample(
    csv_files: Sequence[str],
    sample_per_class: int,
    chunksize: int,
) -> Tuple[pd.DataFrame, ExplorationStats]:
    print(f"[1.1] 开始读取 {len(csv_files)} 个 CSV 文件（管道分隔符 |）。")
    rng = np.random.default_rng(RANDOM_STATE)
    reservoirs: Dict[int, Optional[pd.DataFrame]] = {0: None, 1: None}

    total_rows = 0
    column_names: List[str] = []
    dtypes: Optional[pd.Series] = None
    missing_counts: Optional[pd.Series] = None
    label_counts: Optional[pd.Series] = None
    binary_counts = pd.Series(dtype="int64")

    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        print(f"  读取: {file_name}")
        reader = pd.read_csv(csv_path, sep="|", chunksize=chunksize, low_memory=False)
        for chunk_idx, chunk in enumerate(reader, start=1):
            if not column_names:
                column_names = list(chunk.columns)
                dtypes = chunk.dtypes
                missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")

            total_rows += len(chunk)

            na_counts = chunk.isna().sum()
            dash_counts = chunk.astype("object").eq("-").sum()
            missing_counts = missing_counts.add(na_counts.add(dash_counts, fill_value=0), fill_value=0)

            display_label = build_display_label(chunk)
            label_counts = display_label.value_counts().add(
                label_counts if label_counts is not None else pd.Series(dtype="int64"),
                fill_value=0,
            )

            binary_label = (~chunk["label"].astype(str).str.strip().eq("Benign")).astype(int)
            binary_counts = binary_label.value_counts().add(binary_counts, fill_value=0)

            chunk = chunk.copy()
            chunk["label_binary"] = binary_label.to_numpy()
            for class_value in (0, 1):
                class_chunk = chunk[chunk["label_binary"] == class_value]
                reservoirs[class_value] = update_reservoir(
                    reservoirs[class_value], class_chunk, sample_per_class, rng
                )

            if chunk_idx % 10 == 0:
                kept_0 = 0 if reservoirs[0] is None else len(reservoirs[0])
                kept_1 = 0 if reservoirs[1] is None else len(reservoirs[1])
                print(
                    f"    已处理 {total_rows:,} 行；采样池 Benign={kept_0:,}, "
                    f"Malicious={kept_1:,}"
                )

    if label_counts is None or dtypes is None or missing_counts is None:
        raise RuntimeError("没有读到任何数据。")

    sample_parts = []
    for class_value in (0, 1):
        reservoir = reservoirs[class_value]
        if reservoir is None or reservoir.empty:
            raise RuntimeError(f"类别 {class_value} 没有样本，无法继续训练。")
        reservoir = reservoir.drop(columns=["_sample_key"], errors="ignore")
        sample_parts.append(reservoir)

    sample_df = pd.concat(sample_parts, ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    stats = ExplorationStats(
        total_rows=total_rows,
        column_names=column_names,
        dtypes=dtypes,
        missing_counts=missing_counts.astype("int64").sort_values(ascending=False),
        label_counts=label_counts.astype("int64").sort_values(ascending=False),
        binary_counts=binary_counts.astype("int64").sort_index(),
        source_file_count=len(csv_files),
    )
    return sample_df, stats


def print_exploration(stats: ExplorationStats, sample_df: pd.DataFrame) -> None:
    print(f"[1.1] 合并统计完成，数据形状: ({stats.total_rows:,}, {len(stats.column_names)})")
    print("[1.1] 各列数据类型:")
    print(stats.dtypes.to_string())

    print("[1.1] 缺失值/占位符 '-' 统计（Top-12）:")
    missing_ratio = stats.missing_counts / max(stats.total_rows, 1)
    missing_table = pd.DataFrame(
        {
            "missing_count": stats.missing_counts,
            "missing_ratio": missing_ratio,
        }
    ).head(12)
    print(missing_table.to_string(formatters={"missing_ratio": "{:.2%}".format}))

    label_ratio = stats.label_counts / max(stats.total_rows, 1)
    print("[1.1] 原始类别分布:")
    for label_name, count in stats.label_counts.items():
        print(f"  {label_name}: {count:,} ({label_ratio[label_name]:.2%})")

    total = stats.binary_counts.sum()
    benign = int(stats.binary_counts.get(0, 0))
    malicious = int(stats.binary_counts.get(1, 0))
    print(
        "[1.1] 二值化后类别分布: "
        f"0(Benign)={benign:,} ({benign / total:.2%}), "
        f"1(Malicious)={malicious:,} ({malicious / total:.2%})"
    )
    print(f"[1.2] 采样后数据形状: {sample_df.shape}")
    print(
        "[1.2] 采样后类别分布: "
        + compact_series_for_print(sample_df["label_binary"].value_counts().sort_index())
    )


def missing_ratio_from_stats(stats: ExplorationStats) -> pd.Series:
    return stats.missing_counts / max(stats.total_rows, 1)


def preprocess_sample(sample_df: pd.DataFrame, stats: ExplorationStats) -> PreparedData:
    print("\n[1.2] 开始数据预处理。")
    df = sample_df.copy()

    high_missing_cols = [
        col
        for col, ratio in missing_ratio_from_stats(stats).items()
        if ratio > 0.80 and col not in {"label", "label_binary"}
    ]
    if high_missing_cols:
        print(f"[1.2] 删除高缺失率列 (>80%): {high_missing_cols}")
        df = df.drop(columns=high_missing_cols, errors="ignore")
    else:
        print("[1.2] 未发现缺失率 >80% 的列。")

    existing_id_cols = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    print(f"[1.2] 删除标识符/无关列: {existing_id_cols}")
    df = df.drop(columns=existing_id_cols, errors="ignore")

    if "label_binary" not in df.columns:
        df["label_binary"] = (~df["label"].astype(str).str.strip().eq("Benign")).astype(int)

    for col in STRING_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace("-", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            df[col] = df[col].fillna(median_value)

    for col in KNOWN_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [col for col in df.columns if col not in {"label", "label_binary"}]
    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_cols = [col for col in KNOWN_CATEGORICAL_COLUMNS if col in feature_cols]
    additional_categorical_cols = [
        col
        for col in feature_cols
        if col not in numeric_cols and col not in categorical_cols
    ]
    categorical_cols.extend(additional_categorical_cols)

    for col in numeric_cols:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = df[col].fillna(median_value)

    label_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        mode = df[col].mode(dropna=True)
        fill_value = "unknown" if mode.empty else mode.iloc[0]
        df[col] = df[col].replace("-", np.nan).fillna(fill_value).astype(str)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoders[col] = encoder

    feature_cols = [col for col in df.columns if col not in {"label", "label_binary"}]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    outlier_rows = []
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_ratio = 0.0
            lower = q1
            upper = q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (X[col] < lower) | (X[col] > upper)
            outlier_ratio = float(mask.mean())
            X[col] = X[col].clip(lower=lower, upper=upper)
        outlier_rows.append(
            {
                "feature": col,
                "outlier_ratio": outlier_ratio,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )

    outlier_report = pd.DataFrame(outlier_rows).sort_values("outlier_ratio", ascending=False)
    print("[1.2] IQR 异常值比例（Top-10，已截断至上下界）:")
    print(
        outlier_report.head(10).to_string(
            index=False,
            formatters={"outlier_ratio": "{:.2%}".format},
        )
    )

    y = df["label_binary"].astype(int).to_numpy()
    print(f"[1.2] 预处理完成，特征数: {X.shape[1]}")
    return PreparedData(
        X=X,
        y=y,
        feature_names=list(X.columns),
        label_encoders=label_encoders,
        outlier_report=outlier_report,
    )


def remove_highly_correlated_features(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.95,
) -> pd.DataFrame:
    print("\n[1.3] 计算特征相关矩阵并删除高度相关特征。")
    if X.shape[1] <= 1:
        return X

    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    target_corr = {}
    for col in X.columns:
        if X[col].nunique(dropna=False) <= 1:
            target_corr[col] = 0.0
        else:
            value = np.corrcoef(X[col].to_numpy(dtype=float), y)[0, 1]
            target_corr[col] = 0.0 if np.isnan(value) else abs(float(value))

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    correlated_pairs = []
    for col in upper.columns:
        for row in upper.index[upper[col] > threshold].tolist():
            if row in to_drop or col in to_drop:
                continue
            drop_col = row if target_corr[row] < target_corr[col] else col
            keep_col = col if drop_col == row else row
            to_drop.add(drop_col)
            correlated_pairs.append((keep_col, drop_col, float(upper.loc[row, col])))

    if correlated_pairs:
        print(f"[1.3] 发现 {len(correlated_pairs)} 对相关系数 > {threshold} 的特征。")
        for keep_col, drop_col, corr_value in correlated_pairs[:10]:
            print(f"  保留 {keep_col}，删除 {drop_col}，corr={corr_value:.4f}")
        if len(correlated_pairs) > 10:
            print(f"  ... 其余 {len(correlated_pairs) - 10} 对已省略")
    else:
        print(f"[1.3] 未发现相关系数 > {threshold} 的特征对。")

    X_reduced = X.drop(columns=sorted(to_drop), errors="ignore")
    print(f"[1.3] 相关性筛选后特征数: {X_reduced.shape[1]}")
    return X_reduced


def select_features_with_random_forest(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    print("[1.3] 使用 Random Forest + SelectFromModel 思路进一步筛选特征。")
    if X.shape[1] <= 1:
        return X

    selector_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    selector_rf.fit(X, y)
    importances = selector_rf.feature_importances_
    threshold = 0.5 * float(np.mean(importances))
    keep_mask = importances > threshold
    if keep_mask.sum() == 0:
        keep_mask[np.argmax(importances)] = True

    selected_columns = X.columns[keep_mask].tolist()
    importance_table = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[1.3] 重要性阈值: {threshold:.6f}，保留 {len(selected_columns)} 个特征。")
    print("[1.3] RF 内置重要性 Top-10:")
    print(importance_table.head(10).to_string(index=False))
    return X[selected_columns].copy()


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_layers: Sequence[int] = (100,),
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        batch_size: int = 512,
        patience: int = 5,
        validation_size: float = 0.2,
        device: Optional[str] = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_size = validation_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        input_dim = self.input_dim or X.shape[1]
        self.device_ = self._resolve_device()
        self.model_ = TorchMLP(input_dim, self.hidden_layers).to(self.device_)

        if len(np.unique(y)) == 2 and len(y) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=y,
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        X_val_tensor = torch.from_numpy(X_val).to(self.device_)
        y_val_tensor = torch.from_numpy(y_val).to(self.device_)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        best_state = copy.deepcopy(self.model_.state_dict())
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_y)

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_tensor)
                val_loss = float(criterion(val_logits, y_val_tensor).item())
                val_pred = val_logits.argmax(dim=1)
                val_accuracy = float((val_pred == y_val_tensor).float().mean().item())

            improved = val_loss < best_val_loss - 1e-5
            if improved:
                best_state = copy.deepcopy(self.model_.state_dict())
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                train_loss = total_loss / max(len(train_dataset), 1)
                print(
                    f"    epoch={epoch:02d}, train_loss={train_loss:.5f}, "
                    f"val_loss={val_loss:.5f}, val_acc={val_accuracy:.4f}"
                )

            if epochs_without_improvement >= self.patience:
                break

        self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_val_loss
        self.best_val_accuracy_ = best_val_accuracy
        self.best_epoch_ = best_epoch
        return self

    def _predict_batches(self, X: np.ndarray, return_proba: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(X[start : start + self.batch_size]).to(self.device_)
                logits = self.model_(batch)
                if return_proba:
                    value = torch.softmax(logits, dim=1).cpu().numpy()
                else:
                    value = logits.argmax(dim=1).cpu().numpy()
                outputs.append(value)
        if not outputs:
            return np.empty((0, 2), dtype=float) if return_proba else np.empty((0,), dtype=int)
        return np.concatenate(outputs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batches(X, return_proba=False).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batches(X, return_proba=True)


def prepare_train_test(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    skip_smote: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    print("\n[1.4] 划分训练集/测试集，并进行标准化。")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    print(f"[1.4] 训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    print(
        "[1.4] 训练集原始类别分布: "
        + compact_series_for_print(pd.Series(y_train).value_counts().sort_index())
    )

    if skip_smote:
        print("[1.4] 已跳过 SMOTE。")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    class_counts = pd.Series(y_train).value_counts()
    min_count = int(class_counts.min())
    if len(class_counts) < 2 or min_count < 2:
        print("[1.4] 类别数量不足，跳过 SMOTE。")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(
        "[1.4] SMOTE 后训练集类别分布: "
        + compact_series_for_print(pd.Series(y_train_resampled).value_counts().sort_index())
    )
    return (
        X_train_resampled.astype(np.float32),
        X_test_scaled,
        y_train_resampled.astype(int),
        y_test.astype(int),
        scaler,
    )


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
) -> TorchMLPClassifier:
    print("\n[1.4] 训练 PyTorch MLP（MPS 可用则自动使用 Apple Silicon GPU）。")
    hidden_layer_sizes = [(100,), (100, 50), (200, 100)]
    learning_rates = [0.001, 0.01]
    device_name = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1.4] MLP 使用设备: {device_name}")

    best_model: Optional[TorchMLPClassifier] = None
    best_score = -np.inf
    best_tuple = None

    for hidden_layers in hidden_layer_sizes:
        for lr in learning_rates:
            print(f"  搜索 MLP 参数: hidden_layers={hidden_layers}, lr={lr}")
            model = TorchMLPClassifier(
                input_dim=X_train.shape[1],
                hidden_layers=hidden_layers,
                learning_rate=lr,
                num_epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(X_train, y_train)
            val_acc = model.best_val_accuracy_
            val_loss = model.best_val_loss_
            print(
                f"    val_acc={val_acc:.4f}, val_loss={val_loss:.5f}, "
                f"best_epoch={model.best_epoch_}"
            )
            score = val_acc - 1e-4 * val_loss
            if score > best_score:
                best_model = model
                best_score = score
                best_tuple = (hidden_layers, lr, model.best_epoch_, val_acc, val_loss)

    assert best_model is not None and best_tuple is not None
    hidden_layers, lr, epoch, val_acc, val_loss = best_tuple
    print(
        "[1.4] MLP 最优超参数: "
        f"hidden_layers={hidden_layers}, lr={lr}, epochs={epoch}, "
        f"val_acc={val_acc:.4f}, val_loss={val_loss:.5f}"
    )
    return best_model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, cv: int) -> RandomForestClassifier:
    print("\n[1.4] 训练 Random Forest（GridSearchCV 3 折调参）。")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    }

    def build_grid(n_jobs: int) -> GridSearchCV:
        rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            class_weight="balanced",
        )
        return GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
        )

    grid = build_grid(n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
    except PermissionError as exc:
        print(
            "[1.4] 当前运行环境限制了 joblib 并行后端，"
            "Random Forest GridSearchCV 自动回退到 n_jobs=1。"
        )
        print(f"      原始错误: {exc}")
        grid = build_grid(n_jobs=1)
        grid.fit(X_train, y_train)

    print(f"[1.4] RF 最优参数: {grid.best_params_}，CV F1(weighted)={grid.best_score_:.4f}")
    return grid.best_estimator_


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    metrics = {
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
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["auc"] = float("nan")
    return metrics


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationResult:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred.astype(float)
    metrics = compute_metrics(y_test, y_pred, y_score)

    print(
        f"[1.5] {name:<14} -> Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision(weighted): {metrics['precision_weighted']:.4f}, "
        f"Recall(weighted): {metrics['recall_weighted']:.4f}, "
        f"F1(weighted): {metrics['f1_weighted']:.4f}, AUC: {metrics['auc']:.4f}"
    )
    print(f"\n[1.5] {name} 分类报告:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malware"], zero_division=0))
    return EvaluationResult(name=name, y_pred=y_pred, y_score=y_score, metrics=metrics)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    file_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks([0, 1], labels=["Benign", "Malware"])
    ax.set_yticks([0, 1], labels=["Benign", "Malware"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = os.path.join(plots_dir(), file_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[1.5] 混淆矩阵已保存: {output_path}")


def plot_roc_curves(
    y_test: np.ndarray,
    mlp_result: EvaluationResult,
    rf_result: EvaluationResult,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result, color in [(mlp_result, "#4472C4"), (rf_result, "#ED7D31")]:
        fpr, tpr, _ = roc_curve(y_test, result.y_score)
        ax.plot(
            fpr,
            tpr,
            label=f"{result.name} (AUC={result.metrics['auc']:.4f})",
            linewidth=2,
            color=color,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("MLP 与 Random Forest ROC 曲线对比")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path = os.path.join(plots_dir(), "roc_curves.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[1.5] ROC 曲线已保存: {output_path}")


def sample_for_permutation(
    X: np.ndarray,
    y: np.ndarray,
    max_size: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_size <= 0 or len(X) <= max_size:
        return X, y
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=max_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[2] {label} 用于 Permutation Importance 的样本数: {len(X_sample):,}/{len(X):,}")
    return X_sample, y_sample


def permutation_table(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int,
    n_jobs: int,
) -> pd.DataFrame:
    try:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    except PermissionError as exc:
        if n_jobs == 1:
            raise
        print(
            "[2] 当前运行环境限制了 joblib 并行后端，"
            "Permutation Importance 自动回退到 n_jobs=1。"
        )
        print(f"    原始错误: {exc}")
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table.reset_index(drop=True)


def plot_permutation_importance(
    table: pd.DataFrame,
    title: str,
    file_name: str,
    top_n: int = 10,
) -> None:
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(top))))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#4472C4",
        alpha=0.85,
    )
    ax.set_xlabel("Accuracy 下降值")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path = os.path.join(plots_dir(), file_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] 特征重要性图已保存: {output_path}")


def print_top_features(prefix: str, table: pd.DataFrame, top_n: int = 10) -> None:
    print(prefix)
    for idx, row in table.head(top_n).iterrows():
        print(
            f"  {idx + 1:>2}. {row['feature']}: "
            f"{row['importance_mean']:.5f} ± {row['importance_std']:.5f}"
        )


def compare_importance_plot(
    mlp_table: pd.DataFrame,
    rf_table: pd.DataFrame,
    file_name: str = "permutation_importance_comparison.png",
) -> None:
    top_features = list(dict.fromkeys(mlp_table.head(10)["feature"].tolist() + rf_table.head(10)["feature"].tolist()))
    mlp_values = mlp_table.set_index("feature").reindex(top_features)["importance_mean"].fillna(0.0)
    rf_values = rf_table.set_index("feature").reindex(top_features)["importance_mean"].fillna(0.0)

    x = np.arange(len(top_features))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(top_features)), 6))
    ax.bar(x - width / 2, mlp_values, width, label="MLP", color="#4472C4")
    ax.bar(x + width / 2, rf_values, width, label="Random Forest", color="#ED7D31")
    ax.set_xticks(x, labels=top_features, rotation=45, ha="right")
    ax.set_ylabel("Permutation Importance")
    ax.set_title("MLP 与 Random Forest Top-10 特征重要性对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = os.path.join(plots_dir(), file_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[2.4] 特征重要性对比图已保存: {output_path}")


def run_permutation_importance_analysis(
    mlp_model: TorchMLPClassifier,
    rf_model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    permutation_sample_size: int,
    n_repeats: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section("第二部分：Permutation Importance 解释")
    print(
        "Permutation Importance 的思想：随机打乱一个特征，观察模型准确率下降多少；"
        "下降越多，说明模型越依赖该特征。"
    )

    X_train_pi, y_train_pi = sample_for_permutation(
        X_train, y_train, permutation_sample_size, "训练集"
    )
    X_test_pi, y_test_pi = sample_for_permutation(
        X_test, y_test, permutation_sample_size, "测试集"
    )

    print("[2.2] 计算 MLP 在训练集上的 Permutation Importance。")
    mlp_train_table = permutation_table(
        mlp_model,
        X_train_pi,
        y_train_pi,
        feature_names,
        n_repeats=n_repeats,
        n_jobs=1,
    )
    plot_permutation_importance(
        mlp_train_table,
        "MLP - 训练集 Permutation Importance",
        "permutation_importance_mlp_train.png",
    )

    print("[2.2] 计算 MLP 在测试集上的 Permutation Importance。")
    mlp_test_table = permutation_table(
        mlp_model,
        X_test_pi,
        y_test_pi,
        feature_names,
        n_repeats=n_repeats,
        n_jobs=1,
    )
    print_top_features("[2.2] MLP Top-10 关键特征（测试集）:", mlp_test_table)
    plot_permutation_importance(
        mlp_test_table,
        "MLP - 测试集 Permutation Importance",
        "permutation_importance_mlp_test.png",
    )

    mlp_train_top = set(mlp_train_table.head(10)["feature"])
    mlp_test_top = set(mlp_test_table.head(10)["feature"])
    print(
        "[2.2] MLP 训练集/测试集 Top-10 交集: "
        f"{len(mlp_train_top & mlp_test_top)}/10"
    )

    print("[2.3] 计算 Random Forest 在测试集上的 Permutation Importance。")
    rf_test_table = permutation_table(
        rf_model,
        X_test_pi,
        y_test_pi,
        feature_names,
        n_repeats=n_repeats,
        n_jobs=-1,
    )
    print_top_features("[2.3] RF Top-10 关键特征:", rf_test_table)
    plot_permutation_importance(
        rf_test_table,
        "Random Forest - 测试集 Permutation Importance",
        "permutation_importance_rf_test.png",
    )

    mlp_top = set(mlp_test_table.head(10)["feature"])
    rf_top = set(rf_test_table.head(10)["feature"])
    intersection = sorted(mlp_top & rf_top)
    mlp_only = sorted(mlp_top - rf_top)
    rf_only = sorted(rf_top - mlp_top)
    print(f"[2.4] MLP 和 RF 的 Top-10 交集: {len(intersection)}/10")
    print(f"  交集: {intersection}")
    print(f"  仅 MLP: {mlp_only}")
    print(f"  仅 RF: {rf_only}")
    if len(intersection) < 10:
        print(
            "[2.4] 两个模型依赖的特征不完全相同，说明神经网络和树模型在同一任务上"
            "可能学习到不同的判别边界与特征组合。"
        )
    else:
        print("[2.4] 两个模型的 Top-10 完全一致，说明关键判别信号非常稳定。")

    compare_importance_plot(mlp_test_table, rf_test_table)
    return mlp_train_table, mlp_test_table, rf_test_table


def targeted_fgsm_attack(
    model: TorchMLPClassifier,
    X_malicious: np.ndarray,
    epsilon: float,
    target_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if len(X_malicious) == 0:
        return X_malicious.copy(), np.array([], dtype=int), 0.0

    model.model_.eval()
    x_tensor = torch.tensor(
        X_malicious,
        dtype=torch.float32,
        device=model.device_,
        requires_grad=True,
    )
    target = torch.full(
        (len(X_malicious),),
        fill_value=target_class,
        dtype=torch.long,
        device=model.device_,
    )
    criterion = nn.CrossEntropyLoss()
    logits = model.model_(x_tensor)
    loss = criterion(logits, target)
    model.model_.zero_grad()
    loss.backward()

    # 目标攻击：让恶意流量更接近 Benign，因此沿目标类损失下降方向移动。
    x_adv = x_tensor - epsilon * x_tensor.grad.sign()
    x_adv_np = x_adv.detach().cpu().numpy().astype(np.float32)
    adv_pred = model.predict(x_adv_np)
    mean_l2 = float(np.linalg.norm(x_adv_np - X_malicious, axis=1).mean())
    return x_adv_np, adv_pred, mean_l2


def plot_fgsm_success(success_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        success_table["epsilon"],
        success_table["attack_success_rate"],
        marker="o",
        linewidth=2,
        color="#C00000",
    )
    ax.set_xlabel("epsilon")
    ax.set_ylabel("攻击成功率")
    ax.set_title("FGSM 白盒攻击成功率")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    for _, row in success_table.iterrows():
        ax.text(
            row["epsilon"],
            row["attack_success_rate"] + 0.02,
            f"{row['attack_success_rate']:.1%}",
            ha="center",
        )
    fig.tight_layout()
    output_path = os.path.join(plots_dir(), "fgsm_success_rate.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[3.4] FGSM 成功率折线图已保存: {output_path}")


def run_fgsm_attack(
    mlp_model: TorchMLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    before_result: EvaluationResult,
    epsilon_arg: Optional[float],
) -> Tuple[EvaluationResult, pd.DataFrame, float, float]:
    section("第三部分：FGSM 白盒对抗攻击")
    print(
        "FGSM 利用 MLP 的输入梯度生成扰动。这里对测试集中的恶意样本做目标攻击，"
        "目标是让模型把 class=1 的 Malware 误判为 class=0 的 Benign。"
    )

    malicious_mask = y_test == 1
    X_malicious = X_test[malicious_mask]
    print(f"[3.2] 测试集中恶意样本数: {len(X_malicious):,}")
    if len(X_malicious) == 0:
        print("[3.2] 没有恶意样本，跳过 FGSM。")
        return before_result, pd.DataFrame(), 0.0, 0.0

    epsilons = [float(epsilon_arg)] if epsilon_arg is not None else [0.01, 0.05, 0.10, 0.20]
    records = []
    adv_cache: Dict[float, Tuple[np.ndarray, np.ndarray, float]] = {}
    print("[3.3] epsilon 扫描结果:")
    for epsilon in epsilons:
        X_adv, adv_pred, mean_l2 = targeted_fgsm_attack(mlp_model, X_malicious, epsilon)
        success_rate = float((adv_pred == 0).mean())
        records.append(
            {
                "epsilon": epsilon,
                "attack_success_rate": success_rate,
                "mean_l2": mean_l2,
                "success_count": int((adv_pred == 0).sum()),
                "total_malicious": int(len(X_malicious)),
            }
        )
        adv_cache[epsilon] = (X_adv, adv_pred, mean_l2)
        print(f"  epsilon={epsilon:.2f} -> Attack Success: {success_rate:.2%}")

    success_table = pd.DataFrame(records)
    plot_fgsm_success(success_table)

    above_threshold = success_table[success_table["attack_success_rate"] > 0.50]
    if not above_threshold.empty:
        best_row = above_threshold.sort_values("epsilon").iloc[0]
    else:
        best_row = success_table.sort_values(
            ["attack_success_rate", "epsilon"], ascending=[False, True]
        ).iloc[0]
    best_epsilon = float(best_row["epsilon"])
    best_success = float(best_row["attack_success_rate"])
    X_best_adv, _, mean_l2 = adv_cache[best_epsilon]

    X_test_adv = X_test.copy()
    X_test_adv[malicious_mask] = X_best_adv
    adv_pred_full = mlp_model.predict(X_test_adv)
    adv_score_full = mlp_model.predict_proba(X_test_adv)[:, 1]
    adv_metrics = compute_metrics(y_test, adv_pred_full, adv_score_full)
    attack_result = EvaluationResult(
        name="MLP (攻击后)",
        y_pred=adv_pred_full,
        y_score=adv_score_full,
        metrics=adv_metrics,
    )

    print(
        f"[3.4] 最佳 epsilon={best_epsilon:.2f}，攻击成功率: {best_success:.2%} "
        f"({int(best_row['success_count'])}/{int(best_row['total_malicious'])})"
    )
    print(f"[3.4] 平均 L2 扰动: {mean_l2:.4f}")
    print(
        "[3.4] 攻击后 MLP 指标: "
        f"Accuracy={adv_metrics['accuracy']:.4f}, "
        f"Precision(class=1)={adv_metrics['precision_class1']:.4f}, "
        f"Recall(class=1)={adv_metrics['recall_class1']:.4f}, "
        f"F1(class=1)={adv_metrics['f1_class1']:.4f}"
    )

    print("[3.5] 攻击前后对比（class=1 / Malware）:")
    compare_df = pd.DataFrame(
        {
            "Before Attack": {
                "Accuracy": before_result.metrics["accuracy"],
                "Precision": before_result.metrics["precision_class1"],
                "Recall": before_result.metrics["recall_class1"],
                "F1": before_result.metrics["f1_class1"],
            },
            "After Attack": {
                "Accuracy": adv_metrics["accuracy"],
                "Precision": adv_metrics["precision_class1"],
                "Recall": adv_metrics["recall_class1"],
                "F1": adv_metrics["f1_class1"],
            },
        }
    )
    print(compare_df.to_string(float_format=lambda x: f"{x:.4f}"))

    plot_confusion_matrix(
        y_test,
        adv_pred_full,
        "MLP 攻击后混淆矩阵",
        "confusion_matrix_after_fgsm.png",
    )

    f1_drop_pct = 0.0
    before_f1 = before_result.metrics["f1_class1"]
    if before_f1 > 0:
        f1_drop_pct = (before_f1 - adv_metrics["f1_class1"]) / before_f1
    print(
        "[3.5] 结论：FGSM 白盒攻击使 MLP 的 class=1 F1 "
        f"从 {before_f1:.4f} 降至 {adv_metrics['f1_class1']:.4f}，"
        f"下降 {f1_drop_pct:.2%}。"
    )

    return attack_result, success_table, best_epsilon, f1_drop_pct


def format_metric(value: float) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.4f}"


def create_summary_table(
    mlp_result: EvaluationResult,
    rf_result: EvaluationResult,
    attack_result: EvaluationResult,
    best_epsilon: float,
    success_table: pd.DataFrame,
    f1_drop_pct: float,
    mlp_importance: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> None:
    section("综合对比表格")
    mlp_top10 = set(mlp_importance.head(10)["feature"])
    rf_top10 = set(rf_importance.head(10)["feature"])
    intersection_count = len(mlp_top10 & rf_top10)
    mlp_top3 = ", ".join(mlp_importance.head(3)["feature"].tolist())
    rf_top3 = ", ".join(rf_importance.head(3)["feature"].tolist())
    best_model = (
        "MLP"
        if mlp_result.metrics["f1_weighted"] >= rf_result.metrics["f1_weighted"]
        else "Random Forest"
    )
    best_f1 = max(mlp_result.metrics["f1_weighted"], rf_result.metrics["f1_weighted"])
    best_success = 0.0 if success_table.empty else float(
        success_table.loc[success_table["epsilon"].eq(best_epsilon), "attack_success_rate"].iloc[0]
    )

    rows = [
        ["模型性能对比", "", "", ""],
        [
            "Accuracy",
            format_metric(mlp_result.metrics["accuracy"]),
            format_metric(rf_result.metrics["accuracy"]),
            format_metric(attack_result.metrics["accuracy"]),
        ],
        [
            "Precision(weighted)",
            format_metric(mlp_result.metrics["precision_weighted"]),
            format_metric(rf_result.metrics["precision_weighted"]),
            format_metric(attack_result.metrics["precision_weighted"]),
        ],
        [
            "Recall(weighted)",
            format_metric(mlp_result.metrics["recall_weighted"]),
            format_metric(rf_result.metrics["recall_weighted"]),
            format_metric(attack_result.metrics["recall_weighted"]),
        ],
        [
            "F1-score(weighted)",
            format_metric(mlp_result.metrics["f1_weighted"]),
            format_metric(rf_result.metrics["f1_weighted"]),
            format_metric(attack_result.metrics["f1_weighted"]),
        ],
        ["对抗攻击影响", "", "", ""],
        ["最佳 epsilon", f"{best_epsilon:.2f}", "-", f"{best_epsilon:.2f}"],
        ["攻击成功率", "-", "-", f"{best_success:.2%}"],
        ["MLP class=1 F1 下降", f"{f1_drop_pct:.2%}", "-", f"{f1_drop_pct:.2%}"],
        ["特征重要性 Top-10", "", "", ""],
        [f"Top-10 交集数量", f"{intersection_count}/10", f"{intersection_count}/10", "-"],
        ["MLP Top-3", mlp_top3, "-", "-"],
        ["RF Top-3", "-", rf_top3, "-"],
        ["结论", "", "", ""],
        [
            "最佳分类模型及 F1",
            f"{best_model}: {best_f1:.4f}",
            f"{best_model}: {best_f1:.4f}",
            "-",
        ],
        [
            "MLP 鲁棒性评估",
            f"FGSM 使 F1 下降 {f1_drop_pct:.2%}",
            "-",
            f"攻击后 F1={attack_result.metrics['f1_class1']:.4f}",
        ],
    ]

    fig_height = max(7.0, 0.48 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title("网络流量恶意软件检测 —— 综合分析总结表", fontsize=16, fontweight="bold", pad=18)

    table = ax.table(
        cellText=rows,
        colLabels=["指标", "MLP", "Random Forest", "MLP (攻击后)"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.27, 0.25, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    section_titles = {"模型性能对比", "对抗攻击影响", "特征重要性 Top-10", "结论"}
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if row_idx == 0:
            cell.set_facecolor("#4472C4")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            row_label = rows[row_idx - 1][0]
            if row_label in section_titles:
                cell.set_facecolor("#D9E2F3")
                cell.get_text().set_weight("bold")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#F7F9FC")

    fig.tight_layout()
    output_path = os.path.join(plots_dir(), "summary_table.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"综合对比表格已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTU-IoT 网络恶意软件检测：MLP/RF、Permutation Importance、FGSM。"
    )
    parser.add_argument("--data_path", default="data", help="数据集目录路径，默认 data/")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例，默认 0.2")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="指定 FGSM 扰动强度；若提供则跳过 [0.01, 0.05, 0.1, 0.2] 扫描。",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=100_000,
        help="每个二分类类别抽样数量，默认 100000。",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="流式读取 CSV 的 chunk 大小，默认 500000。",
    )
    parser.add_argument(
        "--permutation_sample_size",
        type=int,
        default=20_000,
        help="Permutation Importance 最大样本数；设为 0 表示使用完整 train/test。",
    )
    parser.add_argument("--n_repeats", type=int, default=10, help="Permutation Importance 重复次数。")
    parser.add_argument("--epochs", type=int, default=50, help="MLP 最大训练轮数，默认 50。")
    parser.add_argument("--batch_size", type=int, default=512, help="MLP batch size，默认 512。")
    parser.add_argument("--patience", type=int, default=5, help="早停 patience，默认 5。")
    parser.add_argument("--rf_cv", type=int, default=3, help="Random Forest GridSearchCV 折数，默认 3。")
    parser.add_argument("--skip_smote", action="store_true", help="跳过 SMOTE。")
    parser.add_argument("--no_download", action="store_true", help="无 CSV 时不尝试 kagglehub 下载。")
    return parser.parse_args()


def main() -> None:
    configure_chinese_font()
    args = parse_args()

    data_path = resolve_data_path(args.data_path)
    os.makedirs(plots_dir(), exist_ok=True)

    section("第一部分：分类器（MLP vs Random Forest）")
    print(
        "本部分完成数据加载、缺失值处理、类别编码、异常值截断、特征筛选，"
        "然后训练 PyTorch MLP 与 Random Forest 并进行对比。"
    )

    csv_files = download_dataset_if_needed(data_path, no_download=args.no_download)
    sample_df, stats = read_explore_and_sample(
        csv_files,
        sample_per_class=args.sample_per_class,
        chunksize=args.chunksize,
    )
    print_exploration(stats, sample_df)

    prepared = preprocess_sample(sample_df, stats)
    X_no_corr = remove_highly_correlated_features(prepared.X, prepared.y)
    X_selected = select_features_with_random_forest(X_no_corr, prepared.y)
    feature_names = X_selected.columns.tolist()

    X_train, X_test, y_train, y_test, _ = prepare_train_test(
        X_selected,
        prepared.y,
        test_size=args.test_size,
        skip_smote=args.skip_smote,
    )

    mlp_model = train_mlp(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    rf_model = train_random_forest(X_train, y_train, cv=args.rf_cv)

    mlp_result = evaluate_model("MLP", mlp_model, X_test, y_test)
    rf_result = evaluate_model("Random Forest", rf_model, X_test, y_test)

    plot_confusion_matrix(y_test, mlp_result.y_pred, "MLP 混淆矩阵", "confusion_matrix_mlp.png")
    plot_confusion_matrix(
        y_test,
        rf_result.y_pred,
        "Random Forest 混淆矩阵",
        "confusion_matrix_random_forest.png",
    )
    plot_roc_curves(y_test, mlp_result, rf_result)

    best_result = mlp_result if mlp_result.metrics["f1_weighted"] >= rf_result.metrics["f1_weighted"] else rf_result
    print(
        f"[1.5] 最佳模型: {best_result.name} "
        f"(F1-weighted={best_result.metrics['f1_weighted']:.4f})"
    )

    _, mlp_importance, rf_importance = run_permutation_importance_analysis(
        mlp_model,
        rf_model,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        permutation_sample_size=args.permutation_sample_size,
        n_repeats=args.n_repeats,
    )

    attack_result, success_table, best_epsilon, f1_drop_pct = run_fgsm_attack(
        mlp_model,
        X_test,
        y_test,
        before_result=mlp_result,
        epsilon_arg=args.epsilon,
    )

    create_summary_table(
        mlp_result,
        rf_result,
        attack_result,
        best_epsilon,
        success_table,
        f1_drop_pct,
        mlp_importance,
        rf_importance,
    )

    print("\n全部流程运行完成。PNG 图表输出目录:")
    print(f"  {plots_dir()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断运行。", file=sys.stderr)
        sys.exit(130)
