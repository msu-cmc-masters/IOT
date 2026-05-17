#!/usr/bin/env python3
"""
Обнаружение вредоносного ПО в сетевом трафике — трёхэтапный анализ
  1. Бинарная классификация (MLP + Random Forest)
  2. Permutation Importance — объяснение модели
  3. FGSM — белая состязательная атака
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Без GUI

# Настройка кириллических шрифтов
import platform
if platform.system() == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
elif platform.system() == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Times New Roman", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Глобальные настройки
# ============================================================================
RANDOM_STATE = 42
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ============================================================================
# MLP модель (PyTorch)
# ============================================================================
class MLPClassifier(nn.Module):
    """Многослойный перцептрон с настраиваемой структурой скрытых слоёв"""

    def __init__(self, input_dim, hidden_layers=(100, 50), num_classes=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-совместимая обёртка для PyTorch MLP
    Предоставляет методы fit / predict для использования с sklearn.inspection.permutation_importance
    """

    def __init__(self, input_dim, hidden_layers=(100, 50), lr=1e-3, epochs=100,
                 batch_size=512, patience=5):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, X_val=None, y_val=None):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        self.model = MLPClassifier(self.input_dim, self.hidden_layers).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_y)
            train_loss /= len(X_t)

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        return self

    def predict(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


# ============================================================================
# Вспомогательные функции
# ============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def print_sep(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ============================================================================
# Часть 1: Загрузка и предобработка данных
# ============================================================================
def load_and_merge_data(data_path):
    """Чтение всех CSV-файлов из data/ (разделитель |), объединение в один DataFrame"""
    csv_files = sorted([
        f for f in os.listdir(data_path) if f.endswith(".csv")
    ])
    if not csv_files:
        print("[Ошибка] CSV-файлы не найдены в data/, попытка загрузки через kagglehub...")
        import kagglehub
        import shutil
        path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")
        for file in os.listdir(path):
            if file.endswith(".csv"):
                src = os.path.join(path, file)
                dst = os.path.join(data_path, file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Скопирован {file} -> {data_path}/")
        csv_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])

    print(f"  Чтение {len(csv_files)} CSV-файлов...")
    dfs = []
    for f in csv_files:
        fp = os.path.join(data_path, f)
        df = pd.read_csv(fp, sep="|", low_memory=False)
        dfs.append(df)
        print(f"    {f}: {df.shape}")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Объединение завершено, размер данных: {merged.shape}")
    return merged


def explore_data(df):
    """Исследование данных: типы столбцов, пропуски, распределение меток"""
    print(f"\n  Типы столбцов:\n{df.dtypes.value_counts().to_string()}")

    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    if len(missing_nonzero) > 0:
        print(f"\n  Пропущенные значения (первые 10):\n{missing_nonzero.head(10).to_string()}")
    else:
        print("  Пропущенных значений: 0")

    label_counts = df["label"].value_counts()
    print(f"\n  Исходное распределение label (всего {len(label_counts)} типов):")
    for lbl, cnt in label_counts.items():
        print(f"    {lbl}: {cnt} ({cnt / len(df) * 100:.2f}%)")

    return df


def binarize_label(df):
    """Бинаризация меток: Benign -> 0, остальные -> 1"""
    df["label_binary"] = df["label"].apply(lambda x: 0 if str(x).strip() == "Benign" else 1)
    counts = df["label_binary"].value_counts()
    print(f"\n  Распределение после бинаризации:")
    for cls in [0, 1]:
        label_name = "Benign" if cls == 0 else "Malicious"
        print(f"    {cls}({label_name}): {counts.get(cls, 0)} ({counts.get(cls, 0) / len(df) * 100:.2f}%)")
    return df


def preprocess_data(df):
    """Полный конвейер предобработки"""

    # 1. Удаление столбцов с высокой долей пропусков (>80%)
    threshold = 0.8
    high_missing = [c for c in df.columns if df[c].isnull().mean() > threshold]
    if high_missing:
        print(f"\n  Удалены столбцы с пропусками >{threshold*100}%: {high_missing}")
        df = df.drop(columns=high_missing)

    # 2. Удаление неинформативных идентификаторов
    drop_cols = ["uid", "id.orig_h", "id.resp_h", "tunnel_parents", "detailed-label"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    print(f"  Удалены неинформативные столбцы: {existing_drop}")
    df = df.drop(columns=existing_drop)

    # 3. Преобразование строковых столбцов в числовые
    str_to_num_cols = ["duration", "orig_bytes", "resp_bytes"]
    for col in str_to_num_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")
    print(f"  Преобразованы в числовые: {[c for c in str_to_num_cols if c in df.columns]}")

    # 4. Разделение на числовые и категориальные столбцы
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif col not in ["label", "label_binary"]:
            categorical_cols.append(col)

    print(f"  Числовые столбцы ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Категориальные столбцы ({len(categorical_cols)}): {categorical_cols}")

    # 5. Заполнение пропусков: числовые — медианой, категориальные — модой
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")

    # 6. Label Encoding для категориальных столбцов
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # 7. Обрезка выбросов методом IQR (только числовые столбцы, исключая label_binary)
    outlier_cols = [c for c in numeric_cols if c != "label_binary"]
    total_outliers = 0
    total_cells = len(df) * len(outlier_cols)
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        total_outliers += outlier_count
        df[col] = df[col].clip(lower, upper)
    outlier_ratio = total_outliers / total_cells if total_cells > 0 else 0
    print(f"  Доля выбросов (IQR): {outlier_ratio:.4f} ({total_outliers}/{total_cells})")

    return df, numeric_cols, categorical_cols


def sample_data(df, n_per_class=100000):
    """Сэмплирование: по n_per_class образцов Benign и Malicious"""
    df_benign = df[df["label_binary"] == 0]
    df_malicious = df[df["label_binary"] == 1]

    n_benign = min(n_per_class, len(df_benign))
    n_malicious = min(n_per_class, len(df_malicious))

    sampled_benign = df_benign.sample(n=n_benign, random_state=RANDOM_STATE)
    sampled_malicious = df_malicious.sample(n=n_malicious, random_state=RANDOM_STATE)

    sampled = pd.concat([sampled_benign, sampled_malicious], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\n  Размер после сэмплирования: {sampled.shape}")
    print(f"  Распределение после сэмплирования: {sampled['label_binary'].value_counts().to_dict()}")
    return sampled


# ============================================================================
# Отбор признаков
# ============================================================================
def select_features(df, feature_cols):
    """Отбор признаков: удаление высококоррелирующих + SelectFromModel"""
    X = df[feature_cols]
    y = df["label_binary"]

    # Корреляционная матрица
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Поиск пар с |корреляцией| > 0.95
    high_corr = [
        (col, row, upper_tri.loc[col, row])
        for col in upper_tri.columns
        for row in upper_tri.index
        if upper_tri.loc[col, row] > 0.95
    ]
    print(f"\n  Пар с корреляцией > 0.95: {len(high_corr)}")
    for a, b, v in high_corr[:10]:
        print(f"    {a} <-> {b}: {v:.4f}")

    # Из каждой группы оставить признак с более высокой корреляцией с целью
    drop_from_corr = set()
    for col_a, col_b, _ in high_corr:
        corr_a = abs(df[col_a].corr(df["label_binary"])) if col_a in X.columns else 0
        corr_b = abs(df[col_b].corr(df["label_binary"])) if col_b in X.columns else 0
        if col_a not in drop_from_corr and col_b not in drop_from_corr:
            if corr_a >= corr_b:
                drop_from_corr.add(col_b)
            else:
                drop_from_corr.add(col_a)

    if drop_from_corr:
        print(f"  Удалены высококоррелирующие признаки: {list(drop_from_corr)}")

    X_reduced = X.drop(columns=list(drop_from_corr), errors="ignore")
    feature_cols = list(X_reduced.columns)

    # SelectFromModel (на основе Random Forest)
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rf_selector.fit(X_reduced, y)
    selector = SelectFromModel(rf_selector, threshold="0.5*mean", prefit=True)
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    print(f"  После SelectFromModel сохранено признаков: {len(selected_features)} / {len(feature_cols)}")
    print(f"  Итоговые признаки: {selected_features}")

    return selected_features


# ============================================================================
# Обучение моделей
# ============================================================================
def train_mlp(X_train, y_train, input_dim):
    """MLP: подбор гиперпараметров + обучение"""
    hidden_options = [(100,), (100, 50), (200, 100)]
    lr_options = [0.001, 0.01]

    # 20% обучающей выборки как валидационная
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    best_acc = 0
    best_params = None
    best_model = None

    for hidden_layers in hidden_options:
        for lr in lr_options:
            wrapper = TorchMLPWrapper(
                input_dim=input_dim, hidden_layers=hidden_layers,
                lr=lr, epochs=50, batch_size=512, patience=5
            )
            wrapper.fit(X_tr, y_tr, X_val, y_val)
            preds = wrapper.predict(X_val)
            acc = accuracy_score(y_val, preds)
            print(f"    hidden={hidden_layers}, lr={lr} -> Val Acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_params = (hidden_layers, lr)
                best_model = wrapper

    hidden, lr = best_params
    print(f"\n  MLP лучшие гиперпараметры: hidden_layers={hidden}, lr={lr}, val_acc={best_acc:.4f}")
    return best_model


def train_random_forest(X_train, y_train):
    """Random Forest: GridSearchCV"""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"\n  RF лучшие параметры: {grid.best_params_}, CV Acc: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ============================================================================
# Оценка моделей
# ============================================================================
def evaluate_model(model, X_test, y_test, name, plots_dir):
    """Оценка модели: метрики, отчёт, сохранение графиков"""
    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
    else:
        y_pred = model(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1(macro): {f1_macro:.4f}")
    print(f"    F1(weighted): {f1_weighted:.4f}")
    print(f"    Отчёт о классификации:")
    print(classification_report(y_test, y_pred, target_names=["Benign(0)", "Malicious(1)"]))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Матрица ошибок сохранена: {path}")

    return {
        "name": name, "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "y_pred": y_pred, "y_proba": y_proba,
    }


def plot_roc_curves(results, y_test, plots_dir):
    """ROC-кривые двух моделей на одном графике"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        if "y_proba" in res:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{res["name"]} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(plots_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC-кривые сохранены: {path}")

    # Определение лучшей модели
    best = max(results, key=lambda r: r["f1_weighted"])
    others = [r for r in results if r != best]
    print(f"\n  Лучшая модель: {best['name']} (F1 weighted: {best['f1_weighted']:.4f})")
    for o in others:
        diff = best["f1_weighted"] - o["f1_weighted"]
        print(f"    {best['name']} опережает {o['name']} на {diff * 100:.1f} п.п.")
    return best


# ============================================================================
# Часть 2: Permutation Importance
# ============================================================================
def compute_permutation_importance(model, X, y, name):
    """Вычисление Permutation Importance"""
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=RANDOM_STATE, scoring="accuracy", n_jobs=-1
    )
    importances = result.importances_mean
    stds = result.importances_std
    return importances, stds


def plot_permutation_importance(importances, stds, feature_names, title, filepath, top_n=10):
    """Столбчатая диаграмма Permutation Importance"""
    top_n = min(top_n, len(importances))
    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(top_n),
        importances[top_idx][::-1],
        xerr=stds[top_idx][::-1],
        color="steelblue", edgecolor="black", alpha=0.8
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel("Importance (accuracy decrease)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  График сохранён: {filepath}")

    # Вывод Top-N
    print(f"\n  {title}:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s} ({importances[idx]:.4f} +/- {stds[idx]:.4f})")

    return top_idx, {feature_names[i]: importances[i] for i in top_idx}


def compare_top_features(mlp_top, rf_top, plots_dir):
    """Сравнение Top-N признаков MLP и RF"""
    all_features = list(set(mlp_top.keys()) | set(rf_top.keys()))
    mlp_vals = [mlp_top.get(f, 0) for f in all_features]
    rf_vals = [rf_top.get(f, 0) for f in all_features]

    sorted_order = np.argsort([mlp_vals[i] + rf_vals[i] for i in range(len(all_features))])
    features_sorted = [all_features[i] for i in sorted_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features_sorted))
    width = 0.35
    mlp_sorted = [mlp_top.get(f, 0) for f in features_sorted]
    rf_sorted = [rf_top.get(f, 0) for f in features_sorted]

    ax.barh(y_pos + width/2, mlp_sorted, width, label="MLP", color="steelblue", edgecolor="black", alpha=0.8)
    ax.barh(y_pos - width/2, rf_sorted, width, label="RF", color="darkorange", edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel("Importance")
    ax.set_title("Permutation Importance: MLP vs Random Forest")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plots_dir, "permutation_importance_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сравнительный график сохранён: {path}")

    # Анализ пересечений
    mlp_set = set(mlp_top.keys())
    rf_set = set(rf_top.keys())
    intersection = mlp_set & rf_set
    n_top = len(mlp_top)
    print(f"\n  Пересечение Top-{n_top} MLP и RF: {len(intersection)}/{n_top}")
    if intersection:
        print(f"    Общие признаки: {intersection}")
    print(f"    Только MLP: {mlp_set - rf_set}")
    print(f"    Только RF: {rf_set - mlp_set}")
    if len(intersection) >= 8:
        print("  Вывод: обе модели опираются на высоко согласованный набор признаков.")
    else:
        print("  Вывод: ранжирование важности признаков существенно различается, модели разного типа по-разному используют признаки.")


# ============================================================================
# Часть 3: FGSM состязательная атака
# ============================================================================
def fgsm_attack(model_wrapper, X_malicious, y_malicious_true, epsilon):
    """
    FGSM белая атака на MLP модель
    x_adv = x + epsilon * sign(grad_x L(x, y_target))
    y_target = 0 (заставить модель предсказать Benign)
    """
    model = model_wrapper.model
    model.eval()

    X_t = torch.tensor(X_malicious, dtype=torch.float32).to(DEVICE)
    X_t.requires_grad = True

    y_target = torch.zeros(len(X_t), dtype=torch.long).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    logits = model(X_t)
    loss = criterion(logits, y_target)
    model.zero_grad()
    loss.backward()

    grad_sign = X_t.grad.sign()
    X_adv = X_t + epsilon * grad_sign

    with torch.no_grad():
        logits_adv = model(X_adv)
        y_pred_adv = torch.argmax(logits_adv, dim=1).cpu().numpy()

    X_adv_np = X_adv.detach().cpu().numpy()

    # Успешные атаки: было Malicious(1) -> предсказано Benign(0)
    success_mask = (y_malicious_true == 1) & (y_pred_adv == 0)
    success_rate = success_mask.sum() / len(success_mask) if len(success_mask) > 0 else 0

    # Среднее L2-возмущение
    l2_dist = np.linalg.norm(X_adv_np - X_malicious, axis=1).mean()

    return X_adv_np, y_pred_adv, success_rate, l2_dist


def evaluate_attack_impact(model_wrapper, X_adv, X_benign_test, y_benign_test,
                           y_malicious_test, y_pred_adv_on_malicious):
    """Оценка метрик модели после атаки"""
    if hasattr(model_wrapper, "predict"):
        y_pred_benign = model_wrapper.predict(X_benign_test)
    else:
        y_pred_benign = model_wrapper(X_benign_test)

    y_all_true = np.concatenate([y_benign_test, y_malicious_test])
    y_all_pred = np.concatenate([y_pred_benign, y_pred_adv_on_malicious])

    acc = accuracy_score(y_all_true, y_all_pred)
    prec = precision_score(y_all_true, y_all_pred)
    rec = recall_score(y_all_true, y_all_pred)
    f1 = f1_score(y_all_true, y_all_pred)

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "y_all_pred": y_all_pred, "y_all_true": y_all_true}


def plot_attack_success_curve(epsilons, success_rates, plots_dir):
    """График зависимости успешности атаки от epsilon"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epsilons, success_rates, "o-", color="crimson", lw=2, markersize=8)
    ax.set_xlabel("epsilon (сила возмущения)")
    ax.set_ylabel("Успешность атаки")
    ax.set_title("FGSM: успешность атаки vs epsilon")
    ax.grid(True, alpha=0.3)
    for e, s in zip(epsilons, success_rates):
        ax.annotate(f"{s*100:.1f}%", (e, s), textcoords="offset points", xytext=(0, 10), fontsize=9)
    plt.tight_layout()
    path = os.path.join(plots_dir, "fgsm_success_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  График успешности атаки сохранён: {path}")


def plot_attack_confusion_matrix(y_true, y_pred, plots_dir):
    """Матрица ошибок после атаки"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    ax.set_title("Confusion Matrix - After FGSM Attack")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrix_after_fgsm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Матрица ошибок после атаки сохранена: {path}")


# ============================================================================
# Сводная сравнительная таблица
# ============================================================================
def plot_summary_table(mlp_results, rf_results, attack_metrics_after,
                       mlp_top_dict, rf_top_dict, best_eps, success_rate, plots_dir):
    """Сводная таблица анализа, сохранение в PNG"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    col_labels = ["Метрика", "MLP", "Random Forest", "MLP (после атаки)"]
    cell_text = []

    # ---- Сравнение производительности моделей ----
    cell_text.append(["—— Сравнение моделей ——", "", "", ""])
    for label, key in [("Accuracy", "acc"), ("Precision", "prec"),
                        ("Recall", "rec"), ("F1-score", "f1")]:
        cell_text.append([
            label,
            f"{mlp_results[key]:.4f}",
            f"{rf_results[key]:.4f}",
            f"{attack_metrics_after[key]:.4f}"
        ])

    # ---- Влияние состязательной атаки ----
    cell_text.append(["—— Влияние атаки (epsilon={:.2f}) ——".format(best_eps), "", "", ""])
    cell_text.append(["Успешность атаки", f"{success_rate*100:.1f}%", "N/A (дерево)", ""])
    f1_drop = (mlp_results["f1"] - attack_metrics_after["f1"]) / mlp_results["f1"] * 100
    cell_text.append(["Падение F1", f"{f1_drop:.1f}%", "N/A", ""])

    # ---- Важность признаков Top-10 ----
    cell_text.append(["—— Важность признаков Top-10 ——", "", "", ""])
    intersection = set(mlp_top_dict.keys()) & set(rf_top_dict.keys())
    cell_text.append(["MLP + RF пересечение", f"{len(intersection)}/10", "", ""])
    mlp_top3 = list(mlp_top_dict.keys())[:3]
    rf_top3 = list(rf_top_dict.keys())[:3]
    cell_text.append(["MLP Top-3", ", ".join(mlp_top3), "", ""])
    cell_text.append(["RF Top-3", ", ".join(rf_top3), "", ""])

    # ---- Вывод ----
    best_name = "MLP" if mlp_results["f1"] >= rf_results["f1"] else "Random Forest"
    best_f1 = max(mlp_results["f1"], rf_results["f1"])
    cell_text.append(["—— Вывод ——", "", "", ""])
    cell_text.append(["Лучшая модель", f"{best_name} (F1={best_f1:.4f})", "", ""])
    cell_text.append(["Устойчивость MLP", f"FGSM(epsilon={best_eps:.2f}): падение F1 на {f1_drop:.1f}%", "", ""])

    # Отрисовка таблицы
    n_rows = len(cell_text)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # Стилизация
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        elif "Вывод" in str(cell.get_text()):
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")
        elif any(kw in str(cell.get_text()) for kw in ["Сравнение моделей", "Влияние атаки", "Важность признаков"]):
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")

    ax.set_title("Network Traffic Malware Detection — Comprehensive Analysis Summary",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(plots_dir, "summary_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\nСводная таблица сохранена: {path}")


# ============================================================================
# Главный процесс
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Обнаружение вредоносного ПО в сетевом трафике: трёхэтапный анализ")
    parser.add_argument("--data_path", type=str, default="data", help="Путь к директории с датасетом")
    parser.add_argument("--test_size", type=float, default=0.2, help="Доля тестовой выборки")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Сила возмущения FGSM, если указана — сканирование epsilon пропускается")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "plots")
    ensure_dir(plots_dir)

    print(f"Используемое устройство: {DEVICE}")
    print(f"Директория для графиков: {plots_dir}")

    # ========================================================================
    # Часть 1: Классификатор
    # ========================================================================
    print_sep("Часть 1: Классификатор (MLP vs Random Forest)")

    print("\n[1.1] Загрузка и исследование данных")
    df = load_and_merge_data(args.data_path)
    explore_data(df)
    df = binarize_label(df)

    print("\n[1.2] Предобработка данных")
    df, numeric_cols, categorical_cols = preprocess_data(df)

    # Сэмплирование
    df_sampled = sample_data(df, n_per_class=100000)

    # Определение столбцов признаков
    feature_cols = [c for c in df_sampled.columns
                    if c not in ["label", "label_binary"]]

    print("\n[1.3] Отбор признаков")
    selected_features = select_features(df_sampled, feature_cols)

    X = df_sampled[selected_features].values
    y = df_sampled["label_binary"].values

    # Стандартизация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # SMOTE
    print("  Применение SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X, y = smote.fit_resample(X, y)
    print(f"  Размер после SMOTE: {X.shape}, распределение: {Counter(y)}")

    # Стратифицированное разделение train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")

    print("\n[1.4] Обучение моделей")
    print("  Обучение MLP (PyTorch + MPS)...")
    mlp_model = train_mlp(X_train, y_train, X_train.shape[1])

    print("  Обучение Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("\n[1.5] Оценка моделей")
    mlp_results = evaluate_model(mlp_model, X_test, y_test, "MLP", plots_dir)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", plots_dir)

    plot_roc_curves([mlp_results, rf_results], y_test, plots_dir)

    # ========================================================================
    # Часть 2: Permutation Importance
    # ========================================================================
    print_sep("Часть 2: Объяснение Permutation Importance")
    print("\n  Принцип Permutation Importance: случайно перемешать значения одного признака")
    print("  и измерить падение точности модели. Чем сильнее падение, тем важнее признак.")
    print("  Метод не зависит от внутренней структуры модели.")

    print("\n[2.2] MLP Permutation Importance")
    mlp_imp, mlp_std = compute_permutation_importance(mlp_model, X_test, y_test, "MLP")
    _, mlp_top_dict = plot_permutation_importance(
        mlp_imp, mlp_std, selected_features,
        "MLP Permutation Importance (Test)",
        os.path.join(plots_dir, "permutation_importance_mlp_test.png"),
        top_n=10
    )

    print("\n  Permutation Importance на обучающей выборке:")
    mlp_imp_train, _ = compute_permutation_importance(mlp_model, X_train, y_train, "MLP")
    sorted_train = np.argsort(mlp_imp_train)[::-1][:10]
    sorted_test = np.argsort(mlp_imp)[::-1][:10]
    overlap = len(set(sorted_train[:10]) & set(sorted_test[:10]))
    print(f"  Пересечение Top-{len(sorted_train)} (обучение vs тест): {overlap}/{len(sorted_train)}")

    print("\n[2.3] RF Permutation Importance")
    rf_imp, rf_std = compute_permutation_importance(rf_model, X_test, y_test, "RF")
    _, rf_top_dict = plot_permutation_importance(
        rf_imp, rf_std, selected_features,
        "RF Permutation Importance (Test)",
        os.path.join(plots_dir, "permutation_importance_rf_test.png"),
        top_n=10
    )

    print("\n[2.4] Сравнение важности признаков MLP и RF")
    compare_top_features(mlp_top_dict, rf_top_dict, plots_dir)

    # ========================================================================
    # Часть 3: FGSM состязательная атака
    # ========================================================================
    print_sep("Часть 3: FGSM белая состязательная атака")
    print("\n  FGSM (Fast Gradient Sign Method): использует информацию о градиенте модели,")
    print("  добавляя малое возмущение вдоль направления градиента ко входным данным,")
    print("  чтобы изменить предсказание модели.")
    print("  Основная формула: x_adv = x + epsilon * sign(grad_x L(x, y_target))")
    print("  Только MLP (дифференцируемая модель) может быть атакован; RF (дерево) — нет.")

    # Выделение вредоносных образцов из тестовой выборки
    malicious_mask = y_test == 1
    benign_mask = y_test == 0
    X_malicious = X_test[malicious_mask]
    y_malicious = y_test[malicious_mask]
    X_benign = X_test[benign_mask]
    y_benign = y_test[benign_mask]
    print(f"\n  Вредоносных образцов: {len(X_malicious)}, Нормальных образцов: {len(X_benign)}")

    if args.epsilon is not None:
        # Использовать указанный epsilon напрямую
        print(f"\n[3.3] Использование заданного epsilon = {args.epsilon}")
        best_eps = args.epsilon
        X_adv, y_pred_adv, success_rate, l2_dist = fgsm_attack(
            mlp_model, X_malicious, y_malicious, args.epsilon
        )
        print(f"  Успешность атаки: {success_rate*100:.2f}%")
        print(f"  Среднее L2-возмущение: {l2_dist:.4f}")

        attack_metrics_after = evaluate_attack_impact(
            mlp_model, X_adv, X_benign, y_benign, y_malicious, y_pred_adv
        )
    else:
        # Сканирование epsilon
        print("\n[3.3] Сканирование epsilon")
        epsilons = [0.01, 0.05, 0.1, 0.2]
        scan_results = []

        for eps in epsilons:
            X_adv, y_pred_adv, success_rate, l2_dist = fgsm_attack(
                mlp_model, X_malicious, y_malicious, eps
            )
            scan_results.append({
                "eps": eps, "success_rate": success_rate,
                "l2_dist": l2_dist, "X_adv": X_adv, "y_pred_adv": y_pred_adv
            })
            marker = " <-- лучший" if success_rate > 0.5 else ""
            print(f"    epsilon={eps:.2f} -> Успешность: {success_rate*100:.2f}%{marker}")

        # Выбор лучшего epsilon (минимальный с успешностью > 50%)
        valid = [r for r in scan_results if r["success_rate"] > 0.5]
        if valid:
            best_result = min(valid, key=lambda r: r["eps"])
        else:
            best_result = max(scan_results, key=lambda r: r["success_rate"])
            print("  ! Нет epsilon с успешностью > 50%, выбран epsilon с максимальной успешностью")

        epsilons_vals = [r["eps"] for r in scan_results]
        success_rates_vals = [r["success_rate"] for r in scan_results]
        plot_attack_success_curve(epsilons_vals, success_rates_vals, plots_dir)

        best_eps = best_result["eps"]
        X_adv = best_result["X_adv"]
        y_pred_adv = best_result["y_pred_adv"]
        success_rate = best_result["success_rate"]
        l2_dist = best_result["l2_dist"]

        print(f"\n[3.4] Лучший epsilon = {best_eps:.2f}")
        print(f"  Успешность атаки: {success_rate*100:.2f}% "
              f"({int(success_rate * len(y_malicious))}/{len(y_malicious)})")
        print(f"  Среднее L2-возмущение: {l2_dist:.4f}")

        attack_metrics_after = evaluate_attack_impact(
            mlp_model, X_adv, X_benign, y_benign, y_malicious, y_pred_adv
        )

    print(f"\n  Метрики MLP после атаки:"
          f"  Acc={attack_metrics_after['acc']:.4f}, "
          f"Prec={attack_metrics_after['prec']:.4f}, "
          f"Recall={attack_metrics_after['rec']:.4f}, "
          f"F1={attack_metrics_after['f1']:.4f}")

    # Сравнение до и после атаки
    print("\n[3.5] Сравнение до и после атаки")
    print(f"\n  {'Метрика':<12} {'До атаки':<12} {'После атаки':<12} {'Изменение':<12}")
    print(f"  {'-' * 48}")
    before_metrics = mlp_results
    metrics_names = [("Acc", "acc"), ("Prec", "prec"), ("Recall", "rec"), ("F1", "f1")]
    for label, key in metrics_names:
        before_val = before_metrics[key]
        after_val = attack_metrics_after[key]
        change = after_val - before_val
        print(f"  {label:<12} {before_val:<12.4f} {after_val:<12.4f} {change:+.4f}")

    # Матрица ошибок после атаки
    plot_attack_confusion_matrix(
        attack_metrics_after["y_all_true"],
        attack_metrics_after["y_all_pred"],
        plots_dir
    )

    # Вывод
    f1_before = before_metrics["f1"]
    f1_after = attack_metrics_after["f1"]
    f1_drop_pct = (f1_before - f1_after) / f1_before * 100
    print(f"\n  Вывод: FGSM белая атака снизила F1 модели MLP с {f1_before:.4f} до {f1_after:.4f},")
    print(f"  падение на {f1_drop_pct:.1f}%. MLP, хотя и имеет градиент (позволяя использовать")
    print(f"  белую атаку), сама модель не обладает устойчивостью к малым возмущениям.")

    # Сводная таблица
    plot_summary_table(
        mlp_results, rf_results, attack_metrics_after,
        mlp_top_dict, rf_top_dict,
        best_eps, success_rate, plots_dir
    )

    print("\n" + "=" * 60)
    print("Анализ завершён!")
    print("=" * 60)


if __name__ == "__main__":
    main()
