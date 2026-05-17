<div align="center">

# Обнаружение вредоносного IoT-трафика с кодом, сгенерированным LLM

[![English](https://img.shields.io/badge/README-English-2ea44f?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh.md)
[![Русский](https://img.shields.io/badge/README-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue?style=for-the-badge)](README.ru.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)

**Многоязычный эксперимент: может ли LLM по промптам на английском, китайском и русском языках сгенерировать рабочую Python-программу для классификации IoT-трафика, объяснения моделей и проверки устойчивости к состязательным атакам.**

</div>

---

## Обзор

Этот репозиторий исследует полный ML-процесс для задач сетевой безопасности:

1. **Бинарная классификация** IoT-трафика на нормальный и вредоносный.
2. **Объяснение моделей** с помощью Permutation Importance.
3. **Проверка устойчивости** MLP-модели с помощью градиентных состязательных атак.

Проект сравнивает вручную написанный baseline с кодом, сгенерированным LLM по семантически эквивалентным промптам на **английском**, **китайском** и **русском** языках. В репозитории сохранены промпты, сгенерированный код, таблицы результатов и графики, чтобы эксперимент можно было проверить и воспроизвести.

## Дизайн эксперимента

| Этап                   | Цель                                                                                                              | Реализация                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Классификация | Отличить нормальные сетевые потоки от вредоносных                         | Random Forest и PyTorch MLP                                                  |
| Объяснение       | Найти признаки, влияющие на решение модели                                        | `sklearn.inspection.permutation_importance`                                 |
| Атака                 | Проверить устойчивость MLP-классификатора                                          | FGSM / PGD-подобные состязательные возмущения |
| Сравнение         | Оценить влияние языка промпта и качество сгенерированного кода | Baseline vs. LLM-код для EN / ZH / RU                                   |

Общая схема следует цепочке «классификация -> объяснение -> атака». При этом вместо маршрута с древесной моделью и SHAP используется связка MLP + Permutation Importance + градиентная атака.

## Датасет

В эксперименте используется Kaggle-датасет [Malware Detection in Network Traffic Data](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis), содержащий логи сетевых потоков CTU-IoT-Malware-Capture.

| Пункт                                    | Описание                                                                                                                                      |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Файлы                                    | 12 CSV-файлов с разделителем `                                                                                                     |
| Расположение                      | `data/CTU-IoT-Malware-Capture-*-1conn.log.labeled.csv`                                                                                              |
| Объем                                    | Около 25 млн записей сетевых потоков                                                                                     |
| Столбец метки                     | `label`                                                                                                                                             |
| Бинарное преобразование | `Benign` -> 0, все вредоносные метки -> 1                                                                                        |
| Примечание                          | Локальная папка `data/` игнорируется Git, потому что исходные данные слишком большие |

Загрузка данных:

```bash
python download_data.py
```

## Структура репозитория

```text
.
├── README.md
├── README.zh.md
├── README.ru.md
├── download_data.py
├── requirements.txt
├── prompts/
│   ├── en/prompt_v1.md
│   ├── zh/prompt_v1.md
│   └── ru/prompt_v1.md
├── baseline/
│   ├── baseline_simple.py
│   ├── baseline_with_mlp.py
│   ├── baseline_with_importance.py
│   ├── baseline_full.py
│   ├── baseline_final.py
│   ├── results/
│   └── plots/
├── generated_code/
│   ├── gpt-5.5/
│   └── DS_V4pro/
└── materials/
    ├── proccess.md
    └── reference PDFs
```

## Быстрый старт

Создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Загрузите датасет:

```bash
python download_data.py
```

Запустите простой baseline:

```bash
python baseline/baseline_simple.py
```

Запустите baseline с важностью признаков:

```bash
python baseline/baseline_with_importance.py
```

Запустите полный baseline / сценарий атаки:

```bash
python baseline/baseline_final.py
```

Запустите один из вариантов, сгенерированных LLM, например:

```bash
python generated_code/gpt-5.5/en_v1/prompt_v1.py
```

## Текущие результаты

Файлы baseline-результатов находятся в `baseline/results/`.

| Файл                        | Содержимое                                                            |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `baseline_metrics.csv`        | Метрики baseline для Random Forest                                    |
| `model_comparison.csv`        | Сравнение метрик Random Forest и MLP                            |
| `feature_importance.csv`      | Рейтинг признаков по Permutation Importance                   |
| `pgd_results_optimized.csv`   | Успешность атаки для разных epsilon                     |
| `model_comparison_attack.csv` | Метрики до и после атакующих экспериментов |

Зафиксированные метрики классификации baseline:

| Модель  | Accuracy |     F1 |
| ------------- | -------: | -----: |
| Random Forest |   0.9997 | 0.9997 |
| MLP           |   0.9990 | 0.9990 |

Сгенерированные графики находятся в:

- `baseline/plots/`
- `generated_code/<model>/<language_version>/plots/`

## Матрица промптов и генераций

| Язык промпта | Файл промпта     | Примеры сгенерированного кода              |
| ----------------------- | --------------------------- | --------------------------------------------------------------------- |
| Английский    | `prompts/en/prompt_v1.md` | `generated_code/gpt-5.5/en_v1/`, `generated_code/DS_V4pro/en_v1/` |
| Китайский      | `prompts/zh/prompt_v1.md` | `generated_code/gpt-5.5/zh_v1/`, `generated_code/DS_V4pro/zh_v1/` |
| Русский          | `prompts/ru/prompt_v1.md` | `generated_code/gpt-5.5/ru_v1/`, `generated_code/DS_V4pro/ru_v1/` |

## Соглашение о выходных файлах

Каждый запускаемый эксперимент обычно сохраняет артефакты рядом со скриптом, в папке `plots/`:

```text
generated_code/<model>/<language_version>/
├── prompt_v1.py
└── plots/
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_mlp.png
    ├── permutation_importance_comparison.png
    ├── roc_curves.png
    └── summary_table.png
```

## Основные зависимости

| Пакет            | Назначение                                                              |
| --------------------- | --------------------------------------------------------------------------------- |
| `pandas`, `numpy` | Загрузка и предобработка данных                       |
| `scikit-learn`      | Random Forest, метрики, предобработка, Permutation Importance |
| `imbalanced-learn`  | Работа с дисбалансом классов                             |
| `matplotlib`        | Визуализация результатов                                   |
| `kagglehub`         | Загрузка датасета                                                 |
| `torch`             | MLP-модель и градиентные атаки                             |
