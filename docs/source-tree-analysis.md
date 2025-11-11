# OmniDocBench 原始碼樹狀結構分析

## 概覽

本文檔提供 OmniDocBench 項目的完整目錄結構分析，包括各目錄的用途說明和關鍵文件標識。

## 完整目錄樹

```
OmniDocBench/
├── configs/                    # 評估配置文件
│   ├── end2end.yaml           # 端到端評估配置 🔑
│   ├── formula_detection.yaml # 公式檢測配置
│   ├── formula_recognition.yaml # 公式識別配置
│   ├── layout_detection.yaml  # 版面檢測配置
│   ├── md2md.yaml             # Markdown 評估配置
│   ├── ocr.yaml               # OCR 評估配置
│   └── table_recognition.yaml # 表格識別配置
│
├── dataset/                    # 數據集加載模塊 🔑
│   ├── __init__.py            # 模塊初始化和註冊
│   ├── detection_dataset.py   # 檢測任務數據集
│   ├── end2end_dataset.py     # 端到端數據集加載器
│   ├── md2md_dataset.py       # Markdown 評估數據集
│   └── recog_dataset.py       # 識別任務數據集
│
├── demo_data/                  # 演示數據目錄
│   ├── detection/             # 檢測任務演示數據
│   ├── end2end/               # 端到端演示數據
│   ├── omnidocbench_demo/     # 完整演示數據集
│   │   ├── images/            # 示例圖像
│   │   ├── mds/               # Markdown 標註
│   │   └── OmniDocBench_demo.json # 演示標註文件 🔑
│   └── recognition/           # 識別任務演示數據
│
├── docs/                       # 項目文檔 📚
│   ├── technical/             # 技術文檔
│   ├── project-overview.md    # 項目概覽
│   ├── source-tree-analysis.md # 本文件
│   └── project-scan-report.json # 掃描狀態文件
│
├── metrics/                    # 評估指標實現 🔑
│   ├── __init__.py            # 指標註冊
│   ├── cal_metric.py          # 核心指標計算邏輯
│   ├── cdm_metric.py          # CDM 指標包裝器
│   ├── show_result.py         # 結果顯示工具
│   ├── table_metric.py        # TEDS 表格指標
│   └── cdm/                   # CDM 評估模塊
│       ├── app.py             # CDM 應用入口
│       ├── convert2cdm_format.py # 格式轉換
│       ├── evaluation.py      # CDM 評估邏輯
│       ├── README.md          # CDM 文檔（英文）
│       ├── README-CN.md       # CDM 文檔（中文）
│       ├── assets/            # CDM 資源文件
│       └── modules/           # CDM 子模塊
│           ├── latex2bbox_color.py  # LaTeX 邊界框渲染
│           ├── latex_processor.py   # LaTeX 處理器
│           ├── latex_render_percentage.py # LaTeX 渲染百分比
│           ├── visual_matcher.py    # 視覺匹配器
│           └── tokenize_latex/      # LaTeX 分詞
│
├── registry/                   # 註冊表系統 🔑
│   ├── __init__.py
│   └── registry.py            # 核心註冊表實現
│
├── result/                     # 評估結果輸出目錄
│   ├── end2end_quick_match_*.json  # 端到端評估結果
│   └── *_metric_result.json   # 各類評估指標結果
│
├── signatures/                 # 數字簽名/版本管理
│   └── version1/              # 版本 1 簽名
│
├── task/                       # 評估任務定義 🔑
│   ├── __init__.py            # 任務註冊
│   ├── detection_eval.py      # 檢測評估任務
│   ├── end2end_run_eval.py    # 端到端評估任務
│   └── recognition_eval.py    # 識別評估任務
│
├── tools/                      # 推理和輔助工具
│   └── model_infer/           # 模型推理腳本
│
├── utils/                      # 工具函數庫 🔑
│   ├── data_preprocess.py     # 數據預處理
│   ├── extract.py             # 內容提取工具
│   ├── match.py               # 匹配算法基礎
│   ├── match_full.py          # 完整匹配算法
│   ├── match_quick.py         # 快速匹配算法
│   ├── ocr_utils.py           # OCR 工具函數
│   ├── read_files.py          # 文件讀取工具
│   └── table_utils.py         # 表格處理工具
│
├── .github/                    # GitHub 配置
│   └── workflows/             # CI/CD 工作流
│       └── cla.yml            # CLA 簽署自動化
│
├── .gitignore                  # Git 忽略配置
├── LICENSE                     # MIT 許可證
├── OmniDocBench_CLA.md        # 貢獻者許可協議
├── README.md                   # 主要 README（英文）🔑
├── README_zh-CN.md            # 主要 README（簡體中文）🔑
├── pdf_validation.py          # PDF 驗證腳本 🔧
└── requirements.txt            # Python 依賴清單 🔑
```

## 關鍵目錄詳解

### 1. `configs/` - 配置中心

存放所有評估任務的 YAML 配置文件。每個配置文件定義：
- 使用的評估指標
- 數據路徑（ground truth 和 prediction）
- 匹配方法
- 過濾條件

**入口點**：通過配置文件啟動對應的評估任務

### 2. `dataset/` - 數據加載層

負責加載和預處理評估數據：
- **end2end_dataset.py**：處理端到端評估數據，包含複雜的 GT-Pred 匹配邏輯
- **detection_dataset.py**：處理檢測任務的 COCO 格式數據
- **recog_dataset.py**：處理識別任務數據（公式、表格、OCR）
- **md2md_dataset.py**：處理 Markdown 格式的評估數據

所有數據集類通過 `@DATASET_REGISTRY.register()` 註冊。

### 3. `metrics/` - 評估指標庫

實現各種評估指標：
- **Edit Distance**：標準化編輯距離
- **TEDS**：樹編輯距離相似度（表格評估）
- **CDM**：字符檢測指標（公式評估）
- **BLEU/METEOR**：NLP 評估指標
- **COCODet**：目標檢測指標（mAP, mAR）

#### CDM 子模塊

CDM（Character Detection Metric）是專門用於公式識別評估的複雜指標：
- 將 LaTeX 渲染為圖像
- 執行視覺匹配
- 計算字符級別的檢測精度

### 4. `task/` - 評估任務層

定義三種主要評估任務類型：
- **DetectionEval**：版面/公式檢測評估
- **End2EndEval**：端到端文檔解析評估
- **RecognitionBaseEval**：識別任務評估（公式、表格、OCR）

所有任務類通過 `@EVAL_TASK_REGISTRY.register()` 註冊。

### 5. `utils/` - 工具函數集

提供核心功能支持：
- **match.py/match_quick.py/match_full.py**：不同策略的 GT-Pred 匹配算法
- **data_preprocess.py**：數據標準化和清理
- **extract.py**：從複雜格式中提取內容
- **read_files.py**：多格式文件讀取
- **table_utils.py/ocr_utils.py**：領域特定工具

### 6. `registry/` - 註冊表系統

實現動態組件註冊機制：
```python
EVAL_TASK_REGISTRY  # 評估任務註冊表
METRIC_REGISTRY     # 評估指標註冊表
DATASET_REGISTRY    # 數據集註冊表
```

這種設計模式允許：
- 解耦組件定義和使用
- 通過配置文件動態選擇組件
- 輕鬆添加新的任務、指標或數據集

### 7. `demo_data/` - 演示數據

包含小規模演示數據用於：
- 快速測試評估流程
- 驗證新模型輸出格式
- 文檔和教程示例

### 8. `result/` - 結果輸出

存放評估結果：
- JSON 格式的詳細結果
- 每頁/每元素的評分
- 分組統計結果

### 9. `tools/model_infer/` - 推理工具

包含各種模型的推理腳本，用於生成評估所需的預測結果。

## 入口點

### 主要入口點

1. **Python 模塊執行**：
   ```bash
   python -m task.end2end_run_eval
   ```

2. **配置驅動執行**：
   ```bash
   # 通過配置文件啟動評估
   # 配置文件路徑：configs/*.yaml
   ```

### 模塊初始化順序

1. `registry/__init__.py` - 註冊表系統初始化
2. `dataset/__init__.py` - 數據集類註冊
3. `metrics/__init__.py` - 指標類註冊
4. `task/__init__.py` - 任務類註冊

## 代碼組織模式

### 註冊表模式（Registry Pattern）

```python
# 定義
from registry.registry import EVAL_TASK_REGISTRY

@EVAL_TASK_REGISTRY.register("task_name")
class MyTask:
    pass

# 使用
task_class = EVAL_TASK_REGISTRY.get("task_name")
```

### 配置驅動設計

所有評估流程通過 YAML 配置文件驅動，實現：
- 靈活的任務定義
- 可複現的評估流程
- 易於擴展和維護

## 數據流

```
配置文件 (configs/*.yaml)
    ↓
數據集加載 (dataset/)
    ↓
GT-Pred 匹配 (utils/match*.py)
    ↓
評估任務執行 (task/)
    ↓
指標計算 (metrics/)
    ↓
結果輸出 (result/)
```

## 關鍵文件清單

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `requirements.txt` | Python 依賴清單 | ⭐⭐⭐ |
| `README.md` | 項目主文檔 | ⭐⭐⭐ |
| `configs/end2end.yaml` | 端到端評估配置 | ⭐⭐⭐ |
| `task/end2end_run_eval.py` | 端到端評估入口 | ⭐⭐⭐ |
| `dataset/end2end_dataset.py` | 數據加載核心邏輯 | ⭐⭐⭐ |
| `metrics/cal_metric.py` | 指標計算核心 | ⭐⭐⭐ |
| `registry/registry.py` | 註冊表實現 | ⭐⭐⭐ |
| `utils/match_quick.py` | 快速匹配算法 | ⭐⭐ |
| `metrics/cdm/evaluation.py` | CDM 評估邏輯 | ⭐⭐ |

## 開發時的導航建議

### 添加新的評估任務

1. 在 `task/` 中創建新的任務類
2. 使用 `@EVAL_TASK_REGISTRY.register()` 註冊
3. 在 `configs/` 中添加對應配置文件

### 添加新的評估指標

1. 在 `metrics/` 中實現指標類
2. 使用 `@METRIC_REGISTRY.register()` 註冊
3. 在配置文件中引用新指標

### 添加新的數據集加載器

1. 在 `dataset/` 中創建數據集類
2. 使用 `@DATASET_REGISTRY.register()` 註冊
3. 在配置文件中指定數據集名稱

## 測試文件位置

由於項目專注於評估而非開發，測試主要通過：
- `demo_data/` 中的演示數據
- 實際評估流程的端到端測試

## 依賴關係

```
task (評估任務)
  ↓ 依賴
metrics (評估指標) + dataset (數據加載)
  ↓ 依賴
utils (工具函數) + registry (註冊表)
```

---

*本文檔由 BMM document-project workflow 自動生成*
*生成日期：2025-11-11*
