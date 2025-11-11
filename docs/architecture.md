# OmniDocBench 架構文檔

## 執行摘要

OmniDocBench 採用基於 Python 的模塊化評估管道架構，結合註冊表模式實現靈活的組件管理。系統設計專注於可擴展性、配置驅動和模塊解耦，使其能夠輕鬆支持新的評估任務、指標和數據集類型。

### 關鍵架構決策

| 決策 | 選擇 | 理由 |
|------|------|------|
| **架構模式** | 管道架構 + 註冊表模式 | 靈活配置、動態組件加載 |
| **語言** | Python 3.x | 豐富的數據科學和 ML 生態系統 |
| **配置管理** | YAML 文件 | 人類可讀、易於維護 |
| **組件註冊** | 裝飾器模式 | 解耦定義與使用、支持動態發現 |
| **數據格式** | JSON | 標準化、易於解析和處理 |

## 技術棧

### 核心依賴

#### 數據處理層
- **pandas** (2.0.3) - 數據分析和處理
- **numpy** (1.24.4) - 數值計算基礎
- **datasets** (3.1.0) - HuggingFace 數據集支持
- **pyarrow** (17.0.0) - 高效數據序列化

#### 計算機視覺層
- **opencv-python** (4.10.0.84) - 圖像處理
- **Pillow** (10.4.0) - 圖像操作
- **matplotlib** (3.7.5) - 可視化

#### 評估指標層
- **mmeval** (0.2.1) - 多模態評估框架
- **evaluate** (0.4.3) - HuggingFace 評估工具
- **scikit-learn** (1.1.2) - 機器學習指標
- **pycocotools** (2.0.7) - COCO 格式檢測評估

#### 文本處理層
- **nltk** (3.9.1) - 自然語言處理
- **Levenshtein** (0.25.1) - 編輯距離計算
- **rapidfuzz** (3.9.7) - 快速模糊匹配
- **pylatexenc** (3.0a30) - LaTeX 編碼處理

#### 配置和工具層
- **PyYAML** (6.0.2) - YAML 解析
- **click** (8.1.7) - CLI 構建
- **loguru** (0.7.2) - 日誌記錄
- **tqdm** (4.67.1) - 進度條

## 架構模式

### 1. 註冊表模式（Registry Pattern）

核心設計模式，用於動態管理組件。

#### 實現細節

```python
# registry/registry.py
class Registry:
    def __init__(self, name):
        self._registry = {}
        self._name = name

    def register(self, name):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        return self._registry[name]

# 全局註冊表實例
EVAL_TASK_REGISTRY = Registry("EvalTask")
METRIC_REGISTRY = Registry("Metric")
DATASET_REGISTRY = Registry("Dataset")
```

#### 使用示例

```python
# 註冊組件
@EVAL_TASK_REGISTRY.register("end2end_eval")
class End2EndEval:
    pass

# 獲取組件
task_class = EVAL_TASK_REGISTRY.get("end2end_eval")
task_instance = task_class(dataset, metrics, ...)
```

#### 優勢
- **解耦**：組件定義與使用分離
- **動態性**：運行時查找和實例化
- **擴展性**：新組件無需修改核心代碼

### 2. 管道架構（Pipeline Architecture）

評估流程組織為線性管道。

```
配置加載 → 數據加載 → GT-Pred 匹配 → 評估執行 → 指標計算 → 結果輸出
```

#### 階段詳解

1. **配置加載階段**
   - 讀取 YAML 配置文件
   - 解析任務類型、數據路徑、指標選擇
   - 設置過濾條件

2. **數據加載階段**
   - 通過 `DATASET_REGISTRY` 獲取數據集類
   - 加載 Ground Truth 和 Prediction 數據
   - 執行數據預處理

3. **GT-Pred 匹配階段**
   - 選擇匹配算法（quick_match, full_match等）
   - 建立 GT 和 Pred 元素的對應關係
   - 處理特殊情況（截斷文本、跨頁元素）

4. **評估執行階段**
   - 通過 `EVAL_TASK_REGISTRY` 獲取任務類
   - 初始化任務實例
   - 協調指標計算

5. **指標計算階段**
   - 通過 `METRIC_REGISTRY` 獲取指標類
   - 對匹配的樣本對計算指標
   - 支持分組統計（按語言、文檔類型等）

6. **結果輸出階段**
   - 生成 JSON 格式結果
   - 控制台顯示彙總統計
   - 保存詳細評分到 `result/` 目錄

### 3. 模塊化設計

系統分為清晰的模塊層次。

```
┌─────────────────────────────────────────┐
│           配置層 (configs/)              │
│          YAML 配置文件                    │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│           任務層 (task/)                 │
│     End2EndEval, DetectionEval,         │
│     RecognitionEval                      │
└────────┬───────────────┬────────────────┘
         │               │
┌────────▼──────┐  ┌────▼────────────────┐
│  數據層        │  │  指標層              │
│  (dataset/)   │  │  (metrics/)         │
│               │  │                      │
│  數據加載      │  │  TEDS, Edit Dist,   │
│  格式轉換      │  │  CDM, BLEU, etc.    │
└────────┬──────┘  └────┬────────────────┘
         │               │
         └───────┬───────┘
                 │
┌────────────────▼────────────────────────┐
│        工具層 (utils/)                   │
│   match.py, data_preprocess.py,         │
│   extract.py, read_files.py             │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│        註冊表層 (registry/)              │
│      動態組件管理                         │
└─────────────────────────────────────────┘
```

## 數據架構

### 數據格式

#### Ground Truth 格式（JSON）

```json
[{
    "page_info": {
        "image_path": "path/to/image.jpg",
        "page_attribute": {
            "language": "english",
            "doc_type": "scientific_paper",
            "layout_type": "single_column"
        }
    },
    "layout_dets": [
        {
            "anno_id": 1,
            "category_type": "text_block",
            "poly": [x1, y1, x2, y2, ...],
            "text": "內容...",
            "order": 1
        },
        {
            "anno_id": 2,
            "category_type": "table",
            "poly": [...],
            "html": "<table>...</table>",
            "latex": "LaTeX 表格",
            "order": 2
        }
    ],
    "extra": {
        "relation": [
            {
                "relation_type": "truncated",
                "source_anno_id": 1,
                "target_anno_id": 3
            }
        ]
    }
}]
```

#### Prediction 格式（Markdown/JSON）

支持多種預測輸出格式：
- Markdown 文件
- JSON 結構化數據
- COCO 格式（檢測任務）

### 數據流

```
JSON Ground Truth
       │
       ├─→ dataset/end2end_dataset.py
       │       │
       │       ├─ get_page_elements()     # 提取頁面元素
       │       ├─ handle_truncated_text() # 處理截斷文本
       │       └─ get_matched_elements()  # GT-Pred 匹配
       │
       ↓
Matched Samples
       │
       ├─→ task/end2end_run_eval.py
       │       │
       │       └─ 遍歷每個元素類型
       │
       ↓
Metric Calculation
       │
       ├─→ metrics/cal_metric.py
       │       │
       │       ├─ call_Edit_dist()
       │       ├─ call_TEDS()
       │       └─ call_CDM()
       │
       ↓
結果輸出 (JSON)
```

## 核心算法

### 1. GT-Pred 匹配算法

#### Quick Match（快速匹配）

**位置**：`utils/match_quick.py`

**策略**：
1. 基於空間位置的快速匹配
2. 使用 IoU（Intersection over Union）閾值
3. 優先匹配相同類別的元素

**適用場景**：大規模評估、實時反饋

#### Full Match（完整匹配）

**位置**：`utils/match_full.py`

**策略**：
1. 使用匈牙利算法（Hungarian Algorithm）
2. 考慮多種特徵：位置、類別、內容相似度
3. 全局最優匹配

**適用場景**：精確評估、研究分析

### 2. 混合匹配（Hybrid Matching）

**版本**：v1.5 引入

**創新點**：
- 允許公式和文本相互匹配
- 解決模型將公式輸出為 Unicode 的問題
- 提高評分準確性

### 3. CDM 評估算法

**位置**：`metrics/cdm/`

**流程**：
1. **LaTeX → 圖像渲染**
   - 將 LaTeX 公式渲染為圖像
   - 標準化字體和大小

2. **視覺匹配**
   - 提取字符級別的邊界框
   - 計算視覺相似度

3. **指標計算**
   - Precision、Recall、F1
   - 字符級別的檢測準確率

### 4. TEDS（樹編輯距離）

**位置**：`metrics/table_metric.py`

**策略**：
1. 將 HTML 表格解析為樹結構
2. 計算樹編輯距離
3. 支持 structure_only 模式（僅結構）

## API 設計

### 評估任務 API

#### End2EndEval

```python
class End2EndEval:
    def __init__(
        self,
        dataset,          # 數據集實例
        metrics_list,     # 指標配置字典
        page_info_path,   # 頁面信息路徑
        save_name         # 結果保存名稱
    ):
        # 對每個元素類型執行評估
        for element in metrics_list.keys():
            samples = dataset.samples[element]
            for metric in metrics_list[element]['metric']:
                # 獲取並執行指標
                metric_val = METRIC_REGISTRY.get(metric)
                samples, result = metric_val(samples).evaluate(...)
```

### 數據集 API

#### End2EndDataset

```python
class End2EndDataset:
    def __init__(self, cfg_task):
        # 加載 GT 和 Pred 數據
        gt_path = cfg_task['dataset']['ground_truth']['data_path']
        pred_folder = cfg_task['dataset']['prediction']['data_path']

        # 應用過濾器
        filtered_types = cfg_task['dataset'].get('filter')

        # 執行匹配
        self.samples = self.get_matched_elements(...)

    def get_page_elements(self, annos):
        # 處理截斷文本
        # 按類別組織元素
        # 返回元素字典
        pass
```

### 指標 API

所有指標遵循統一接口：

```python
@METRIC_REGISTRY.register("metric_name")
class MetricClass:
    def __init__(self, samples):
        self.samples = samples

    def evaluate(self, group_info=[], save_name='default'):
        # 計算指標
        # 返回 (samples, results)
        return self.samples, results
```

## 組件交互

### 配置驅動流程

```
1. 用戶定義配置文件
   configs/end2end.yaml

2. 配置解析
   ├─ 選擇數據集類型："end2end_dataset"
   ├─ 選擇評估任務："end2end_eval"
   └─ 選擇指標：["Edit_dist", "TEDS", "CDM_plain"]

3. 動態組件實例化
   ├─ dataset = DATASET_REGISTRY.get("end2end_dataset")(cfg)
   ├─ task = EVAL_TASK_REGISTRY.get("end2end_eval")(dataset, ...)
   └─ 內部調用 METRIC_REGISTRY.get("Edit_dist") 等

4. 執行評估
   └─ task 自動協調數據和指標計算

5. 結果輸出
   └─ result/*.json
```

## 測試策略

### 驗證方法

1. **演示數據測試**
   - 使用 `demo_data/` 中的小規模數據
   - 驗證完整評估流程
   - 確保各組件正確協作

2. **端到端測試**
   - 從配置文件到結果輸出的完整流程
   - 覆蓋所有評估任務類型

3. **指標單元測試**
   - 已知輸入-輸出對的指標計算驗證
   - 邊界情況測試

### 測試數據

- **demo_data/omnidocbench_demo/**：完整演示數據集
- **demo_data/end2end/**：端到端評估樣例
- **demo_data/detection/**：檢測任務樣例
- **demo_data/recognition/**：識別任務樣例

## 部署架構

### 執行環境

- **本地執行**：直接運行 Python 腳本
- **Docker 支持**：提供 Docker 環境（v1.5+）
- **批處理**：支持批量評估多個模型

### 資源需求

- **CPU**：多核處理器（並行評估）
- **內存**：取決於數據集大小（建議 16GB+）
- **存儲**：數據集和結果存儲空間

### 擴展性

- **水平擴展**：支持並行處理多個評估任務
- **垂直擴展**：利用多核 CPU 加速計算

## 配置管理

### 配置文件結構

```yaml
end2end_eval:
  metrics:
    text_block:        # 元素類型
      metric:          # 指標列表
        - Edit_dist
    display_formula:
      metric:
        - Edit_dist
        - CDM_plain
    table:
      metric:
        - TEDS
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./path/to/gt.json
    prediction:
      data_path: ./path/to/pred/
    match_method: quick_match
    filter:              # 可選過濾器
      language: english
```

### 配置選項

- **metrics**：為每種元素類型選擇評估指標
- **dataset.match_method**：選擇匹配算法
- **dataset.filter**：按頁面屬性過濾數據集
- **dataset.group**：定義分組統計

## 錯誤處理

### 策略

1. **異常捕獲**：關鍵操作包裹 try-except
2. **降級處理**：指標計算失敗時設為 0 並記錄
3. **超時保護**：使用 `func_timeout` 防止無限執行
4. **日誌記錄**：使用 `loguru` 詳細記錄錯誤

### 示例

```python
try:
    score = teds.evaluate(pred, gt)
except:
    score = 0
    print(f'TEDS score error for table {sample["gt_idx"]}. Set to 0.')
```

## 性能考慮

### 優化策略

1. **批量處理**：使用 pandas 進行向量化操作
2. **緩存**：避免重複計算
3. **並行計算**：`ProcessPoolExecutor` 用於 CPU 密集型任務
4. **惰性加載**：按需加載數據

### 瓶頸

- **CDM 計算**：LaTeX 渲染和視覺匹配耗時
- **大規模數據集**：內存佔用和 I/O 開銷
- **匹配算法**：Full Match 的複雜度較高

## 安全考慮

- **輸入驗證**：驗證配置文件和數據格式
- **路徑安全**：防止路徑遍歷攻擊
- **依賴安全**：定期更新依賴以修復漏洞

## 未來架構改進

- **微服務化**：將評估任務拆分為獨立服務
- **API 服務**：提供 REST API 進行遠程評估
- **分佈式計算**：支持大規模並行評估
- **Web UI**：圖形化配置和結果可視化

---

*本文檔由 BMM document-project workflow 自動生成*
*生成日期：2025-11-11*
