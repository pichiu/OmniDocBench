# OmniDocBench 開發指南

## 快速開始

本指南提供在本地設置、運行和擴展 OmniDocBench 所需的所有信息。

## 先決條件

### 系統需求

- **操作系統**：Linux / macOS / Windows（推薦 Linux）
- **Python版本**：Python 3.7+（推薦 3.8 或更高）
- **內存**：建議 16GB+ RAM
- **存儲**：至少 10GB 可用空間（用於數據集和結果）

### 必需工具

- **Python 3.7+**
- **pip** （Python 包管理器）
- **Git** （版本控制）
- **Docker** （可選，用於容器化環境）

## 環境設置

### 方法 1：標準 Python 環境

#### 1. 克隆倉庫

```bash
git clone https://github.com/opendatalab/OmniDocBench.git
cd OmniDocBench
```

#### 2. 創建虛擬環境（推薦）

```bash
# 使用 venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 使用 conda
conda create -n omnidocbench python=3.8
conda activate omnidocbench
```

#### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 4. 驗證安裝

```bash
python -c "import numpy, pandas, opencv; print('Installation successful!')"
```

### 方法 2：使用 Docker

**Docker 環境**（v1.5+ 支持）

```bash
# 構建 Docker 鏡像
docker build -t omnidocbench:latest .

# 運行容器
docker run -it --rm \
  -v $(pwd):/workspace \
  omnidocbench:latest
```

## 項目結構導覽

```
OmniDocBench/
├── configs/          # 評估配置
├── dataset/          # 數據加載
├── metrics/          # 評估指標
├── task/             # 評估任務
├── utils/            # 工具函數
├── demo_data/        # 演示數據
├── result/           # 結果輸出
└── requirements.txt  # 依賴清單
```

## 基本使用

### 1. 準備數據

#### 下載數據集

- **Hugging Face**：https://huggingface.co/datasets/opendatalab/OmniDocBench
- **OpenDataLab**：https://opendatalab.com/OpenDataLab/OmniDocBench

#### 數據集結構

```
demo_data/
└── omnidocbench_demo/
    ├── OmniDocBench_demo.json  # Ground Truth 標註
    ├── images/                  # PDF 圖像
    └── mds/                     # Markdown 標註
```

### 2. 準備模型預測

將您的模型輸出放在一個目錄中：

```
demo_data/
└── end2end/
    ├── page_001.md
    ├── page_002.md
    └── ...
```

### 3. 配置評估

編輯或創建配置文件：`configs/end2end.yaml`

```yaml
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
    display_formula:
      metric:
        - Edit_dist
        - CDM_plain
    table:
      metric:
        - TEDS
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json
    prediction:
      data_path: ./demo_data/end2end
    match_method: quick_match
```

### 4. 運行評估

```bash
# 端到端評估
python run_eval.py --config configs/end2end.yaml

# 公式識別評估
python run_eval.py --config configs/formula_recognition.yaml

# 版面檢測評估
python run_eval.py --config configs/layout_detection.yaml
```

### 5. 查看結果

結果保存在 `result/` 目錄：

```
result/
├── end2end_quick_match_text_block_result.json
├── end2end_quick_match_formula_result.json
├── end2end_quick_match_table_result.json
└── end2end_quick_match_metric_result.json  # 彙總結果
```

## 評估任務類型

### 1. 端到端評估（End-to-End）

評估完整的文檔解析流程。

```bash
python run_eval.py --config configs/end2end.yaml
```

**評估內容**：
- 文本塊識別和 OCR
- 公式識別
- 表格識別
- 閱讀順序

**支持的指標**：Edit Distance, BLEU, METEOR, TEDS, CDM

### 2. 版面檢測（Layout Detection）

評估文檔版面元素的檢測性能。

```bash
python run_eval.py --config configs/layout_detection.yaml
```

**評估內容**：
- 邊界框位置精度
- 類別分類準確性

**支持的指標**：COCO mAP, mAR

### 3. 公式識別（Formula Recognition）

評估數學公式的識別準確性。

```bash
python run_eval.py --config configs/formula_recognition.yaml
```

**支持的指標**：Edit Distance, CDM

### 4. 表格識別（Table Recognition）

評估表格結構和內容的識別。

```bash
python run_eval.py --config configs/table_recognition.yaml
```

**支持的指標**：TEDS, Edit Distance

### 5. 文字 OCR

評估文字識別準確性。

```bash
python run_eval.py --config configs/ocr.yaml
```

**支持的指標**：Edit Distance, BLEU, METEOR

## 高級配置

### 數據過濾

按頁面屬性過濾數據集：

```yaml
dataset:
  filter:
    language: english     # 或 chinese
    doc_type: scientific_paper
    layout_type: single_column
```

### 分組統計

按屬性分組計算指標：

```yaml
metrics:
  text_block:
    metric:
      - Edit_dist
    group:
      - language: english
      - doc_type: financial_report
```

### 匹配方法選擇

```yaml
dataset:
  match_method: quick_match  # 或 full_match, simple_match
```

- **quick_match**：快速匹配，適合大規模評估
- **full_match**：完整匹配，使用匈牙利算法
- **simple_match**：簡單匹配，適合特定場景

## 添加新組件

### 添加新的評估指標

#### 1. 創建指標類

在 `metrics/` 目錄中創建新文件：

```python
# metrics/my_metric.py
from registry.registry import METRIC_REGISTRY
from collections import defaultdict

@METRIC_REGISTRY.register("MyMetric")
class call_MyMetric:
    def __init__(self, samples):
        self.samples = samples

    def evaluate(self, group_info=[], save_name='default'):
        group_scores = defaultdict(list)

        for sample in self.samples:
            gt = sample['gt']
            pred = sample['pred']

            # 計算您的指標
            score = compute_my_metric(gt, pred)

            group_scores['all'].append(score)
            sample['metric']['MyMetric'] = score

        # 計算平均值
        result = {
            'MyMetric': sum(group_scores['all']) / len(group_scores['all'])
        }

        return self.samples, result
```

#### 2. 在 `metrics/__init__.py` 中導入

```python
from .my_metric import call_MyMetric

__all__ = [
    ...,
    "call_MyMetric"
]
```

#### 3. 在配置文件中使用

```yaml
metrics:
  text_block:
    metric:
      - MyMetric
```

### 添加新的評估任務

#### 1. 創建任務類

在 `task/` 目錄中創建新文件：

```python
# task/my_task_eval.py
from registry.registry import EVAL_TASK_REGISTRY
from registry.registry import METRIC_REGISTRY

@EVAL_TASK_REGISTRY.register("my_task_eval")
class MyTaskEval:
    def __init__(self, dataset, metrics_list, page_info_path, save_name):
        samples = dataset.samples
        result = {}

        for metric in metrics_list:
            metric_val = METRIC_REGISTRY.get(metric)
            samples, score = metric_val(samples).evaluate({}, save_name)
            result.update(score)

        # 顯示和保存結果
        print(result)
        # ...
```

#### 2. 在 `task/__init__.py` 中導入

```python
from .my_task_eval import MyTaskEval

__all__ = [
    ...,
    "MyTaskEval"
]
```

#### 3. 創建配置文件

```yaml
# configs/my_task.yaml
my_task_eval:
  metrics:
    - MyMetric
  dataset:
    dataset_name: my_dataset
    ground_truth:
      data_path: ./data/gt.json
    prediction:
      data_path: ./data/pred/
```

### 添加新的數據集加載器

#### 1. 創建數據集類

```python
# dataset/my_dataset.py
from registry.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("my_dataset")
class MyDataset:
    def __init__(self, cfg_task):
        # 加載和處理數據
        self.samples = self.load_data(cfg_task)

    def load_data(self, cfg):
        # 實現數據加載邏輯
        return samples
```

#### 2. 在 `dataset/__init__.py` 中導入

```python
from .my_dataset import MyDataset

__all__ = [
    ...,
    "MyDataset"
]
```

## 測試

### 運行測試

使用演示數據測試評估流程：

```bash
# 測試端到端評估
python run_eval.py --config configs/end2end.yaml

# 檢查結果
ls result/
```

### 驗證輸出

檢查 `result/` 目錄中的 JSON 文件：

```bash
# 查看彙總結果
cat result/end2end_quick_match_metric_result.json | jq .
```

## 常見任務

### 評估您自己的模型

1. **運行模型推理**：生成預測結果
2. **組織輸出**：按評估格式組織文件
3. **配置路徑**：在配置文件中設置 `prediction.data_path`
4. **運行評估**：執行評估腳本
5. **分析結果**：查看 `result/` 目錄中的輸出

### 比較多個模型

```bash
# 評估模型 A
python run_eval.py --config configs/end2end.yaml \
  --pred_path ./predictions/modelA \
  --save_name modelA_eval

# 評估模型 B
python run_eval.py --config configs/end2end.yaml \
  --pred_path ./predictions/modelB \
  --save_name modelB_eval

# 比較結果
python tools/compare_results.py \
  result/modelA_eval_metric_result.json \
  result/modelB_eval_metric_result.json
```

### 僅在特定數據子集上評估

使用配置文件中的過濾器：

```yaml
dataset:
  filter:
    language: english
    doc_type: scientific_paper
```

### 生成評估報告

```bash
# 生成 HTML 報告（如果支持）
python tools/generate_report.py \
  --result result/end2end_quick_match_metric_result.json \
  --output report.html
```

## 故障排除

### 問題：安裝失敗

```bash
# 更新 pip
pip install --upgrade pip

# 逐個安裝依賴以識別問題
pip install pandas
pip install numpy
# ...
```

### 問題：CUDA/GPU 錯誤

某些依賴可能需要特定的 CUDA 版本。查看依賴項的文檔以獲取兼容性信息。

### 問題：內存不足

- **減少批量大小**
- **使用較小的數據子集進行測試**
- **增加系統交換空間**

### 問題：CDM 計算緩慢

CDM 涉及 LaTeX 渲染和視覺匹配，計算密集：

- 使用 `CDM_plain` 僅輸出匹配對（後處理計算）
- 考慮並行處理
- 使用更快的硬件

### 問題：JSON 解析錯誤

確保預測輸出格式正確：

```bash
# 驗證 JSON
python -m json.tool your_prediction.json
```

## 性能優化

### 並行評估

```python
# 在 metrics/cal_metric.py 中
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(compute_metric, sample)
               for sample in samples]
    results = [f.result() for f in futures]
```

### 緩存結果

避免重複計算：

```python
import functools

@functools.lru_cache(maxsize=1000)
def expensive_computation(input_hash):
    # 計算
    return result
```

## 開發工作流程

### 典型開發循環

1. **修改代碼**：在相應的模塊中進行更改
2. **測試**：使用演示數據運行評估
3. **調試**：使用 `pdb` 或日誌進行調試
4. **驗證**：檢查結果的正確性
5. **提交**：提交經過測試的更改

### 調試技巧

#### 使用 Python 調試器

```python
import pdb

# 在代碼中設置斷點
pdb.set_trace()
```

#### 使用日誌

```python
from loguru import logger

logger.debug("調試信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("錯誤信息")
```

#### 打印變量

```python
print(f"Variable value: {variable}")
print(f"Type: {type(variable)}")
```

## 代碼風格

### Python 風格指南

遵循 PEP 8 風格指南：

```bash
# 安裝 linter
pip install flake8 black

# 檢查代碼風格
flake8 your_file.py

# 自動格式化
black your_file.py
```

### 文檔字符串

使用清晰的文檔字符串：

```python
def my_function(arg1, arg2):
    """
    簡短描述函數作用。

    Args:
        arg1 (type): 參數 1 描述
        arg2 (type): 參數 2 描述

    Returns:
        type: 返回值描述
    """
    pass
```

## 貢獻

### 貢獻流程

1. **Fork 倉庫**
2. **創建分支**：`git checkout -b feature/my-feature`
3. **進行更改**
4. **測試更改**：確保所有測試通過
5. **提交更改**：`git commit -m "Add my feature"`
6. **推送分支**：`git push origin feature/my-feature`
7. **創建 Pull Request**

### 貢獻者許可協議（CLA）

閱讀並簽署 `OmniDocBench_CLA.md`

### 代碼審查

所有 PR 都需要代碼審查才能合併。

## 相關資源

### 文檔

- [項目概覽](./project-overview.md)
- [架構文檔](./architecture.md)
- [原始碼樹分析](./source-tree-analysis.md)
- [主要 README](../README.md)
- [CDM 指標文檔](../metrics/cdm/README.md)

### 外部資源

- [論文（arXiv）](https://arxiv.org/abs/2412.07626)
- [Hugging Face 數據集](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- [OpenDataLab 數據集](https://opendatalab.com/OpenDataLab/OmniDocBench)
- [GitHub 倉庫](https://github.com/opendatalab/OmniDocBench)

### 社區支持

- **Issues**：https://github.com/opendatalab/OmniDocBench/issues
- **Discussions**：https://github.com/opendatalab/OmniDocBench/discussions

## 版本控制

### Git 工作流程

```bash
# 克隆倉庫
git clone https://github.com/opendatalab/OmniDocBench.git

# 創建功能分支
git checkout -b feature/my-feature

# 進行更改並提交
git add .
git commit -m "Descriptive commit message"

# 推送到遠程
git push origin feature/my-feature
```

### 分支策略

- `main` - 穩定版本
- `v1_0` - 版本 1.0 分支
- `feature/*` - 新功能開發
- `bugfix/*` - 錯誤修復

## 常見問題 (FAQ)

### Q: 如何添加新的評估指標？

參見"添加新組件"部分中的"添加新的評估指標"。

### Q: 支持哪些數據格式？

- Ground Truth：JSON 格式
- Prediction：Markdown, JSON, COCO 格式

### Q: 如何處理大數據集？

- 使用過濾器僅評估子集
- 增加系統內存
- 使用批量處理

### Q: CDM 指標是什麼？

CDM（Character Detection Metric）是一種基於視覺匹配的公式評估指標，通過渲染 LaTeX 為圖像並計算字符級別的檢測準確率。

### Q: 如何獲取完整數據集？

從 Hugging Face 或 OpenDataLab 下載：
- [Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- [OpenDataLab](https://opendatalab.com/OpenDataLab/OmniDocBench)

---

*本文檔由 BMM document-project workflow 自動生成*
*生成日期：2025-11-11*
