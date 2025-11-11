# OmniDocBench 項目概覽

## 執行摘要

**OmniDocBench** 是一個綜合性的文檔解析評估基準測試系統，專門設計用於評估真實場景中多樣化的文檔解析任務。該項目包含 1355 個 PDF 頁面，涵蓋 9 種文檔類型、4 種版面類型和 3 種語言類型。

### 核心能力

- **多維度評估**：支持端到端評估、版面檢測、表格識別、公式識別和文字 OCR
- **豐富的註釋資訊**：包含 15 種塊級註釋（20k+）和 4 種跨度級註釋（80k+）
- **高質量數據集**：通過人工篩選、智能標註、專家審核和大型模型質檢確保品質
- **靈活的評估框架**：基於配置驅動的模塊化架構，支持多種評估指標

## 項目分類

| 屬性 | 值 |
|------|-----|
| **倉庫類型** | Monolith（單一代碼庫） |
| **項目類型** | 數據處理與評估管道 |
| **主要語言** | Python 3.x |
| **領域** | 文檔智能、計算機視覺、OCR |

## 技術棧摘要

### 核心框架

- **數據處理**：pandas (2.0.3), numpy (1.24.4), datasets (3.1.0)
- **計算機視覺**：opencv-python (4.10.0.84), Pillow (10.4.0)
- **評估框架**：mmeval (0.2.1), evaluate (0.4.3), scikit-learn (1.1.2)
- **檢測指標**：pycocotools (2.0.7)
- **文本處理**：nltk (3.9.1), Levenshtein (0.25.1), rapidfuzz (3.9.7)
- **LaTeX 支援**：pylatexenc (3.0a30)

### 架構模式

- **註冊表模式（Registry Pattern）**：動態註冊和查找評估任務、指標和數據集
- **管道架構（Pipeline Architecture）**：配置驅動的評估流程
- **模塊化設計**：task、metrics、dataset、utils 模塊清晰分離

## 項目結構

```
OmniDocBench/
├── configs/          # 評估配置文件（YAML）
├── dataset/          # 數據集加載器
├── metrics/          # 評估指標實現
├── task/             # 評估任務定義
├── utils/            # 工具函數和匹配算法
├── demo_data/        # 演示數據
├── result/           # 評估結果輸出
├── tools/            # 推理工具
└── registry/         # 註冊表系統
```

## 評估類別

OmniDocBench 支持以下評估類別：

1. **端到端評估**（end2end）：完整的文檔解析流程評估
2. **版面檢測**（layout_detection）：文檔版面元素檢測
3. **表格識別**（table_recognition）：表格結構和內容識別
4. **公式識別**（formula_recognition）：數學公式識別
5. **文字 OCR**（ocr）：文字識別
6. **公式檢測**（formula_detection）：公式位置檢測

## 支持的評估指標

- **Normalized Edit Distance**：標準化編輯距離
- **BLEU**：雙語評估替換（機器翻譯指標）
- **METEOR**：基於對齊的翻譯評估指標
- **TEDS**：樹編輯距離相似度（表格評估）
- **CDM**：字符檢測指標（公式評估）
- **COCODet**：mAP, mAR 等檢測指標

## 數據集特點

- **1355 個 PDF 頁面**
- **9 種文檔類型**：學術論文、財務報告、報紙、教科書、手寫筆記等
- **豐富的標註**：
  - 15 種塊級標註：文本段落、標題、表格等（20k+）
  - 4 種跨度級標註：文本行、內聯公式、下標等（80k+）
  - 閱讀順序標註
  - 5 種頁面屬性標籤
  - 3 種文本屬性標籤
  - 6 種表格屬性標籤

## 關鍵特性

### 1. 配置驅動評估

通過 YAML 配置文件定義評估任務：
- 選擇評估指標
- 指定數據路徑
- 配置匹配方法
- 設置過濾條件

### 2. 註冊表系統

使用裝飾器模式註冊組件：
- `@EVAL_TASK_REGISTRY.register()` - 評估任務
- `@METRIC_REGISTRY.register()` - 評估指標
- `@DATASET_REGISTRY.register()` - 數據集加載器

### 3. 靈活的匹配策略

支持多種 GT-Prediction 匹配方法：
- **quick_match**：快速匹配算法
- **full_match**：完整匹配算法
- **simple_match**：簡單匹配算法

### 4. 多語言支持

- 英文文檔
- 簡體中文文檔
- 混合語言數據集

## 快速參考

| 資源 | 位置 |
|------|------|
| **配置文件** | `configs/` |
| **評估任務** | `task/` |
| **評估指標** | `metrics/` |
| **數據集加載** | `dataset/` |
| **工具函數** | `utils/` |
| **演示數據** | `demo_data/` |
| **結果輸出** | `result/` |
| **主要 README** | `README.md` (英文), `README_zh-CN.md` (中文) |
| **依賴清單** | `requirements.txt` |

## 文檔鏈接

- [README（英文）](../README.md)
- [README（簡體中文）](../README_zh-CN.md)
- [CDM 指標文檔](../metrics/cdm/README.md)
- [原始碼樹狀結構分析](./source-tree-analysis.md)
- [架構文檔](./architecture.md)
- [開發指南](./development-guide.md)

## 研究與發表

- **論文**：[arXiv:2412.07626](https://arxiv.org/abs/2412.07626)
- **數據集**：[Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench) | [OpenDataLab](https://opendatalab.com/OpenDataLab/OmniDocBench)
- **接受會議**：CVPR 2025

## 版本資訊

**當前版本**：v1.5

主要更新：
- 更新混合匹配算法
- 整合 CDM 計算到指標模塊
- 提高圖像解析度（報紙和筆記類型：72 DPI → 200 DPI）
- 新增 374 個頁面
- 平衡中英文頁面數量
- 修正評分計算公式

## 使用場景

OmniDocBench 適用於：

1. **模型評估**：評估文檔解析模型的性能
2. **基準比較**：與其他模型進行公平比較
3. **研究開發**：開發和測試新的文檔解析算法
4. **質量控制**：確保文檔處理管道的準確性

---

*本文檔由 BMM document-project workflow 自動生成*
*生成日期：2025-11-11*
