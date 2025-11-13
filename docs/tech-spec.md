# OmniDocBench - Technical Specification

**Author:** BMad
**Date:** 2025-11-13
**Project Level:** 0
**Change Type:** New Feature
**Development Context:** Brownfield

---

## Context

### Available Documents

**已找到的文件:**

✅ **專案文件分析 (document-project)**: 已完成
- `docs/index.md` - 完整的專案索引
- `docs/architecture.md` - 架構文檔
- `docs/development-guide.md` - 開發指南
- `docs/project-overview.md` - 專案概覽
- `docs/source-tree-analysis.md` - 原始碼樹狀結構

❌ **產品簡報 (Product Brief)**: 未找到
❌ **研究文件 (Research)**: 未找到

**文件品質**: 技術文檔非常完整，涵蓋架構、開發指南、API 設計等各方面。

### Project Stack

**專案名稱**: OmniDocBench
**專案類型**: 數據處理與評估管道 (Monolith 架構)
**版本**: v1.5
**主要語言**: Python 3.x (推薦 3.8+)

**核心框架與依賴版本**:

**數據處理層**:
- pandas 2.0.3 - 數據分析和處理
- numpy 1.24.4 - 數值計算基礎
- datasets 3.1.0 - HuggingFace 數據集支持
- pyarrow 17.0.0 - 高效數據序列化

**計算機視覺層**:
- opencv-python 4.10.0.84 - 圖像處理
- Pillow 10.4.0 - 圖像操作
- matplotlib 3.7.5 - 可視化

**評估指標層**:
- mmeval 0.2.1 - 多模態評估框架
- evaluate 0.4.3 - HuggingFace 評估工具
- scikit-learn 1.1.2 - 機器學習指標
- pycocotools 2.0.7 - COCO 格式檢測評估

**文本處理層**:
- nltk 3.9.1 - 自然語言處理
- Levenshtein 0.25.1 - 編輯距離計算
- rapidfuzz 3.9.7 - 快速模糊匹配
- pylatexenc 3.0a30 - LaTeX 編碼處理

**配置和工具層**:
- PyYAML 6.0.2 - YAML 解析
- click 8.1.7 - CLI 構建
- loguru 0.7.2 - 日誌記錄
- tqdm 4.67.1 - 進度條

**API 調用層**:
- openai - OpenAI SDK (需確認是否已安裝)

**測試框架**: 基於 demo_data 進行驗證測試（無正式單元測試框架）

### Existing Codebase Structure

**專案類型**: Brownfield - 現有成熟的程式碼庫

**架構模式**:
- 📦 **註冊表模式 (Registry Pattern)** - 動態組件管理
- 🔄 **管道架構 (Pipeline Architecture)** - 線性評估流程
- 🧩 **模塊化設計** - 清晰的層次分離

**目錄結構**:
```
OmniDocBench/
├── configs/          # YAML 配置文件
│   └── DeepSeek-OCR-vllm/  # DeepSeek-OCR 本地配置
├── task/             # 評估任務實現
├── dataset/          # 數據集加載器
├── metrics/          # 評估指標
├── utils/            # 工具函數和匹配算法
├── tools/
│   └── model_infer/  # 模型推理腳本 ⭐ (我們的目標位置)
├── registry/         # 註冊表系統
├── demo_data/        # 演示數據
└── result/           # 評估結果輸出
```

**關鍵設計模式**:
1. **裝飾器註冊** - 使用 `@REGISTRY.register()` 註冊組件
2. **YAML 配置驅動** - 所有評估通過 YAML 文件配置
3. **統一 API 接口** - 所有指標遵循相同的 `evaluate()` 接口
4. **GT-Pred 匹配算法** - quick_match, full_match 等匹配策略

**現有推理腳本**:
- `tools/model_infer/gpt_4o_inf.py` - GPT-4o 通過 OpenAI API 調用
- `configs/run_dpsk_ocr_eval_batch.py` - DeepSeek-OCR 本地 vllm 批處理

---

## The Change

### Problem Statement

目前 OmniDocBench 已支持通過 OpenAI API 調用 GPT-4o 進行文檔解析評估（`gpt_4o_inf.py`），也支持本地 vllm 批處理 DeepSeek-OCR（`run_dpsk_ocr_eval_batch.py`）。

但缺少 **通過 OpenAI Compatible API 調用遠程 vllm DeepSeek-OCR 服務** 的方式。這導致：

1. 無法利用已架設的遠程 vllm 服務
2. 必須在本地環境安裝完整的 vllm + torch + transformers 依賴
3. 無法使用輕量級客戶端進行批量推理

### Proposed Solution

新增 `tools/model_infer/deepseek_ocr_inf.py` 推理腳本，通過 OpenAI Compatible API 調用已架設的 vllm DeepSeek-OCR 服務。

**核心特性**:
1. ✅ 使用 OpenAI SDK 調用 vllm API（base64 圖像傳輸）
2. ✅ 集成 DeepSeek-OCR 專用後處理邏輯
3. ✅ 生成雙輸出：原始 + 清理後的 Markdown
4. ✅ 支持多線程並行處理
5. ✅ 完全參數化配置（API endpoint, model name, threads）

**技術優勢**:
- 客戶端輕量化（不需要 torch/transformers）
- 圖像預處理由 vllm 服務端處理（已配置 DeepseekOCRProcessor）
- 與現有 `gpt_4o_inf.py` 架構一致，易於維護

### Scope

**In Scope:**

✅ 新建 `tools/model_infer/deepseek_ocr_inf.py`
✅ 使用 OpenAI Compatible API 調用 vllm
✅ Base64 圖像編碼和傳輸
✅ DeepSeek-OCR 專用 prompt: `'Convert the document to markdown.'`
✅ 生成雙輸出:
  - `{basename}_det.md` - 原始 API 輸出（含特殊標記）
  - `{basename}.md` - 清理後輸出（移除標記、清理公式）
✅ 集成後處理函數:
  - `clean_formula()` - 清理公式中的 `\quad(...)` 標記
  - `re_match()` - 提取並移除 `<|ref|>...<|det|>` 標記
✅ 多線程並行處理 + 進度條
✅ 命令行參數配置
✅ 錯誤處理和日誌

**Out of Scope:**

❌ 不修改現有的 `gpt_4o_inf.py`
❌ 不修改現有的 `run_dpsk_ocr_eval_batch.py`
❌ 不包含客戶端圖像預處理邏輯（由 vllm 服務端處理）
❌ 不包含 vllm 服務配置或部署說明
❌ 不添加新的評估指標或任務
❌ 不修改註冊表系統或核心架構

---

## Implementation Details

### Source Tree Changes

**新建文件**:
```
tools/model_infer/
├── gpt_4o_inf.py              # 現有（參考模板）
├── deepseek_ocr_inf.py        # 新建 ⭐
└── ...其他推理腳本
```

**操作**: CREATE `tools/model_infer/deepseek_ocr_inf.py`

**文件用途**: 通過 OpenAI Compatible API 批量調用 vllm DeepSeek-OCR 服務，生成文檔 Markdown 輸出

### Technical Approach

**設計決策 - 為什麼選擇 OpenAI Compatible API?**

1. **服務端處理圖像預處理** ✅
   - vllm 服務已配置 `DeepseekOCRProcessor`
   - 自動處理動態裁剪、padding、normalization
   - 客戶端只需發送原始 base64 圖像

2. **輕量級客戶端** ✅
   - 不需要安裝 torch (1.5GB+), transformers, vllm
   - 只需 openai SDK + 基礎依賴
   - 適合在任何環境運行

3. **標準化接口** ✅
   - 遵循 OpenAI API 標準
   - 與 `gpt_4o_inf.py` 架構一致
   - 易於維護和擴展

**核心架構**:

```python
# 主要組件
main()                      # 命令行 + 多線程協調
├── process_image()         # 單圖像處理（ThreadPool 調用）
    ├── get_deepseek_response()  # API 調用
    │   └── OpenAI SDK -> vllm API
    ├── 保存原始輸出 (_det.md)
    └── 後處理並保存 (.md)
        ├── clean_formula()      # 公式清理
        └── re_match() + 移除標記  # 特殊標記處理
```

**API 調用流程**:

```python
# 1. 圖像編碼
with open(image_path, "rb") as f:
    image_bytes = f.read()
img_str = base64.b64encode(image_bytes).decode()

# 2. 調用 OpenAI Compatible API
client = OpenAI(api_key=api_key, base_url=base_url)
completion = client.chat.completions.create(
    model=model_name,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            },
            {"type": "text", "text": PROMPT}
        ]
    }],
    temperature=0.0
)

# 3. 獲取響應
response = completion.choices[0].message.content
```

**後處理邏輯** (從 `run_dpsk_ocr_eval_batch.py` 移植):

```python
# 1. 清理公式
def clean_formula(text):
    """移除公式中的 \\quad(...) 標記"""
    formula_pattern = r'\\\[(.*?)\\\]'

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        return r'\[' + formula.strip() + r'\]'

    return re.sub(formula_pattern, process_formula, text)

# 2. 提取並移除特殊標記
def re_match(text):
    """提取 <|ref|>...<|det|> 標記"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    matches_other = [match[0] for match in matches]
    return matches, matches_other

# 3. 清理換行
cleaned = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
cleaned = cleaned.replace('<center>', '').replace('</center>', '')
```

**Prompt 使用**:

```python
PROMPT = 'Convert the document to markdown.'
```

簡化的 prompt 適用於 OpenAI Compatible API 調用：
- 圖像通過 API `messages` 結構的 `image_url` 傳遞，不需要 `<image>` 標記
- `<|grounding|>` 是 DeepSeek-OCR 本地調用時的模式標記，OpenAI API 可能自動處理或不需要
- 保持 prompt 簡潔清晰

### Existing Patterns to Follow

**從 `gpt_4o_inf.py` 遵循的模式**:

1. **OpenAI SDK 使用**:
```python
from openai import OpenAI
client = OpenAI(api_key=..., base_url=...)
```

2. **Base64 圖像編碼**:
```python
with open(image_path, "rb") as f:
    image_bytes = f.read()
img_str = base64.b64encode(image_bytes).decode()
```

3. **多線程並行處理**:
```python
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(tqdm(
        executor.map(process_image, image_files),
        total=len(image_files),
        desc="處理進度"
    ))
```

4. **錯誤處理**:
```python
try:
    response = get_gpt_response(image_path)
except Exception as e:
    print(f"[ERROR] Failed to get response: {e}")
    return ""
```

5. **命令行參數**:
```python
parser.add_argument("--image_root", type=str, help="圖像文件夾路徑")
parser.add_argument("--save_root", type=str, help="保存結果的文件夾路徑")
parser.add_argument("--threads", type=int, default=10, help="並行線程數")
```

**從 `run_dpsk_ocr_eval_batch.py` 遵循的模式**:

1. **雙輸出生成**:
```python
# 原始輸出
mmd_det_path = output_path + image_name.replace('.jpg', '_det.md')
with open(mmd_det_path, 'w', encoding='utf-8') as f:
    f.write(raw_content)

# 清理後輸出
mmd_path = output_path + image_name.replace('.jpg', '.md')
with open(mmd_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)
```

2. **後處理函數**: `clean_formula()`, `re_match()` 完全移植

3. **進度條和日誌**: 使用 tqdm 顯示處理進度

### Integration Points

**外部依賴**:
- **vllm API 服務** - 必須已架設並配置 DeepSeek-OCR
  - 需要支持 OpenAI Compatible API (`/v1/chat/completions`)
  - 需要配置 `trust_remote_code=True`
  - 需要配置 `DeepseekOCRProcessor` 進行圖像預處理

**輸入**:
- 圖像文件夾（支援 `.jpg`, `.png`, `.jpeg`）
- API 配置（base_url, api_key, model_name）

**輸出**:
- `{basename}_det.md` - 原始 API 輸出
- `{basename}.md` - 清理後的 Markdown

**不與其他模塊交互**:
- 這是獨立的推理腳本
- 不調用註冊表系統
- 不調用評估任務或指標
- 只負責: 圖像 → API → Markdown 文件

---

## Development Context

### Relevant Existing Code

**參考文件 1**: `tools/model_infer/gpt_4o_inf.py`
- 位置: tools/model_infer/gpt_4o_inf.py:1-113
- 用途: OpenAI API 調用架構參考
- 關鍵函數:
  - `get_gpt_response()` - API 調用模板
  - `process_image()` - 單圖處理模板
  - `main()` - 多線程協調模板

**參考文件 2**: `configs/run_dpsk_ocr_eval_batch.py`
- 位置: configs/run_dpsk_ocr_eval_batch.py:53-162
- 用途: DeepSeek-OCR 後處理邏輯參考
- 關鍵函數:
  - `clean_formula()` (L53-68) - 公式清理
  - `re_match()` (L70-79) - 特殊標記提取
  - 雙輸出邏輯 (L145-161)

**參考文件 3**: `configs/DeepSeek-OCR-vllm/config.py`
- 位置: configs/DeepSeek-OCR-vllm/config.py:27
- 用途: DeepSeek-OCR 官方 prompt 參考
- 注意: 本地調用使用 `'<image>\n<|grounding|>Convert the document to markdown.'`，但 OpenAI API 調用簡化為 `'Convert the document to markdown.'`

### Dependencies

**Framework/Libraries:**

現有依賴（已在 requirements.txt）:
- Pillow 10.4.0 - 圖像讀取
- tqdm 4.67.1 - 進度條
- Python 3.7+ - 基礎運行環境

需要確認的依賴:
- **openai** - OpenAI SDK（需確認是否已安裝，可能需要添加到 requirements.txt）

不需要的依賴（vllm 服務端處理）:
- torch - vllm 服務端使用
- transformers - vllm 服務端使用
- vllm - vllm 服務端使用
- DeepseekOCRProcessor - vllm 服務端使用

**Internal Modules:**

無內部模塊依賴 - 這是獨立的推理腳本

### Configuration Changes

**可能需要更新** `requirements.txt`:

```txt
# 檢查是否已包含 openai
openai>=1.0.0  # 需要支持 chat.completions API
```

**無需修改其他配置文件**:
- 不修改 YAML 評估配置
- 不修改註冊表
- 不修改現有腳本

### Existing Conventions (Brownfield)

**代碼風格**:
- **語言**: Python 3.x
- **縮進**: 4 空格
- **引號**: 雙引號為主
- **命名**:
  - 函數: snake_case (`get_deepseek_response`, `process_image`)
  - 常量: UPPER_CASE (`PROMPT`, `API_KEY`)
  - 變量: snake_case

**文件組織**:
- 推理腳本放在 `tools/model_infer/`
- 配置腳本放在 `configs/`
- 輸出結果放在用戶指定的 `save_root`

**錯誤處理**:
- 使用 try-except 捕獲異常
- 打印 `[ERROR]` 前綴的錯誤信息
- 失敗時返回空字符串或跳過

**日誌風格**:
- 使用 `print()` 輸出日誌（推理腳本不使用 loguru）
- 使用 tqdm 顯示進度
- 顯示成功/失敗統計

**命令行參數風格**:
- 使用 `argparse`
- 提供 `--help` 說明
- 使用長參數名（`--image_root` 而非 `-i`）

### Test Framework & Standards

**測試方法**:
- **無正式單元測試框架**（項目使用 demo_data 驗證）
- 推理腳本通過實際運行驗證

**驗證方式**:
1. **功能測試**: 使用 `demo_data/omnidocbench_demo/images/` 中的圖像測試
2. **輸出驗證**: 檢查生成的 `_det.md` 和 `.md` 文件
3. **後處理驗證**: 確認特殊標記被正確移除
4. **錯誤處理驗證**: 測試 API 失敗、圖像讀取失敗等場景

**驗證清單**:
- ✅ API 調用成功
- ✅ 生成 `_det.md`（原始輸出）
- ✅ 生成 `.md`（清理後輸出）
- ✅ 公式清理正確（`\quad(...)` 被移除）
- ✅ 特殊標記被移除（`<|ref|>`, `<|det|>` 等）
- ✅ 多線程並行工作
- ✅ 進度條正常顯示
- ✅ 錯誤優雅處理

---

## Implementation Stack

**運行環境**: Python 3.8+

**核心依賴**:
- openai >= 1.0.0 - OpenAI SDK
- Pillow 10.4.0 - 圖像讀取
- tqdm 4.67.1 - 進度條
- Python 標準庫:
  - argparse - 命令行參數
  - base64 - 圖像編碼
  - os - 文件操作
  - re - 正則表達式
  - concurrent.futures - 多線程

**開發工具**:
- Git - 版本控制
- Python venv - 虛擬環境

**外部服務**:
- vllm API 服務 - 已架設的 DeepSeek-OCR 服務
  - 支持 OpenAI Compatible API
  - 配置 DeepseekOCRProcessor

---

## Technical Details

### 核心算法和邏輯

**1. 圖像編碼**:
```python
def encode_image_to_base64(image_path: str) -> str:
    """讀取圖像並編碼為 base64 字符串"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode()
```

**2. API 調用邏輯**:
```python
def get_deepseek_response(
    image_path: str,
    client: OpenAI,
    model_name: str
) -> str:
    """調用 vllm API 獲取響應"""
    # 1. 編碼圖像
    img_str = encode_image_to_base64(image_path)

    # 2. 構建請求
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    },
                    {"type": "text", "text": PROMPT}
                ]
            }],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Failed to get response: {e}")
        return ""
```

**3. 公式清理邏輯**:
```python
def clean_formula(text: str) -> str:
    """清理公式中的 \\quad(...) 標記

    示例:
    輸入: \\[ E = mc^2 \\quad(equation 1) \\]
    輸出: \\[ E = mc^2 \\]
    """
    formula_pattern = r'\\\[(.*?)\\\]'

    def process_formula(match):
        formula = match.group(1)
        # 移除 \quad(...) 模式
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'

    return re.sub(formula_pattern, process_formula, text)
```

**4. 特殊標記處理**:
```python
def re_match(text: str) -> tuple:
    """提取 <|ref|>...<|det|> 標記

    返回:
        (matches, matches_other)
        matches: 完整匹配對象列表
        matches_other: 需要移除的標記字符串列表
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_other = []
    for a_match in matches:
        matches_other.append(a_match[0])

    return matches, matches_other
```

**5. 完整處理流程**:
```python
def process_image(args) -> str:
    """處理單張圖像"""
    image_path, save_root, client, model_name = args
    file_name = os.path.basename(image_path)
    base_name = file_name.rsplit('.', 1)[0]

    try:
        # 1. 調用 API
        response = get_deepseek_response(image_path, client, model_name)

        # 2. 保存原始輸出 (_det.md)
        det_path = os.path.join(save_root, f"{base_name}_det.md")
        with open(det_path, "w", encoding="utf-8") as f:
            f.write(response)

        # 3. 清理並保存 (.md)
        cleaned = clean_formula(response)
        matches_ref, matches_other = re_match(cleaned)

        # 移除特殊標記
        for match in matches_other:
            cleaned = cleaned.replace(match, '')

        # 清理多餘換行
        cleaned = cleaned.replace('\n\n\n\n', '\n\n')
        cleaned = cleaned.replace('\n\n\n', '\n\n')
        cleaned = cleaned.replace('<center>', '').replace('</center>', '')

        # 保存清理後的輸出
        output_path = os.path.join(save_root, f"{base_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        return f"成功處理: {file_name}"

    except Exception as e:
        return f"處理失敗 {file_name}: {str(e)}"
```

### 性能考慮

**多線程並行**:
- 默認 10 個線程（可配置）
- I/O 密集型任務（API 調用、文件讀寫）適合多線程
- 避免 GIL 限制（大部分時間在等待 API 響應）

**內存使用**:
- 圖像編碼為 base64 會增加約 33% 大小
- 批量處理時注意內存使用
- 線程數過多可能導致內存壓力

**API 限速**:
- vllm 服務可能有並發限制
- 建議根據服務端配置調整 `--threads` 參數
- 錯誤處理包含重試邏輯（可選）

### 安全考慮

**API Key 保護**:
- API key 通過命令行參數傳入
- 不硬編碼在代碼中
- 建議使用環境變量: `export VLLM_API_KEY=xxx`

**文件操作安全**:
- 檢查輸出目錄存在性（`os.makedirs(exist_ok=True)`）
- 使用 UTF-8 編碼避免亂碼
- 異常處理防止單個文件失敗影響整體

### Edge Cases

**處理的邊界情況**:

1. **圖像文件不存在**:
```python
try:
    with open(image_path, "rb") as f:
        ...
except FileNotFoundError:
    return f"圖像不存在: {image_path}"
```

2. **API 調用失敗**:
```python
except Exception as e:
    print(f"[ERROR] API 調用失敗: {e}")
    return ""
```

3. **空響應**:
```python
if not response:
    print(f"[WARNING] 空響應: {file_name}")
    return
```

4. **特殊字符處理**:
- UTF-8 編碼處理中文、特殊符號
- 正則表達式處理 LaTeX 特殊字符

5. **文件名處理**:
```python
# 支持多種圖像格式
if file.endswith((".jpg", ".png", ".jpeg")):
    ...
```

---

## Development Setup

### 環境準備

```bash
# 1. 確認 Python 版本
python --version  # 需要 >= 3.8

# 2. 創建虛擬環境（推薦）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 確認 openai 已安裝（如果沒有則安裝）
pip show openai || pip install openai>=1.0.0

# 5. 驗證安裝
python -c "import openai, PIL, tqdm; print('依賴已就緒')"
```

### vllm 服務驗證

```bash
# 1. 測試 API 連通性
curl $VLLM_BASE_URL/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

# 2. 測試 chat completions endpoint
curl $VLLM_BASE_URL/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR",
    "messages": [{"role": "user", "content": "test"}]
  }'
```

### 測試數據準備

```bash
# 使用項目自帶的 demo 數據
ls demo_data/omnidocbench_demo/images/

# 或準備自己的測試圖像
mkdir test_images
cp your_document.jpg test_images/
```

---

## Implementation Guide

### Setup Steps

開始實作前的檢查清單:

1. ✅ **環境驗證**:
   - [ ] Python >= 3.8 已安裝
   - [ ] 虛擬環境已創建並激活
   - [ ] requirements.txt 依賴已安裝
   - [ ] openai SDK 已安裝

2. ✅ **vllm 服務驗證**:
   - [ ] vllm API 服務正在運行
   - [ ] API endpoint 可訪問
   - [ ] API key 有效
   - [ ] DeepSeek-OCR 模型已加載

3. ✅ **項目準備**:
   - [ ] Git 工作目錄乾淨
   - [ ] 創建功能分支（可選）
   - [ ] 閱讀參考文件（`gpt_4o_inf.py`, `run_dpsk_ocr_eval_batch.py`）

4. ✅ **測試數據準備**:
   - [ ] 準備測試圖像（5-10 張）
   - [ ] 創建輸出目錄

### Implementation Steps

**步驟 1: 創建文件骨架**

創建 `tools/model_infer/deepseek_ocr_inf.py`，包含:
- 導入必要的庫
- 定義 PROMPT 常量
- 添加 `if __name__ == "__main__":` 入口

**步驟 2: 實現後處理函數**

從 `run_dpsk_ocr_eval_batch.py` 移植:
- `clean_formula()` 函數
- `re_match()` 函數

**步驟 3: 實現 API 調用函數**

參考 `gpt_4o_inf.py` 實現:
- `get_deepseek_response()` - API 調用
  - 圖像編碼
  - OpenAI SDK 調用
  - 錯誤處理

**步驟 4: 實現單圖處理函數**

實現 `process_image()`:
- 調用 `get_deepseek_response()`
- 保存原始輸出 `_det.md`
- 應用後處理
- 保存清理後輸出 `.md`
- 返回處理狀態

**步驟 5: 實現主函數**

實現 `main()`:
- 命令行參數解析
- 初始化 OpenAI client
- 收集圖像文件列表
- 使用 ThreadPoolExecutor 並行處理
- 顯示進度條
- 統計並顯示結果

**步驟 6: 測試和調試**

- 使用 1 張圖像測試
- 檢查雙輸出文件
- 驗證後處理邏輯
- 測試錯誤處理

**步驟 7: 批量測試**

- 使用 10-20 張圖像測試
- 驗證多線程穩定性
- 檢查輸出質量

### Testing Strategy

**測試層級**:

1. **單元測試（手動）**:
```python
# 測試公式清理
text = r'\[ E = mc^2 \quad(equation 1) \]'
result = clean_formula(text)
assert result == r'\[ E = mc^2 \]'

# 測試特殊標記提取
text = '<|ref|>test<|/ref|><|det|>bbox<|/det|>'
matches, matches_other = re_match(text)
assert len(matches_other) > 0
```

2. **集成測試**:
```bash
# 測試單圖處理
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root test_images \
  --save_root test_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 1

# 檢查輸出
ls test_output/
cat test_output/test_det.md
cat test_output/test.md
```

3. **批量測試**:
```bash
# 測試多線程
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root demo_data/omnidocbench_demo/images \
  --save_root result/deepseek_test \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 10
```

4. **錯誤場景測試**:
- 不存在的圖像路徑
- 錯誤的 API endpoint
- 錯誤的 API key
- 網絡超時

### Acceptance Criteria

**必須滿足的標準**:

1. ✅ **文件創建**: `tools/model_infer/deepseek_ocr_inf.py` 存在且可執行

2. ✅ **API 調用**: 成功通過 OpenAI Compatible API 調用 vllm
   - 正確的 endpoint 格式
   - 正確的請求結構
   - 正確處理響應

3. ✅ **雙輸出生成**:
   - 每個圖像生成 `{basename}_det.md`（原始輸出）
   - 每個圖像生成 `{basename}.md`（清理後輸出）
   - 兩個文件內容不同

4. ✅ **後處理正確**:
   - `clean_formula()` 正確移除 `\quad(...)` 標記
   - `re_match()` 正確提取特殊標記
   - 特殊標記被完全移除
   - 多餘換行被清理

5. ✅ **並行處理**:
   - 多線程正常工作
   - 顯示 tqdm 進度條
   - 線程數可配置

6. ✅ **錯誤處理**:
   - API 失敗時優雅處理（不崩潰）
   - 圖像讀取失敗時跳過
   - 顯示錯誤日誌

7. ✅ **參數化配置**:
   - 所有關鍵參數可通過命令行設置
   - 提供 `--help` 說明
   - 參數驗證（必填參數）

8. ✅ **結果統計**:
   - 顯示總圖像數
   - 顯示成功/失敗數量
   - 顯示處理時間（可選）

**驗證方法**:

```bash
# 1. 運行腳本
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root demo_data/omnidocbench_demo/images \
  --save_root result/deepseek_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --model_name deepseek-ai/DeepSeek-OCR \
  --threads 10

# 2. 檢查輸出
ls result/deepseek_output/ | wc -l  # 應該是圖像數 * 2

# 3. 驗證內容
head result/deepseek_output/*_det.md  # 應包含特殊標記
head result/deepseek_output/*.md      # 不應包含特殊標記

# 4. 檢查公式清理
grep "\\quad" result/deepseek_output/*.md  # 應該沒有結果
```

---

## Developer Resources

### File Paths Reference

**新建文件**:
```
tools/model_infer/deepseek_ocr_inf.py  # 主推理腳本
```

**參考文件**:
```
tools/model_infer/gpt_4o_inf.py                    # OpenAI API 調用參考
configs/run_dpsk_ocr_eval_batch.py                 # 後處理邏輯參考
configs/DeepSeek-OCR-vllm/config.py                # Prompt 配置參考
configs/DeepSeek-OCR-vllm/process/image_process.py # 圖像預處理參考（不使用）
```

**測試數據**:
```
demo_data/omnidocbench_demo/images/  # 官方演示圖像
```

**輸出目錄**:
```
result/deepseek_output/              # 用戶指定的輸出目錄
├── page_001_det.md                  # 原始輸出
├── page_001.md                      # 清理後輸出
├── page_002_det.md
├── page_002.md
└── ...
```

### Key Code Locations

**核心函數位置**（在新文件中）:

```python
# tools/model_infer/deepseek_ocr_inf.py

PROMPT = '...'                                    # L13: Prompt 定義

def clean_formula(text: str) -> str:             # L20-35: 公式清理
    ...

def re_match(text: str) -> tuple:                # L37-48: 特殊標記提取
    ...

def get_deepseek_response(                       # L50-75: API 調用
    image_path: str,
    client: OpenAI,
    model_name: str
) -> str:
    ...

def process_image(args) -> str:                  # L77-115: 單圖處理
    ...

def main():                                       # L117-160: 主函數
    ...
```

**參考代碼位置**:

```python
# tools/model_infer/gpt_4o_inf.py
def get_gpt_response(image_path):                # L39-68: API 調用模板
def process_image(args):                          # L70-80: 單圖處理模板
def main():                                       # L82-111: 主函數模板

# configs/run_dpsk_ocr_eval_batch.py
def clean_formula(text):                          # L53-68: 公式清理
def re_match(text):                               # L70-79: 特殊標記提取
# 雙輸出邏輯                                       # L145-161
```

### Testing Locations

**測試方式**:
- 使用實際數據測試（無單元測試框架）
- 測試數據: `demo_data/omnidocbench_demo/images/`
- 測試輸出: 用戶指定的 `save_root` 目錄

**測試命令**:
```bash
# 快速測試（1 線程）
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root demo_data/omnidocbench_demo/images \
  --save_root test_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 1

# 批量測試（10 線程）
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root demo_data/omnidocbench_demo/images \
  --save_root result/deepseek_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 10
```

### Documentation to Update

**需要更新的文檔**:

1. **README.md** - 添加 deepseek_ocr_inf.py 使用說明:
```markdown
## DeepSeek-OCR 推理

通過 OpenAI Compatible API 調用 vllm DeepSeek-OCR 服務:

```bash
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root ./images \
  --save_root ./output \
  --api_key YOUR_API_KEY \
  --base_url http://vllm-server:8000/v1 \
  --model_name deepseek-ai/DeepSeek-OCR \
  --threads 10
```

2. **tools/model_infer/README.md**（如果存在）- 添加腳本說明

3. **CHANGELOG.md**（如果存在）- 記錄新功能:
```markdown
## [Unreleased]
### Added
- DeepSeek-OCR 推理腳本 (tools/model_infer/deepseek_ocr_inf.py)
  - 支持 OpenAI Compatible API 調用
  - 雙輸出: 原始 + 清理後 Markdown
  - 多線程並行處理
```

**不需要更新**:
- 評估配置文件（configs/*.yaml）- 推理腳本不影響評估
- 開發指南（docs/development-guide.md）- 推理腳本是工具，非核心架構
- 架構文檔（docs/architecture.md）- 推理腳本不改變架構

---

## UX/UI Considerations

**無 UI/UX 影響** - 這是純命令行工具。

**命令行 UX 考慮**:

1. **進度反饋**:
   - 使用 tqdm 顯示處理進度條
   - 顯示當前處理的文件名
   - 顯示剩餘時間估計

2. **錯誤信息**:
   - 清晰的錯誤前綴 `[ERROR]`
   - 具體的錯誤原因
   - 不影響其他文件處理

3. **成功反饋**:
   - 顯示處理統計（總數/成功/失敗）
   - 顯示輸出目錄位置

4. **參數驗證**:
   - 必填參數缺失時顯示清晰提示
   - 提供 `--help` 說明

**示例輸出**:
```
開始使用 10 個線程處理 50 張圖像...
處理進度: 100%|████████████████████| 50/50 [00:30<00:00,  1.67it/s]
處理完成: 總共 50 張圖像, 成功 48 張, 失敗 2 張
結果保存到: result/deepseek_output/
```

---

## Testing Approach

### 測試框架

**無正式測試框架** - 使用手動驗證和實際數據測試

### 測試策略

**1. 功能驗證測試**:

```bash
# 測試 1: API 連通性
curl $VLLM_BASE_URL/v1/models

# 測試 2: 單圖處理
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root test_images \
  --save_root test_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 1

# 驗證輸出
ls test_output/*_det.md  # 原始輸出存在
ls test_output/*.md      # 清理後輸出存在
```

**2. 後處理邏輯測試**:

```bash
# 檢查公式清理
grep "\\quad" test_output/*.md  # 應該沒有結果

# 檢查特殊標記移除
grep "<|ref|>" test_output/*.md  # 應該沒有結果
grep "<|det|>" test_output/*.md  # 應該沒有結果

# 對比原始和清理後的文件
diff test_output/test_det.md test_output/test.md  # 應該有差異
```

**3. 多線程測試**:

```bash
# 測試並發處理
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root demo_data/omnidocbench_demo/images \
  --save_root result/deepseek_parallel_test \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 10

# 驗證所有文件都被處理
ls demo_data/omnidocbench_demo/images/ | wc -l
ls result/deepseek_parallel_test/*.md | wc -l  # 應該是前者的 2 倍
```

**4. 錯誤處理測試**:

```bash
# 測試錯誤的 API endpoint
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root test_images \
  --save_root test_output \
  --api_key $VLLM_API_KEY \
  --base_url http://invalid-url:8000/v1 \
  --threads 1
# 應該顯示錯誤但不崩潰

# 測試不存在的圖像目錄
python tools/model_infer/deepseek_ocr_inf.py \
  --image_root non_existent_dir \
  --save_root test_output \
  --api_key $VLLM_API_KEY \
  --base_url $VLLM_BASE_URL \
  --threads 1
# 應該顯示錯誤但不崩潰
```

### 測試覆蓋範圍

**核心功能覆蓋**:
- ✅ API 調用（正常/異常）
- ✅ 圖像編碼
- ✅ 雙輸出生成
- ✅ 公式清理
- ✅ 特殊標記移除
- ✅ 多線程處理
- ✅ 進度顯示
- ✅ 錯誤處理

**邊界情況覆蓋**:
- ✅ 空圖像目錄
- ✅ 不支持的圖像格式
- ✅ API 超時
- ✅ 網絡錯誤
- ✅ 文件寫入權限問題

### 驗證清單

完成實作後檢查:

- [ ] 文件已創建: `tools/model_infer/deepseek_ocr_inf.py`
- [ ] API 調用成功
- [ ] 雙輸出文件都生成
- [ ] `_det.md` 包含原始輸出
- [ ] `.md` 不包含特殊標記
- [ ] 公式清理正確
- [ ] 多線程正常工作
- [ ] 進度條正常顯示
- [ ] 錯誤優雅處理
- [ ] 統計信息正確顯示
- [ ] 命令行參數正常工作
- [ ] `--help` 顯示正確

---

## Deployment Strategy

### Deployment Steps

**這是新增文件，部署非常簡單**:

1. ✅ **提交代碼**:
```bash
git add tools/model_infer/deepseek_ocr_inf.py
git commit -m "feat(infer): add DeepSeek-OCR inference script with OpenAI API

- Add deepseek_ocr_inf.py for remote vllm API calls
- Support dual output: raw (_det.md) and cleaned (.md)
- Integrate clean_formula() and re_match() post-processing
- Support multi-threaded parallel processing
- Configurable via CLI arguments (api_key, base_url, model_name, threads)

Refs: gpt_4o_inf.py, run_dpsk_ocr_eval_batch.py"
```

2. ✅ **更新 requirements.txt**（如果 openai 不在列表中）:
```bash
# 檢查是否需要添加
pip freeze | grep openai

# 如果沒有，添加到 requirements.txt
echo "openai>=1.0.0" >> requirements.txt
git add requirements.txt
git commit -m "chore: add openai dependency for API inference"
```

3. ✅ **更新文檔**:
```bash
# 更新 README.md 添加使用說明
git add README.md
git commit -m "docs: add DeepSeek-OCR inference script usage"
```

4. ✅ **推送到遠程**:
```bash
git push origin main  # 或您的分支名
```

5. ✅ **通知用戶**:
- 提供使用示例
- 說明 vllm 服務配置要求

### Rollback Plan

**回滾非常簡單（新文件不影響現有功能）**:

```bash
# 方案 1: 刪除文件
git rm tools/model_infer/deepseek_ocr_inf.py
git commit -m "revert: remove deepseek_ocr_inf.py"
git push

# 方案 2: 回退 commit
git revert <commit-hash>
git push

# 方案 3: 直接刪除文件（如果未提交）
rm tools/model_infer/deepseek_ocr_inf.py
```

**影響範圍**: 無 - 這是獨立的新文件，不影響:
- 現有推理腳本
- 評估任務
- 核心架構
- 其他模塊

### Monitoring

**運行時監控**:

1. **進度監控**:
   - tqdm 進度條顯示實時進度
   - 估計剩餘時間

2. **成功率監控**:
   - 顯示成功/失敗統計
   - 識別問題文件

3. **錯誤日誌**:
   - `[ERROR]` 前綴的錯誤信息
   - 具體的失敗原因

4. **性能監控**:
   - 處理速度（images/sec）
   - API 響應時間

**示例監控輸出**:
```
開始使用 10 個線程處理 50 張圖像...
處理進度: 100%|████████████████████| 50/50 [00:30<00:00,  1.67it/s]
[ERROR] 處理失敗 page_010.jpg: Connection timeout
[ERROR] 處理失敗 page_025.jpg: Invalid image format
處理完成: 總共 50 張圖像, 成功 48 張, 失敗 2 張
平均速度: 1.67 images/sec
結果保存到: result/deepseek_output/
```

**日誌建議**（可選增強）:
```python
# 可選: 添加詳細日誌到文件
import logging
logging.basicConfig(
    filename='deepseek_infer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

---

## 📝 總結

這是一個 **Level 0** 專案（單一原子變更），新增一個獨立的推理腳本:

**變更範圍**: 新建 `tools/model_infer/deepseek_ocr_inf.py`

**核心功能**:
- ✅ OpenAI Compatible API 調用 vllm
- ✅ DeepSeek-OCR 專用後處理
- ✅ 雙輸出（原始 + 清理）
- ✅ 多線程並行處理

**技術決策**:
- 參考 `gpt_4o_inf.py` 架構（OpenAI SDK + 多線程）
- 移植 `run_dpsk_ocr_eval_batch.py` 後處理邏輯
- 圖像預處理由 vllm 服務端處理（已配置）

**實作準備**:
- ✅ 上下文已充分收集
- ✅ 參考代碼已識別
- ✅ 技術方案已明確
- ✅ 測試策略已規劃

**下一步**: 開始實作腳本並測試驗證。
