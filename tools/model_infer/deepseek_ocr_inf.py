import argparse
import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

# DeepSeek-OCR Prompt (簡化版用於 OpenAI Compatible API)
# 注意: 圖像通過 API 的 image_url 傳遞，不需要 <image> 和 <|grounding|> 標記
PROMPT = "Convert the document to markdown."


def clean_formula(text):
    """清理公式中的 \\quad(...) 標記

    Args:
        text: 包含公式的文本

    Returns:
        清理後的文本

    Example:
        Input: \\[ E = mc^2 \\quad(equation 1) \\]
        Output: \\[ E = mc^2 \\]
    """
    formula_pattern = r"\\\[(.*?)\\\]"

    def process_formula(match):
        formula = match.group(1)
        # 移除 \quad(...) 模式
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", formula)
        formula = formula.strip()
        return r"\[" + formula + r"\]"

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text


def re_match(text):
    """提取 <|ref|>...<|det|> 標記

    Args:
        text: 包含特殊標記的文本

    Returns:
        tuple: (matches, matches_other)
            matches: 完整匹配對象列表
            matches_other: 需要移除的標記字符串列表
    """
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_other = []
    for a_match in matches:
        matches_other.append(a_match[0])

    return matches, matches_other


def get_deepseek_response(image_path, client, model_name):
    """調用 vllm DeepSeek-OCR API 獲取響應

    Args:
        image_path: 圖像文件路徑
        client: OpenAI client 實例
        model_name: 模型名稱（如 "deepseek-ai/DeepSeek-OCR"）

    Returns:
        str: API 返回的 Markdown 內容，失敗時返回空字符串
    """
    try:
        # 讀取圖像並編碼為 base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img_str = base64.b64encode(image_bytes).decode()

        # 調用 OpenAI Compatible API
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            temperature=0.0,  # OCR 任務需要設置為 0
        )

        return completion.choices[0].message.content

    except Exception as e:
        print(f"[ERROR] Failed to get response for {os.path.basename(image_path)}: {e}")
        return ""


def process_image(args):
    """處理單張圖像：調用 API、保存原始輸出、應用後處理、保存清理後輸出

    Args:
        args: tuple (image_path, save_root, client, model_name)

    Returns:
        str: 處理狀態信息
    """
    image_path, save_root, client, model_name = args
    file_name = os.path.basename(image_path)
    base_name = file_name.rsplit(".", 1)[0]

    try:
        # 1. 調用 API 獲取響應
        response = get_deepseek_response(image_path, client, model_name)

        if not response:
            return f"處理失敗 {file_name}: 空響應"

        # 2. 保存原始輸出 (_det.md)
        det_path = os.path.join(save_root, f"{base_name}_det.md")
        with open(det_path, "w", encoding="utf-8") as f:
            f.write(response)

        # 3. 應用後處理
        # 3.1 清理公式
        cleaned = clean_formula(response)

        # 3.2 提取並移除特殊標記
        matches_ref, matches_other = re_match(cleaned)
        for match in matches_other:
            cleaned = cleaned.replace(match, "")

        # 3.3 清理多餘換行和 <center> 標籤
        cleaned = cleaned.replace("\n\n\n\n", "\n\n")
        cleaned = cleaned.replace("\n\n\n", "\n\n")
        cleaned = cleaned.replace("<center>", "").replace("</center>", "")

        # 4. 保存清理後的輸出 (.md)
        output_path = os.path.join(save_root, f"{base_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        return f"成功處理: {file_name}"

    except Exception as e:
        return f"處理失敗 {file_name}: {str(e)}"


def main():
    """主函數：解析參數、初始化客戶端、並行處理圖像、顯示統計"""
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek-OCR API 處理圖像並生成 Markdown"
    )
    parser.add_argument("--image_root", type=str, required=True, help="圖像文件夾路徑")
    parser.add_argument(
        "--save_root", type=str, required=True, help="保存結果的文件夾路徑"
    )
    parser.add_argument("--api_key", type=str, required=True, help="API 密鑰")
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="API base URL (例如: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="模型名稱 (默認: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--threads", type=int, default=10, help="並行處理的線程數 (默認: 10)"
    )

    args = parser.parse_args()

    # 創建輸出目錄
    os.makedirs(args.save_root, exist_ok=True)

    # 初始化 OpenAI client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 收集所有需要處理的圖像
    image_files = []
    for file in os.listdir(args.image_root):
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(args.image_root, file)
            image_files.append((image_path, args.save_root, client, args.model_name))

    if not image_files:
        print(f"[WARNING] 在 {args.image_root} 中未找到圖像文件")
        return

    # 使用線程池並行處理圖像
    print(f"開始使用 {args.threads} 個線程處理 {len(image_files)} 張圖像...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(
            tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="處理進度",
            )
        )

    # 打印處理結果統計
    success_count = sum(1 for result in results if "成功" in result)
    fail_count = len(image_files) - success_count

    print(
        f"\n處理完成: 總共 {len(image_files)} 張圖像, 成功 {success_count} 張, 失敗 {fail_count} 張"
    )

    # 如果有失敗，顯示失敗的文件
    if fail_count > 0:
        print("\n失敗的文件:")
        for result in results:
            if "失敗" in result:
                print(f"  - {result}")

    print(f"\n結果保存到: {args.save_root}/")


if __name__ == "__main__":
    main()
