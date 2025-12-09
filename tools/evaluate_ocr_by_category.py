#!/usr/bin/env python3
"""
OCR 模型分類評估腳本

此腳本用於評估不同 OCR 模型在各文件類型上的表現，
計算 Edit Distance 並輸出分類比較報告。

用法:
    python tools/evaluate_ocr_by_category.py

輸出:
    - result/ocr_evaluation_report.md: 完整評估報告
    - result/ocr_evaluation_details.json: 詳細評估數據
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import Levenshtein


def normalize_text(text: str) -> str:
    """
    正規化文字以進行公平比較

    - 移除多餘空白
    - 統一換行符號
    - 移除 HTML 標籤中的多餘空白
    """
    if not text:
        return ""

    # 統一換行符號
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 移除連續空白（但保留換行）
    text = re.sub(r"[^\S\n]+", " ", text)

    # 移除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 移除連續換行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_gt_text(sample: dict) -> str:
    """
    從 Ground Truth 樣本中提取文字內容

    按照 order 排序，合併所有 text 和 html 內容
    """
    layout_dets = sample.get("layout_dets", [])

    # 按 order 排序
    sorted_dets = sorted(layout_dets, key=lambda x: x.get("order", 0))

    texts = []
    for det in sorted_dets:
        if det.get("ignore", False):
            continue

        category = det.get("category_type", "")

        if category == "table":
            # 表格使用 HTML
            html = det.get("html", "")
            if html:
                texts.append(html)
        elif category in ["text_block", "title", "figure_title", "table_title"]:
            # 文字區塊
            text = det.get("text", "")
            if text:
                texts.append(text)
        elif category in ["display_formula", "inline_formula"]:
            # 公式
            latex = det.get("latex", "")
            if latex:
                texts.append(latex)

    return "\n".join(texts)


def load_prediction(pred_dir: Path, filename: str) -> str:
    """載入預測結果"""
    # 嘗試不同的檔案名稱格式
    base_name = Path(filename).stem

    # 嘗試 .md 檔案
    md_path = pred_dir / f"{base_name}.md"
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


def calculate_edit_distance(pred: str, gt: str) -> dict:
    """
    計算編輯距離和準確率

    返回:
        - edit_distance: 原始編輯距離
        - accuracy: 1 - (edit_distance / max_len)
        - pred_len: 預測長度
        - gt_len: Ground Truth 長度
    """
    pred = normalize_text(pred)
    gt = normalize_text(gt)

    edit_dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))

    if max_len == 0:
        accuracy = 1.0
    else:
        accuracy = 1.0 - (edit_dist / max_len)

    return {
        "edit_distance": edit_dist,
        "accuracy": accuracy,
        "pred_len": len(pred),
        "gt_len": len(gt),
        "max_len": max_len,
    }


def evaluate_model(
    gt_data: list, pred_dir: Path, model_name: str
) -> dict:
    """
    評估單一模型在所有分類上的表現

    返回:
        - by_category: 各分類的評估結果
        - overall: 整體評估結果
        - samples: 各樣本的詳細結果
    """
    results_by_category = defaultdict(list)
    all_results = []
    samples_detail = []

    for sample in gt_data:
        page_info = sample.get("page_info", {})
        page_attr = page_info.get("page_attribute", {})
        category = page_attr.get("data_source", "unknown")
        image_path = page_info.get("image_path", "")
        filename = Path(image_path).name

        # 提取 GT 文字
        gt_text = extract_gt_text(sample)

        # 載入預測結果
        pred_text = load_prediction(pred_dir, filename)

        if not pred_text:
            continue

        # 計算指標
        metrics = calculate_edit_distance(pred_text, gt_text)

        result = {
            "filename": filename,
            "category": category,
            **metrics,
        }

        results_by_category[category].append(result)
        all_results.append(result)
        samples_detail.append(result)

    # 計算各分類統計
    category_stats = {}
    for category, results in results_by_category.items():
        if not results:
            continue

        total_edit = sum(r["edit_distance"] for r in results)
        total_max_len = sum(r["max_len"] for r in results)
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)

        # Whole accuracy: total_edit / total_max_len
        whole_accuracy = 1.0 - (total_edit / total_max_len) if total_max_len > 0 else 0

        category_stats[category] = {
            "count": len(results),
            "avg_accuracy": avg_accuracy,
            "whole_accuracy": whole_accuracy,
            "total_edit_distance": total_edit,
            "total_max_len": total_max_len,
        }

    # 計算整體統計
    if all_results:
        total_edit = sum(r["edit_distance"] for r in all_results)
        total_max_len = sum(r["max_len"] for r in all_results)
        avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
        whole_accuracy = 1.0 - (total_edit / total_max_len) if total_max_len > 0 else 0

        overall_stats = {
            "count": len(all_results),
            "avg_accuracy": avg_accuracy,
            "whole_accuracy": whole_accuracy,
            "total_edit_distance": total_edit,
            "total_max_len": total_max_len,
        }
    else:
        overall_stats = {
            "count": 0,
            "avg_accuracy": 0,
            "whole_accuracy": 0,
            "total_edit_distance": 0,
            "total_max_len": 0,
        }

    return {
        "model_name": model_name,
        "by_category": category_stats,
        "overall": overall_stats,
        "samples": samples_detail,
    }


def generate_markdown_report(results: dict, output_path: Path):
    """
    生成 Markdown 格式的評估報告

    Args:
        results: 包含所有模型評估結果的字典
        output_path: 輸出路徑
    """
    models = list(results.keys())
    all_categories = set()
    for model_result in results.values():
        all_categories.update(model_result["by_category"].keys())

    # 定義分類順序
    category_order = [
        "paper",
        "presentation",
        "handwriting",
        "receipt",
        "eBook",
        "finance",
        "form",
    ]
    all_categories = [c for c in category_order if c in all_categories]

    # 分類中文名稱對照
    category_names = {
        "paper": "學術論文",
        "presentation": "簡報投影片",
        "handwriting": "手寫文件",
        "receipt": "收據發票",
        "eBook": "電子書",
        "finance": "財務報表",
        "form": "表格表單",
    }

    report = []
    report.append("# OCR 模型評估報告")
    report.append("")
    report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 總覽表格
    report.append("## 總覽")
    report.append("")
    report.append("| 模型 | 樣本數 | 平均準確率 | 整體準確率 |")
    report.append("|------|--------|------------|------------|")

    for model in models:
        overall = results[model]["overall"]
        report.append(
            f"| {model} | {overall['count']} | "
            f"{overall['avg_accuracy']:.4f} ({overall['avg_accuracy']*100:.2f}%) | "
            f"{overall['whole_accuracy']:.4f} ({overall['whole_accuracy']*100:.2f}%) |"
        )

    report.append("")

    # 各分類比較表格
    report.append("## 各分類準確率比較")
    report.append("")

    # 表頭
    header = "| 分類 | 數量 |"
    separator = "|------|------|"
    for model in models:
        header += f" {model} |"
        separator += "--------|"

    report.append(header)
    report.append(separator)

    # 各分類數據
    for category in all_categories:
        cat_name = category_names.get(category, category)

        # 取得數量（從第一個有此分類的模型）
        count = 0
        for model in models:
            if category in results[model]["by_category"]:
                count = results[model]["by_category"][category]["count"]
                break

        row = f"| {cat_name} ({category}) | {count} |"

        for model in models:
            if category in results[model]["by_category"]:
                acc = results[model]["by_category"][category]["whole_accuracy"]
                row += f" {acc:.4f} ({acc*100:.2f}%) |"
            else:
                row += " N/A |"

        report.append(row)

    # 整體
    row = "| **整體** | "
    total_count = results[models[0]]["overall"]["count"]
    row += f"**{total_count}** |"

    for model in models:
        acc = results[model]["overall"]["whole_accuracy"]
        row += f" **{acc:.4f} ({acc*100:.2f}%)** |"

    report.append(row)
    report.append("")

    # 詳細統計
    report.append("## 詳細統計")
    report.append("")

    for model in models:
        report.append(f"### {model}")
        report.append("")
        report.append("| 分類 | 數量 | 平均準確率 | 整體準確率 | 總編輯距離 |")
        report.append("|------|------|------------|------------|------------|")

        for category in all_categories:
            if category not in results[model]["by_category"]:
                continue

            stats = results[model]["by_category"][category]
            cat_name = category_names.get(category, category)
            report.append(
                f"| {cat_name} | {stats['count']} | "
                f"{stats['avg_accuracy']:.4f} | "
                f"{stats['whole_accuracy']:.4f} | "
                f"{stats['total_edit_distance']} |"
            )

        overall = results[model]["overall"]
        report.append(
            f"| **整體** | **{overall['count']}** | "
            f"**{overall['avg_accuracy']:.4f}** | "
            f"**{overall['whole_accuracy']:.4f}** | "
            f"**{overall['total_edit_distance']}** |"
        )
        report.append("")

    # 指標說明
    report.append("## 指標說明")
    report.append("")
    report.append("- **平均準確率 (avg_accuracy)**: 每個樣本準確率的算術平均")
    report.append("  - 計算方式: `mean(1 - edit_distance / max(pred_len, gt_len))`")
    report.append("- **整體準確率 (whole_accuracy)**: 所有樣本合併計算的準確率")
    report.append("  - 計算方式: `1 - sum(edit_distance) / sum(max_len)`")
    report.append("- **編輯距離 (Edit Distance)**: Levenshtein 距離，衡量字串相似度")
    report.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"報告已儲存至: {output_path}")


def main():
    # 路徑設定
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "doc_ocr_dataset"
    result_dir = base_dir / "result"
    result_dir.mkdir(exist_ok=True)

    # Ground Truth 路徑
    gt_path = dataset_dir / "omnidocbench_1126_fixed.json"

    # 模型預測結果路徑
    models = {
        "PaddleOCR": dataset_dir / "paddleocr-merged",
        "DeepSeek-OCR": dataset_dir / "deepseek-ocr-merged",
    }

    # 載入 Ground Truth
    print(f"載入 Ground Truth: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"共 {len(gt_data)} 個樣本")

    # 評估各模型
    all_results = {}
    for model_name, pred_dir in models.items():
        print(f"\n評估模型: {model_name}")
        print(f"預測結果目錄: {pred_dir}")

        if not pred_dir.exists():
            print(f"警告: 目錄不存在 {pred_dir}")
            continue

        result = evaluate_model(gt_data, pred_dir, model_name)
        all_results[model_name] = result

        print(f"  - 樣本數: {result['overall']['count']}")
        print(f"  - 整體準確率: {result['overall']['whole_accuracy']:.4f}")

    # 生成報告
    report_path = result_dir / "ocr_evaluation_report.md"
    generate_markdown_report(all_results, report_path)

    # 儲存詳細數據
    details_path = result_dir / "ocr_evaluation_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        # 移除 samples 以減少檔案大小
        export_data = {}
        for model, data in all_results.items():
            export_data[model] = {
                "by_category": data["by_category"],
                "overall": data["overall"],
            }
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"\n詳細數據已儲存至: {details_path}")


if __name__ == "__main__":
    main()
