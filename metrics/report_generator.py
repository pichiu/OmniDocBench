"""
OCR 評估報告生成器

此模組提供 Markdown 和 JSON 報告生成功能，用於多模型 OCR 評估結果的比較報告。
報告內容根據配置中的 metrics 設定動態生成。
"""

import json
import os
from datetime import datetime
from typing import Any


class OCRReportGenerator:
    """
    OCR 評估報告生成器

    根據評估結果生成 Markdown 和 JSON 格式的比較報告。
    報告內容由配置中的 metrics 設定動態決定。
    """

    # data_source 分類順序
    CATEGORY_ORDER = [
        "paper",
        "presentation",
        "handwriting",
        "receipt",
        "eBook",
        "finance",
        "form",
    ]

    # data_source 中英文名稱對照
    CATEGORY_NAMES = {
        "paper": "學術論文",
        "presentation": "簡報投影片",
        "handwriting": "手寫文件",
        "receipt": "收據發票",
        "eBook": "電子書",
        "finance": "財務報表",
        "form": "表格表單",
    }

    # 指標說明
    METRIC_DESCRIPTIONS = {
        "Edit_dist": "歸一化編輯距離 (0-1)，越低越好",
        "TEDS": "表格結構編輯距離相似度 (0-1)，越高越好",
        "TEDS_structure_only": "表格結構相似度（僅結構）(0-1)，越高越好",
        "CDM": "公式字元偵測匹配分數 (0-1)，越高越好",
        "BLEU": "BLEU 分數 (0-1)，越高越好",
        "METEOR": "METEOR 分數 (0-1)，越高越好",
    }

    def __init__(
        self,
        results: dict[str, dict[str, Any]],
        metrics_config: dict[str, dict],
    ):
        """
        初始化報告生成器

        Args:
            results: 多模型評估結果，格式為：
                {
                    "ModelName": {
                        "elements": {
                            "text_block": {
                                "overall": {...},
                                "by_data_source": {...}
                            }
                        }
                    }
                }
            metrics_config: 指標配置，格式與 YAML 中的 metrics 區塊相同
        """
        self.results = results
        self.metrics_config = metrics_config
        self.models = list(results.keys())
        self.generated_at = datetime.now()

    def _get_sorted_categories(self, categories: set[str]) -> list[str]:
        """根據預設順序排序分類"""
        sorted_cats = []
        for cat in self.CATEGORY_ORDER:
            if cat in categories:
                sorted_cats.append(cat)
        # 加入不在預設順序中的分類
        for cat in sorted(categories):
            if cat not in sorted_cats:
                sorted_cats.append(cat)
        return sorted_cats

    def _format_metric_value(
        self, metric_name: str, value: float, show_accuracy: bool = True
    ) -> str:
        """格式化指標值"""
        if value is None or value == "NaN":
            return "N/A"

        if metric_name == "Edit_dist" and show_accuracy:
            accuracy = 1.0 - value
            return f"{value:.4f} ({accuracy*100:.2f}%)"
        else:
            return f"{value:.4f}"

    def _get_category_display_name(self, category: str) -> str:
        """取得分類的顯示名稱"""
        zh_name = self.CATEGORY_NAMES.get(category, category)
        return f"{zh_name} ({category})"

    def generate_markdown(self) -> str:
        """
        生成 Markdown 格式報告

        Returns:
            Markdown 格式的報告字串
        """
        lines = []
        lines.append("# OCR 模型評估報告")
        lines.append("")
        lines.append(f"生成時間: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 總覽區塊
        lines.append("## 總覽")
        lines.append("")

        for element_type, element_config in self.metrics_config.items():
            metrics = element_config.get("metric", [])
            if not metrics:
                continue

            lines.append(f"### {element_type}")
            lines.append("")

            # 表頭
            header = "| 模型 | 樣本數 |"
            separator = "|------|--------|"
            for metric in metrics:
                header += f" {metric} |"
                separator += "--------|"

            lines.append(header)
            lines.append(separator)

            # 各模型數據
            for model in self.models:
                model_data = self.results.get(model, {})
                element_data = model_data.get("elements", {}).get(element_type, {})
                overall = element_data.get("overall", {})

                sample_count = overall.get("sample_count", 0)
                row = f"| {model} | {sample_count} |"

                for metric in metrics:
                    metric_value = overall.get(metric)
                    row += f" {self._format_metric_value(metric, metric_value)} |"

                lines.append(row)

            lines.append("")

        # 按 data_source 分類統計
        lines.append("## 按 data_source 分類統計")
        lines.append("")

        for element_type, element_config in self.metrics_config.items():
            metrics = element_config.get("metric", [])
            if not metrics:
                continue

            for metric in metrics:
                lines.append(f"### {element_type} - {metric}")
                lines.append("")

                # 收集所有分類
                all_categories = set()
                for model in self.models:
                    model_data = self.results.get(model, {})
                    element_data = model_data.get("elements", {}).get(element_type, {})
                    by_ds = element_data.get("by_data_source", {})
                    all_categories.update(by_ds.keys())

                if not all_categories:
                    lines.append("_無數據_")
                    lines.append("")
                    continue

                sorted_categories = self._get_sorted_categories(all_categories)

                # 表頭
                header = "| 分類 | 數量 |"
                separator = "|------|------|"
                for model in self.models:
                    header += f" {model} |"
                    separator += "--------|"

                lines.append(header)
                lines.append(separator)

                # 各分類數據
                for category in sorted_categories:
                    cat_display = self._get_category_display_name(category)

                    # 取得數量（從第一個有此分類的模型）
                    count = 0
                    for model in self.models:
                        model_data = self.results.get(model, {})
                        element_data = model_data.get("elements", {}).get(
                            element_type, {}
                        )
                        by_ds = element_data.get("by_data_source", {})
                        if category in by_ds:
                            count = by_ds[category].get("count", 0)
                            break

                    row = f"| {cat_display} | {count} |"

                    for model in self.models:
                        model_data = self.results.get(model, {})
                        element_data = model_data.get("elements", {}).get(
                            element_type, {}
                        )
                        by_ds = element_data.get("by_data_source", {})
                        cat_data = by_ds.get(category, {})
                        metrics_data = cat_data.get("metrics", {})
                        metric_info = metrics_data.get(metric, {})
                        value = metric_info.get("value")

                        row += f" {self._format_metric_value(metric, value)} |"

                    lines.append(row)

                lines.append("")

        # 指標說明
        lines.append("## 指標說明")
        lines.append("")

        used_metrics = set()
        for element_config in self.metrics_config.values():
            used_metrics.update(element_config.get("metric", []))

        for metric in sorted(used_metrics):
            description = self.METRIC_DESCRIPTIONS.get(metric, "")
            if description:
                lines.append(f"- **{metric}**: {description}")

        lines.append("")

        return "\n".join(lines)

    def generate_json(self) -> dict:
        """
        生成 JSON 結構化報告

        Returns:
            結構化的報告字典
        """
        # 提取配置中的指標列表
        config_metrics = {}
        for element_type, element_config in self.metrics_config.items():
            config_metrics[element_type] = element_config.get("metric", [])

        return {
            "generated_at": self.generated_at.isoformat(),
            "config": {"metrics": config_metrics},
            "models": self.results,
        }

    def save_reports(
        self,
        output_dir: str,
        base_name: str = "ocr_comparison_report",
        generate_markdown: bool = True,
        generate_json: bool = True,
    ) -> dict[str, str]:
        """
        儲存報告到檔案

        Args:
            output_dir: 輸出目錄
            base_name: 報告檔案基礎名稱
            generate_markdown: 是否生成 Markdown 報告
            generate_json: 是否生成 JSON 報告

        Returns:
            生成的檔案路徑字典
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        if generate_markdown:
            md_path = os.path.join(output_dir, f"{base_name}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown())
            saved_files["markdown"] = md_path
            print(f"Markdown 報告已儲存至: {md_path}")

        if generate_json:
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.generate_json(), f, indent=2, ensure_ascii=False)
            saved_files["json"] = json_path
            print(f"JSON 報告已儲存至: {json_path}")

        return saved_files
