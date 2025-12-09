"""
OCR 多模型比較評估任務

此模組提供多模型 OCR 評估比較功能，使用現有的 End2EndDataset 和 Metric 架構，
並整合報告生成功能。
"""

import json
import os
from typing import Any

from registry.registry import DATASET_REGISTRY, EVAL_TASK_REGISTRY, METRIC_REGISTRY
from metrics.report_generator import OCRReportGenerator
from metrics.show_result import get_data_source_summary


@EVAL_TASK_REGISTRY.register("ocr_comparison_eval")
class OCRComparisonEval:
    """
    OCR 多模型比較評估任務

    支援同時評估多個 OCR 模型，並生成比較報告。
    使用現有的 End2EndDataset 進行資料載入和匹配，
    使用已註冊的 Metric 進行評估計算。
    """

    def __init__(self, config: dict):
        """
        初始化並執行評估任務

        Args:
            config: 配置字典，包含 models、ground_truth、metrics、output 等設定
        """
        self.config = config
        self.models = config.get("models", [])
        self.gt_path = config["ground_truth"]["data_path"]
        self.match_method = config.get("match_method", "quick_match")
        self.metrics_config = config.get("metrics", {})
        self.output_config = config.get("output", {})

        # 載入 page_info 用於 data_source 分類
        self.page_info = self._load_page_info()

        # 執行評估
        self.results = self.run()

    def _load_page_info(self) -> dict[str, dict]:
        """載入頁面資訊，用於 data_source 分類"""
        page_info = {}
        with open(self.gt_path, "r", encoding="utf-8") as f:
            self._gt_pages = json.load(f)  # 保存以便後續計算 GT 統計

        for page in self._gt_pages:
            img_path = os.path.basename(page["page_info"]["image_path"])
            # 去除副檔名
            img_name = img_path[:-4] if img_path.endswith((".jpg", ".png")) else img_path
            page_info[img_name] = page["page_info"].get("page_attribute", {})

        return page_info

    def _get_gt_statistics(self) -> dict:
        """
        計算 Ground Truth 中各元素類型的總數

        Returns:
            包含總頁數、GT 來源路徑和各元素類型數量的字典
        """
        gt_stats = {
            "total_pages": len(self._gt_pages),
            "gt_source": os.path.basename(self.gt_path),
            "elements": {},
        }

        # GT 類別到報告元素類型的映射
        type_mapping = {
            "text_block": "text_block",
            "equation_isolated": "display_formula",
            "table": "table",
        }

        for element_type in self.metrics_config.keys():
            if element_type == "reading_order":
                # reading_order 每頁一個
                gt_stats["elements"][element_type] = len(self._gt_pages)
            else:
                count = 0
                for page in self._gt_pages:
                    for det in page.get("layout_dets", []):
                        gt_type = type_mapping.get(det.get("category_type"))
                        if gt_type == element_type:
                            count += 1
                gt_stats["elements"][element_type] = count

        return gt_stats

    def _collect_errors(
        self, samples_dict: dict
    ) -> dict[str, dict[str, list]]:
        """
        收集所有錯誤項目（Edit_dist > 0），按 data_source 分組

        Args:
            samples_dict: 各元素類型的樣本字典

        Returns:
            按元素類型和 data_source 組織的錯誤列表
        """
        errors = {}
        for element_type, samples in samples_dict.items():
            sample_list = samples.samples if hasattr(samples, "samples") else samples
            if not sample_list:
                continue

            errors[element_type] = {}  # key: data_source

            for sample in sample_list:
                edit_dist = sample.get("metric", {}).get("Edit_dist", 0)

                # 只收集有錯誤的項目
                if edit_dist > 0:
                    data_source = self.page_info.get(
                        sample.get("img_id"), {}
                    ).get("data_source", "unknown")

                    if data_source not in errors[element_type]:
                        errors[element_type][data_source] = []

                    item = {
                        "img_id": sample.get("img_id"),
                        "edit_dist": edit_dist,
                        "gt_text": sample.get("gt", ""),
                        "pred_text": sample.get("pred", ""),
                    }

                    # 標記錯誤類型：未匹配（模型完全沒輸出）或錯誤匹配
                    if sample.get("pred_idx") == [""] or sample.get("pred", "") == "":
                        item["error_type"] = "unmatched"
                    else:
                        item["error_type"] = "mismatched"

                    # 加入 GT 屬性（如有）
                    gt_attr = sample.get("gt_attribute")
                    if gt_attr:
                        if isinstance(gt_attr, list) and gt_attr:
                            item["gt_attribute"] = gt_attr[0]
                        else:
                            item["gt_attribute"] = gt_attr

                    errors[element_type][data_source].append(item)

        return errors

    def _build_single_model_config(self, model: dict) -> dict:
        """
        為單一模型構建 end2end_eval 配置

        Args:
            model: 模型配置，包含 name 和 prediction_path

        Returns:
            End2EndDataset 所需的配置字典
        """
        return {
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": self.gt_path},
                "prediction": {"data_path": model["prediction_path"]},
                "match_method": self.match_method,
            },
            "metrics": self.metrics_config,
        }

    def _evaluate_single_model(self, model: dict) -> tuple[dict[str, Any], dict]:
        """
        評估單一模型

        Args:
            model: 模型配置

        Returns:
            tuple: (模型評估結果, 評估後的樣本字典)
        """
        model_name = model["name"]
        print(f"\n評估模型: {model_name}")
        print(f"預測結果目錄: {model['prediction_path']}")

        # 構建配置並載入資料集
        single_config = self._build_single_model_config(model)
        dataset = DATASET_REGISTRY.get("end2end_dataset")(single_config)

        model_result = {"elements": {}}
        evaluated_samples_dict = {}  # 保存評估後的樣本

        # 對每個元素類型執行評估
        for element_type, element_config in self.metrics_config.items():
            metrics_list = element_config.get("metric", [])
            if not metrics_list:
                continue

            # 取得該元素類型的樣本
            samples = dataset.samples.get(element_type)
            if samples is None:
                print(f"  警告: 無法取得 {element_type} 樣本")
                continue

            # 取得樣本列表
            if hasattr(samples, "samples"):
                sample_list = samples.samples
            else:
                sample_list = samples

            if not sample_list:
                print(f"  警告: {element_type} 樣本為空")
                continue

            print(f"  評估 {element_type}: {len(sample_list)} 個樣本")

            element_result = {"overall": {"sample_count": len(sample_list)}, "by_data_source": {}}

            # 執行每個指標的評估
            for metric_name in metrics_list:
                try:
                    metric_class = METRIC_REGISTRY.get(metric_name)
                    save_name = f"{model_name}_{element_type}"
                    evaluated_samples, metric_result = metric_class(sample_list).evaluate(
                        group_info=[], save_name=save_name
                    )

                    # 更新樣本列表（某些指標會修改樣本）
                    if hasattr(evaluated_samples, "samples"):
                        sample_list = evaluated_samples.samples
                    else:
                        sample_list = evaluated_samples

                    # 提取整體指標值
                    if metric_result and metric_name in metric_result:
                        metric_data = metric_result[metric_name]
                        # 優先使用 edit_whole（整體準確率），否則使用 all 或第一個值
                        if isinstance(metric_data, dict):
                            if "edit_whole" in metric_data:
                                element_result["overall"][metric_name] = metric_data["edit_whole"]
                            elif "all" in metric_data:
                                element_result["overall"][metric_name] = metric_data["all"]
                            elif "ALL_page_avg" in metric_data:
                                element_result["overall"][metric_name] = metric_data["ALL_page_avg"]
                            else:
                                # 取第一個非 NaN 值
                                for v in metric_data.values():
                                    if v != "NaN":
                                        element_result["overall"][metric_name] = v
                                        break
                        else:
                            element_result["overall"][metric_name] = metric_data

                except Exception as e:
                    print(f"    警告: {metric_name} 評估失敗 - {e}")
                    continue

            # 計算按 data_source 分類的統計
            by_data_source = get_data_source_summary(sample_list, self.page_info, element_config)
            element_result["by_data_source"] = by_data_source

            model_result["elements"][element_type] = element_result

            # 保存評估後的樣本（包含 metric 資訊）
            evaluated_samples_dict[element_type] = sample_list

            # 輸出整體指標
            overall_metrics = {
                k: v for k, v in element_result["overall"].items() if k != "sample_count"
            }
            print(f"    整體指標: {overall_metrics}")

        return model_result, evaluated_samples_dict

    def run(self) -> dict[str, Any]:
        """
        執行多模型評估

        Returns:
            所有模型的評估結果
        """
        all_results = {}
        all_errors = {}  # 收集各模型的錯誤資訊

        for model in self.models:
            model_name = model["name"]
            prediction_path = model.get("prediction_path", "")

            if not os.path.exists(prediction_path):
                print(f"警告: 模型 {model_name} 的預測目錄不存在: {prediction_path}")
                continue

            result, samples_dict = self._evaluate_single_model(model)
            all_results[model_name] = result

            # 收集該模型的錯誤資訊
            all_errors[model_name] = self._collect_errors(samples_dict)

        # 生成報告
        if all_results and self.output_config:
            gt_statistics = self._get_gt_statistics()
            self._generate_reports(all_results, gt_statistics, all_errors)

        return all_results

    def _generate_reports(
        self,
        results: dict[str, Any],
        gt_statistics: dict,
        errors_data: dict[str, dict],
    ):
        """
        生成評估報告

        Args:
            results: 各模型的評估結果
            gt_statistics: Ground Truth 統計資訊
            errors_data: 各模型的錯誤資訊
        """
        output_dir = self.output_config.get("output_dir", "./result")
        report_name = self.output_config.get("report_name", "ocr_comparison_report")
        generate_markdown = self.output_config.get("generate_markdown", True)
        generate_json = self.output_config.get("generate_json", True)
        generate_errors = self.output_config.get("generate_errors", True)

        generator = OCRReportGenerator(
            results, self.metrics_config, gt_statistics, errors_data
        )
        generator.save_reports(
            output_dir=output_dir,
            base_name=report_name,
            generate_markdown=generate_markdown,
            generate_json=generate_json,
            generate_errors=generate_errors,
        )
