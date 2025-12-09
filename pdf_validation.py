# coding: utf-8
import argparse
import io
import os
import pathlib
import sys

import yaml

from registry.registry import DATASET_REGISTRY, EVAL_TASK_REGISTRY, METRIC_REGISTRY
import dataset
import metrics
import task


def process_args(args):
    parser = argparse.ArgumentParser(
        description="Render latex formulas for comparison."
    )
    parser.add_argument("--config", "-c", type=str, default="./configs/end2end.yaml")
    parameters = parser.parse_args(args)
    return parameters


if __name__ == "__main__":
    parameters = process_args(sys.argv[1:])
    config_path = parameters.config

    if isinstance(config_path, (str, pathlib.Path)):
        with io.open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise TypeError("Unexpected file type")

    if cfg is not None and not isinstance(cfg, (list, dict, str)):
        raise IOError(f"Invalid loaded object type: {type(cfg).__name__}")  # pragma: no cover

    for task_name in cfg.keys():
        if not cfg.get(task_name):
            print(f"No config for task {task_name}")
            continue

        task_config = cfg[task_name]

        # 處理 ocr_comparison_eval 任務（自行處理配置）
        if task_name == "ocr_comparison_eval":
            val_task = EVAL_TASK_REGISTRY.get(task_name)
            val_task(task_config)
            continue

        # 處理傳統任務（end2end_eval, recogition_eval, detection_eval）
        dataset_name = task_config["dataset"]["dataset_name"]
        metrics_list = task_config["metrics"]
        val_dataset = DATASET_REGISTRY.get(dataset_name)(task_config)
        val_task = EVAL_TASK_REGISTRY.get(task_name)

        if task_config["dataset"]["prediction"].get("data_path"):
            save_name = (
                os.path.basename(task_config["dataset"]["prediction"]["data_path"])
                + "_"
                + task_config["dataset"].get("match_method", "quick_match")
            )
        else:
            save_name = os.path.basename(
                task_config["dataset"]["ground_truth"]["data_path"]
            ).split(".")[0]

        print("###### Process: ", save_name)

        if task_config["dataset"]["ground_truth"].get("page_info"):
            val_task(
                val_dataset,
                metrics_list,
                task_config["dataset"]["ground_truth"]["page_info"],
                save_name,
            )
        else:
            val_task(
                val_dataset,
                metrics_list,
                task_config["dataset"]["ground_truth"]["data_path"],
                save_name,
            )
