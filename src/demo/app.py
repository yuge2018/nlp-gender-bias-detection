"""Streamlit demo for the Chinese gender-bias detection course project.

Run with:  streamlit run src/demo/app.py

Only this module imports streamlit; `registry`/`scoring`/`results` stay import-free
so the contract test suite runs on a bare CI image. The page talks to models
exclusively through `load_predictor` + `predict_proba`, and shows real metrics by
reading P4's run files — it never recomputes an evaluation metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.demo import registry, results, scoring  # noqa: E402  (after sys.path bootstrap)

SCORE_WORDING = "模型分数（未做概率校准，不等同于客观偏见概率）"
MOCK_WORDING = "开发替身：固定输出 0.5，没有任何检测能力。此处结果不能作为模型表现或实验结论。"
STATUS_SUFFIX = {
    "ready": "",
    "mock": "",
    "weights_missing": "（权重未就绪）",
    "missing": "（模型目录未就绪）",
}


def _model_caption(predictor) -> tuple[str, str]:
    """Return (model_name, revision) recorded with the loaded model, if any."""
    if bool(getattr(predictor, "is_mock", False)):
        return "开发替身（固定 0.5）", "mock"
    metadata = getattr(predictor, "metadata", None) or {}
    name = str(getattr(predictor, "model_name", None) or metadata.get("model_name") or "unknown")
    revision = str(metadata.get("model_revision") or metadata.get("model_version") or "未记录")
    return name, revision


def _render_eval_tab(config: dict, examples_doc: dict, models: list[dict]) -> None:
    keys = [row["key"] for row in models]
    labels = {row["key"]: row["display_name"] + STATUS_SUFFIX.get(row["status"], "") for row in models}
    default_index = keys.index(config["default_model"]) if config["default_model"] in keys else 0

    chosen = st.selectbox("模型", keys, index=default_index, format_func=lambda key: labels[key], key="model_key")
    row = next(item for item in models if item["key"] == chosen)

    if row["status"] == "mock":
        st.caption("提示：开发替身只用于验证接口，不产生真实判断；结果区会有显著标识。")
    elif row["status"] == "weights_missing":
        st.warning(
            f"该目录只有 `model_metadata.json`，权重文件未随仓库提交：`{row['model_dir']}`。"
            "仓库 `.gitignore` 排除了权重，需在本机按对应角色 README 重新生成后再选它。"
        )
    elif row["status"] == "missing":
        st.warning(f"该模型目录不存在：`{row['model_dir']}`。点「运行检测」后会给出具体原因。")

    example_options = ["（不使用，自行输入）"] + [
        f"{item['id']}｜{item['text']}" for item in examples_doc["examples"]
    ]
    picked = st.selectbox("内置演示例句（自编示例，不是官方金标）", example_options, index=0, key="example_pick")
    default_text = ""
    if picked != example_options[0]:
        default_text = picked.split("｜", 1)[1]

    with st.form("demo_form"):
        text = st.text_area("待分析文本（每行一条）", value=default_text, height=130)
        threshold = st.slider(
            "判定阈值（label=1 当 分数 ≥ 阈值）",
            min_value=0.0,
            max_value=1.0,
            value=float(examples_doc.get("threshold_default", 0.5)),
            step=0.05,
        )
        submitted = st.form_submit_button("运行检测")

    if submitted:
        texts = [line for line in text.splitlines() if line.strip()]
        if not texts:
            st.error("请输入至少一条非空文本。")
            return
        try:
            predictor = registry.load_model(config, chosen)
            rows = scoring.predict_rows(predictor, texts, threshold)
        except registry.ModelUnavailable as exc:
            st.error(str(exc))
            if exc.hint:
                st.caption(f"建议：{exc.hint}")
            return
        except ValueError as exc:
            st.error(f"输入不合法：{exc}")
            return

        is_mock = bool(getattr(predictor, "is_mock", False))
        if is_mock:
            st.error(MOCK_WORDING)
        model_name, revision = _model_caption(predictor)
        recorded = getattr(predictor, "threshold", None)
        st.caption(
            f"模型：{model_name}（revision: {revision}）｜使用阈值 {threshold:.2f}"
            + (f"｜模型记录阈值 {float(recorded):.2f}" if recorded is not None else "")
        )
        for result_row in rows:
            label = "有偏见（1）" if result_row["label"] == 1 else "无偏见（0）"
            st.markdown(f"**{result_row['text']}**")
            st.progress(min(max(result_row["score"], 0.0), 1.0))
            st.markdown(f"{label}｜{result_row['score']:.3f} 分")
            st.caption(SCORE_WORDING)
            st.divider()

    with st.expander("模型与参数说明"):
        st.json(
            {
                "models": models,
                "default_model": config["default_model"],
                "mock_banner_required": config["mock_banner_required"],
            }
        )
        st.caption(
            "分数来自适配器的 predict_proba，定义为“有偏见（1）”的模型分数；"
            "未做概率校准，不能当作事实概率。页面不判断作者性别，也不输出性别对象标签。"
        )


def _render_results_tab() -> None:
    data = results.discover_runs()
    runs = data["runs"]

    st.caption(
        f"数据直接读 `runs/**/metrics.json` 与 `run_metadata.json`（根目录 `{data['root']}`）；"
        "P6 不重算任何指标，正式主表与聚合仍以 P4 的 `comparison.csv` 为准。"
    )
    if data["skipped_mock"]:
        st.caption(f"已排除 {data['skipped_mock']} 个 `is_mock=true` 的开发 run，避免替身结果混入。")

    if not runs:
        st.info("仓库内暂无真实 run（`is_mock=true` 已被排除）。P2/P3 产出真实 run 后这里会自动出现。")
    else:
        st.dataframe(results.as_display_rows(runs), hide_index=True, width="stretch")
        picked = st.selectbox("查看单次 run 详情", [row["run"] for row in runs], key="run_pick")
        detail = results.selected_run(data, picked)
        if detail:
            st.json(
                {
                    "run": detail["run"],
                    "method": detail["method"],
                    "split": detail["split"],
                    "seed": detail["seed"],
                    "threshold": detail["threshold"],
                    "model_name": detail["model_name"],
                    "n": detail["n"],
                    "confusion_matrix": detail["confusion_matrix"],
                    "macro_f1": detail["macro_f1"],
                    "data_hash": detail["data_hash"],
                }
            )
            st.caption("混淆矩阵行是真值 0/1、列是预测 0/1；数值与文件一致，未做四舍五入存储。")

    if data["predictions_only"]:
        st.warning(
            "以下 run 有 `predictions.jsonl` 但没有 `metrics.json`，说明尚未经 P4 评测：\n\n"
            + "\n".join(f"- `{path}`" for path in data["predictions_only"])
        )
    if data["invalid"]:
        st.error("以下 `metrics.json` 未通过契约校验，已被跳过：")
        for item in data["invalid"]:
            st.caption(f"- `{item['run']}`：{item['error']}")


def main() -> None:
    st.set_page_config(page_title="中文文本性别偏见检测演示", page_icon="🔍", layout="wide")
    config = registry.load_config()
    examples_doc = registry.load_examples()
    models = registry.describe_models(config)

    st.title("中文文本性别偏见检测演示")
    st.caption("课程设计第 8 题 · 数据：CORGI-PM · 二分类：0 = 无偏见，1 = 有偏见")

    if config.get("mock_banner_required", True):
        st.info(
            "模型状态：P2 baseline 权重（`model.joblib`）已入库，选中「P2 基线」即可现场打分；"
            "improved（RoBERTa）目录目前只有元信息，需在本机生成 `artifacts/improved-i2-s42` 后才能加载。"
            "「真实结果」页签展示的是已提交的真实 run 指标，与模型加载状态无关。"
        )

    eval_tab, results_tab = st.tabs(["单句检测", "真实结果（只读）"])
    with eval_tab:
        _render_eval_tab(config, examples_doc, models)
    with results_tab:
        _render_results_tab()


if __name__ == "__main__":
    main()
