"""Streamlit demo for the Chinese gender-bias detection course project.

Run with:  streamlit run src/demo/app.py

Only this module imports streamlit; `registry`/`scoring` stay import-free so the
contract test suite runs on a bare CI image. The page talks to models exclusively
through `load_predictor` + `predict_proba`; it never imports torch or a model's
internals, and it never recomputes an evaluation metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.demo import registry, scoring  # noqa: E402  (after sys.path bootstrap)

SCORE_WORDING = "模型分数（未做概率校准，不等同于客观偏见概率）"
MOCK_WORDING = "开发替身：固定输出 0.5，没有任何检测能力。此处结果不能作为模型表现或实验结论。"


def _model_caption(predictor) -> tuple[str, str]:
    """Return (model_name, revision) recorded with the loaded model, if any."""
    if bool(getattr(predictor, "is_mock", False)):
        return "开发替身（固定 0.5）", "mock"
    metadata = getattr(predictor, "metadata", None) or {}
    name = str(getattr(predictor, "model_name", None) or metadata.get("model_name") or "unknown")
    revision = str(metadata.get("model_revision") or metadata.get("model_version") or "未记录")
    return name, revision


def _run(config: dict, key: str, texts: list[str], threshold: float):
    predictor = registry.load_model(config, key)
    rows = scoring.predict_rows(predictor, texts, threshold)
    return predictor, rows


def main() -> None:
    st.set_page_config(page_title="中文文本性别偏见检测演示", page_icon="🔍", layout="centered")
    config = registry.load_config()
    examples_doc = registry.load_examples()
    models = registry.describe_models(config)

    st.title("中文文本性别偏见检测演示")
    st.caption("课程设计第 8 题 · 数据：CORGI-PM · 二分类：0 = 无偏见，1 = 有偏见")

    if config.get("mock_banner_required", True):
        st.info(
            "本页当前处于开发阶段：只有「开发替身」可直接运行；P2/P3 真实模型需要先生成 "
            "`artifacts/` 目录。mock 结果带显著标识，不可用于任何结论。"
        )

    keys = [row["key"] for row in models]
    labels = {}
    for row in models:
        suffix = "" if row["available"] else "（模型目录未就绪）"
        labels[row["key"]] = row["display_name"] + suffix
    default_index = keys.index(config["default_model"]) if config["default_model"] in keys else 0

    chosen = st.selectbox("模型", keys, index=default_index, format_func=lambda key: labels[key], key="model_key")
    entry = registry.find_entry(config, chosen)

    if entry.get("adapter_module") is None:
        st.caption("提示：开发替身只用于验证接口，不产生真实判断；结果区会有显著标识。")
    elif not registry.is_available(entry):
        st.warning(
            f"该模型目录不存在：`{registry.resolve_model_dir(entry)}`。"
            "点「运行检测」后会给出具体原因；这不影响 mock 分支的演示。"
        )

    example_options = ["（不使用，自行输入）"] + [
        f"{item['id']}｜{item['text']}" for item in examples_doc["examples"]
    ]
    picked = st.selectbox("内置演示例句（自编示例，不是官方金标）", example_options, index=0)
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
        else:
            try:
                predictor, rows = _run(config, chosen, texts, threshold)
            except registry.ModelUnavailable as exc:
                st.error(str(exc))
                if exc.hint:
                    st.caption(f"建议：{exc.hint}")
            except ValueError as exc:
                st.error(f"输入不合法：{exc}")
            else:
                is_mock = bool(getattr(predictor, "is_mock", False))
                if is_mock:
                    st.error(MOCK_WORDING)
                model_name, revision = _model_caption(predictor)
                recorded = getattr(predictor, "threshold", None)
                st.caption(
                    f"模型：{model_name}（revision: {revision}）｜使用阈值 {threshold:.2f}"
                    + (f"｜模型记录阈值 {float(recorded):.2f}" if recorded is not None else "")
                )
                for row in rows:
                    label = "有偏见（1）" if row["label"] == 1 else "无偏见（0）"
                    st.markdown(f"**{row['text']}**")
                    st.progress(min(max(row["score"], 0.0), 1.0))
                    st.markdown(f"{label}｜{row['score']:.3f} 分")
                    st.caption(SCORE_WORDING)
                    st.divider()

    with st.expander("模型与参数说明"):
        st.json({"models": models, "default_model": config["default_model"], "mock_banner_required": config["mock_banner_required"]})
        st.caption(
            "分数来自适配器的 predict_proba，定义为“有偏见（1）”的模型分数；"
            "未做概率校准，不能当作事实概率。页面不判断作者性别，也不输出性别对象标签。"
        )


if __name__ == "__main__":
    main()
