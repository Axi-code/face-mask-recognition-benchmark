"""
推理结果封装：根据模型输出概率构建单模型/多模型预测 payload、多数投票与一致性统计、已知真值下的分析结果。
"""
from collections import Counter

import torch


def _round_float(value, digits=4):
    """将数值转为浮点数并四舍五入到指定小数位，用于统一 JSON 输出格式。"""
    return round(float(value), digits)


def _build_ranked_items(scores, label_map):
    """
    将类别分数按从高到低排序，构建带 rank、class_name、label、score 的列表，供前端展示 top-k。

    Args:
        scores: {class_name: score} 字典。
        label_map: 类别名到显示标签的映射。

    Returns:
        list: 每项为 {"rank", "class_name", "label", "score"}。
    """
    ranked_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "rank": rank,
            "class_name": class_name,
            "label": label_map.get(class_name, class_name),
            "score": float(score),
        }
        for rank, (class_name, score) in enumerate(ranked_items, start=1)
    ]


def build_prediction_payload(
    probabilities,
    class_names,
    label_map,
    model_name,
    weights_path,
    confidence_threshold,
    roi_info=None,
    device=None,
    input_source="upload",
    extra_meta=None,
    display_name=None,
    offline_metrics=None,
    training_remark=None,
):
    """
    根据模型输出的概率、类别名和配置构建单模型预测结果 payload，供 API/前端使用。

    Args:
        probabilities: 各类别概率的一维 tensor（或可 flatten 的 tensor）。
        class_names: 类别名列表。
        label_map: 类别名到显示标签的映射。
        model_name: 模型名称。
        weights_path: 权重文件路径（字符串）。
        confidence_threshold: 置信度阈值，低于则标记为不确定。
        roi_info: 可选，ROI 提取信息（如 detector_used、bbox 等）。
        device: 可选，运行设备描述。
        input_source: 输入来源标识，如 "upload"、"single_image"。
        extra_meta: 可选，额外元信息（如 inference_time_ms）。
        display_name: 可选，模型显示名称。
        offline_metrics: 可选，离线指标（如 test_accuracy、macro_f1）。
        training_remark: 可选，训练条件说明。

    Returns:
        dict: 含 model、class_names、label_map、prediction、scores、top_k、quality、roi、meta、offline_metrics。
    """
    probabilities = probabilities.detach().cpu().flatten()
    prediction_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[prediction_index].item())
    predicted_class = class_names[prediction_index]
    predicted_label = label_map.get(predicted_class, predicted_class)
    scores = {class_name: float(probabilities[index].item()) for index, class_name in enumerate(class_names)}
    top_k = _build_ranked_items(scores, label_map)
    is_uncertain = confidence < confidence_threshold
    advice = "结果可信度较高。"
    if is_uncertain:
        advice = "结果可信度较低，建议重新拍摄正脸、减少背景干扰后再试。"
    elif roi_info and not roi_info.get("roi_applied"):
        advice = "未检测到清晰人脸，已使用智能裁剪兜底，结果可能受背景影响。"

    meta = {
        "device": str(device) if device is not None else "",
        "input_source": input_source,
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "model": {
            "model_name": model_name,
            "display_name": display_name or model_name,
            "weights_path": weights_path,
            "training_remark": training_remark,
        },
        "class_names": class_names,
        "label_map": label_map,
        "prediction": {
            "class_name": predicted_class,
            "label": predicted_label,
            "confidence": confidence,
        },
        "scores": scores,
        "top_k": top_k,
        "quality": {
            "is_uncertain": is_uncertain,
            "threshold": confidence_threshold,
            "advice": advice,
        },
        "roi": roi_info or {},
        "meta": meta,
        "offline_metrics": offline_metrics or {},
    }


def build_multi_model_payload(predictions, input_source="upload"):
    """
    将多个模型的预测结果汇总为多模型对比 payload（多数投票、平均分数、一致性、图表数据等）。

    Args:
        predictions: build_prediction_payload 返回的 dict 列表，每个元素对应一个模型。
        input_source: 输入来源标识。

    Returns:
        dict: status、class_names、label_map、models、summary（多数类、比例、平均置信度等）、
              consensus（是否一致、票数分布、分歧模型）、chart_data（用于前端图表）。
        若 predictions 为空则返回错误 status。
    """
    if not predictions:
        return {
            "status": "error",
            "error_code": "no_model_available",
            "user_message": "当前没有可用模型，请先训练或加载权重。",
        }

    class_names = predictions[0]["class_names"]
    label_map = predictions[0]["label_map"]
    label_counter = Counter(item["prediction"]["class_name"] for item in predictions)
    majority_class_name, majority_count = label_counter.most_common(1)[0]
    average_class_scores = []
    for class_name in class_names:
        average_score = sum(item["scores"].get(class_name, 0.0) for item in predictions) / len(predictions)
        average_class_scores.append(
            {
                "class_name": class_name,
                "label": label_map.get(class_name, class_name),
                "score": _round_float(average_score),
            }
        )
    average_class_scores.sort(key=lambda item: item["score"], reverse=True)

    total_latency_ms = sum(item["meta"].get("inference_time_ms", 0.0) for item in predictions)
    uncertain_model_count = sum(1 for item in predictions if item["quality"]["is_uncertain"])
    majority_label = label_map.get(majority_class_name, majority_class_name)
    disagreement_models = [
        item["model"]["display_name"]
        for item in predictions
        if item["prediction"]["class_name"] != majority_class_name
    ]

    chart_data = {
        "confidence_comparison": [
            {
                "model_name": item["model"]["model_name"],
                "display_name": item["model"]["display_name"],
                "predicted_class_name": item["prediction"]["class_name"],
                "predicted_label": item["prediction"]["label"],
                "confidence": _round_float(item["prediction"]["confidence"]),
                "is_majority_vote": item["prediction"]["class_name"] == majority_class_name,
            }
            for item in predictions
        ],
        "latency_comparison": [
            {
                "model_name": item["model"]["model_name"],
                "display_name": item["model"]["display_name"],
                "latency_ms": _round_float(item["meta"].get("inference_time_ms", 0.0), digits=2),
            }
            for item in predictions
        ],
        "class_probability_matrix": [
            {
                "model_name": item["model"]["model_name"],
                "display_name": item["model"]["display_name"],
                "scores": [
                    {
                        "class_name": class_name,
                        "label": label_map.get(class_name, class_name),
                        "score": _round_float(item["scores"].get(class_name, 0.0)),
                    }
                    for class_name in class_names
                ],
            }
            for item in predictions
        ],
        "offline_overview": [
            {
                "model_name": item["model"]["model_name"],
                "display_name": item["model"]["display_name"],
                "test_accuracy": item["offline_metrics"].get("test_accuracy"),
                "macro_f1": item["offline_metrics"].get("macro_f1"),
                "avg_inference_latency_ms": item["offline_metrics"].get("avg_inference_latency_ms"),
                "parameter_count": item["offline_metrics"].get("parameter_count"),
            }
            for item in predictions
        ],
    }

    return {
        "status": "ok",
        "class_names": class_names,
        "label_map": label_map,
        "models": predictions,
        "summary": {
            "total_models": len(predictions),
            "input_source": input_source,
            "majority_class_name": majority_class_name,
            "majority_label": majority_label,
            "majority_count": majority_count,
            "majority_ratio": _round_float(majority_count / len(predictions)),
            "avg_top1_confidence": _round_float(
                sum(item["prediction"]["confidence"] for item in predictions) / len(predictions)
            ),
            "avg_latency_ms": _round_float(total_latency_ms / len(predictions), digits=2),
            "uncertain_model_count": uncertain_model_count,
        },
        "consensus": {
            "is_unanimous": len(label_counter) == 1,
            "vote_distribution": [
                {
                    "class_name": class_name,
                    "label": label_map.get(class_name, class_name),
                    "votes": votes,
                    "ratio": _round_float(votes / len(predictions)),
                }
                for class_name, votes in label_counter.most_common()
            ],
            "average_class_scores": average_class_scores,
            "disagreement_models": disagreement_models,
        },
        "chart_data": chart_data,
    }


def analyze_truth_from_predictions(predictions, truth_class_name):
    """
    在已知真实标签的前提下，分析各模型预测是否正确、置信度与真实类分数差距、高置信度错判等。

    Args:
        predictions: build_prediction_payload 返回的 dict 列表（多模型同一张图）。
        truth_class_name: 该样本的真实类别名（必须在 class_names 中）。

    Returns:
        dict: status、truth_class_name、truth_label、summary（正确数、错误数、多数投票是否对等）、
              per_model（每模型详细分析）、chart_data、high_confidence_misses、insights（文字结论）。

    Raises:
        ValueError: predictions 为空或 truth_class_name 不在已知类别中时抛出。
    """
    if not predictions:
        raise ValueError("predictions is required.")

    class_names = predictions[0]["class_names"]
    label_map = predictions[0]["label_map"]
    if truth_class_name not in class_names:
        raise ValueError("truth_class_name is not in known classes.")

    vote_counter = Counter(item["prediction"]["class_name"] for item in predictions)
    majority_class_name, _ = vote_counter.most_common(1)[0]
    truth_label = label_map.get(truth_class_name, truth_class_name)
    per_model = []
    correct_confidences = []
    wrong_confidences = []
    high_confidence_misses = []

    for item in predictions:
        predicted_class_name = item["prediction"]["class_name"]
        confidence = float(item["prediction"]["confidence"])
        truth_score = float(item["scores"].get(truth_class_name, 0.0))
        is_correct = predicted_class_name == truth_class_name
        miss_threshold = max(0.8, float(item["quality"].get("threshold", 0.65)))
        model_analysis = {
            "model_name": item["model"]["model_name"],
            "display_name": item["model"]["display_name"],
            "training_remark": item["model"].get("training_remark"),
            "predicted_class_name": predicted_class_name,
            "predicted_label": item["prediction"]["label"],
            "confidence": _round_float(confidence),
            "truth_class_name": truth_class_name,
            "truth_label": truth_label,
            "truth_score": _round_float(truth_score),
            "confidence_gap": _round_float(confidence - truth_score),
            "is_correct": is_correct,
            "is_high_confidence_miss": bool((not is_correct) and confidence >= miss_threshold),
        }
        per_model.append(model_analysis)
        if is_correct:
            correct_confidences.append(confidence)
        else:
            wrong_confidences.append(confidence)
            if model_analysis["is_high_confidence_miss"]:
                high_confidence_misses.append(model_analysis)

    correct_count = sum(1 for item in per_model if item["is_correct"])
    wrong_count = len(per_model) - correct_count
    highest_confidence = max(per_model, key=lambda item: item["confidence"])
    highest_truth_score = max(per_model, key=lambda item: item["truth_score"])
    insights = []

    if correct_count == len(per_model):
        insights.append("所有模型在这张图上都预测正确，样本区分度较高。")
    elif wrong_count == len(per_model):
        insights.append("所有模型都预测错误，这张图可能存在明显域偏移或样本噪声。")
    else:
        insights.append(f"共有 {correct_count} 个模型预测正确，{wrong_count} 个模型预测错误。")

    if high_confidence_misses:
        insights.append("存在高置信度错判模型，说明部分模型有过度自信现象。")

    if majority_class_name == truth_class_name:
        insights.append("多数投票与真实标签一致，集成判断在该样本上更稳。")
    else:
        insights.append("多数投票与真实标签不一致，模型集体偏向了错误方向。")

    return {
        "status": "ok",
        "truth_class_name": truth_class_name,
        "truth_label": truth_label,
        "summary": {
            "total_models": len(per_model),
            "correct_models": correct_count,
            "wrong_models": wrong_count,
            "correct_ratio": _round_float(correct_count / len(per_model)),
            "majority_vote_class_name": majority_class_name,
            "majority_vote_label": label_map.get(majority_class_name, majority_class_name),
            "majority_vote_correct": majority_class_name == truth_class_name,
            "avg_confidence_correct": _round_float(sum(correct_confidences) / len(correct_confidences))
            if correct_confidences
            else 0.0,
            "avg_confidence_wrong": _round_float(sum(wrong_confidences) / len(wrong_confidences))
            if wrong_confidences
            else 0.0,
            "highest_confidence_model": highest_confidence,
            "highest_truth_score_model": highest_truth_score,
            "high_confidence_miss_count": len(high_confidence_misses),
        },
        "per_model": per_model,
        "chart_data": {
            "correctness_comparison": [
                {
                    "display_name": item["display_name"],
                    "confidence": item["confidence"],
                    "is_correct": item["is_correct"],
                }
                for item in per_model
            ],
            "truth_score_comparison": [
                {
                    "display_name": item["display_name"],
                    "truth_score": item["truth_score"],
                }
                for item in per_model
            ],
            "confidence_gap": [
                {
                    "display_name": item["display_name"],
                    "confidence_gap": item["confidence_gap"],
                }
                for item in per_model
            ],
        },
        "high_confidence_misses": high_confidence_misses,
        "insights": insights,
    }
