const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const emptyPreview = document.getElementById("emptyPreview");
const startCameraButton = document.getElementById("startCamera");
const captureFrameButton = document.getElementById("captureFrame");
const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const cameraTip = document.getElementById("cameraTip");
const inputModeButtons = document.querySelectorAll(".segmented-item");
const uploadPanel = document.getElementById("uploadPanel");
const cameraPanel = document.getElementById("cameraPanel");
const resultChip = document.getElementById("resultChip");
const resultState = document.getElementById("resultState");
const dashboardChip = document.getElementById("dashboardChip");
const dashboardState = document.getElementById("dashboardState");
const dashboardContent = document.getElementById("dashboardContent");
const summaryGrid = document.getElementById("summaryGrid");
const consensusBox = document.getElementById("consensusBox");
const modelResultsGrid = document.getElementById("modelResultsGrid");
const confidenceChart = document.getElementById("confidenceChart");
const probabilityMatrix = document.getElementById("probabilityMatrix");
const voteChart = document.getElementById("voteChart");
const latencyChart = document.getElementById("latencyChart");
const truthForm = document.getElementById("truthForm");
const truthSelect = document.getElementById("truthSelect");
const truthSubmit = document.getElementById("truthSubmit");
const truthChip = document.getElementById("truthChip");
const truthState = document.getElementById("truthState");
const truthContent = document.getElementById("truthContent");
const truthSummaryGrid = document.getElementById("truthSummaryGrid");
const truthInsights = document.getElementById("truthInsights");
const correctnessChart = document.getElementById("correctnessChart");
const truthScoreChart = document.getElementById("truthScoreChart");
const truthTable = document.getElementById("truthTable");
const uploadSubmit = document.getElementById("uploadSubmit");

const modelCatalog = JSON.parse(document.getElementById("initialModelCatalog").textContent);
const classNames = JSON.parse(document.getElementById("initialClassNames").textContent);

let cameraStream = null;
let lastPredictionPayload = null;

function setInputMode(mode) {
    inputModeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
    uploadPanel.classList.toggle("active", mode === "upload");
    cameraPanel.classList.toggle("active", mode === "camera");
}

function showPreview(source) {
    previewImage.src = source;
    previewImage.classList.remove("hidden");
    emptyPreview.classList.add("hidden");
}

function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatMs(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${Number(value).toFixed(2)} ms`;
}

function formatCount(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    if (Number(value) >= 1e6) {
        return `${(Number(value) / 1e6).toFixed(2)} M`;
    }
    if (Number(value) >= 1e3) {
        return `${(Number(value) / 1e3).toFixed(2)} K`;
    }
    return `${value}`;
}

function createSummaryCard(title, value, hint = "") {
    return `
        <article class="summary-card">
            <span class="summary-title">${title}</span>
            <strong class="summary-value">${value}</strong>
            <span class="summary-hint">${hint}</span>
        </article>
    `;
}

function createBarRow(label, valueText, ratio, tone = "primary") {
    const numericRatio = Math.max(Number(ratio) || 0, 0);
    const safeRatio = numericRatio === 0 ? 0 : Math.max(Math.min(numericRatio * 100, 100), 4);
    return `
        <div class="chart-row">
            <div class="chart-row-head">
                <span>${label}</span>
                <strong>${valueText}</strong>
            </div>
            <div class="score-track">
                <div class="score-bar ${tone}" style="width: ${safeRatio}%;"></div>
            </div>
        </div>
    `;
}

function resetTruthAnalysis(message = "先完成一次在线评测，再填写真实标签进行复盘分析。") {
    truthChip.textContent = "等待真值";
    truthChip.className = "result-chip";
    truthState.textContent = message;
    truthContent.classList.add("hidden");
    truthSummaryGrid.innerHTML = "";
    truthInsights.innerHTML = "";
    correctnessChart.innerHTML = "";
    truthScoreChart.innerHTML = "";
    truthTable.innerHTML = "";
}

function setLoading(message) {
    resultChip.textContent = "评测中";
    resultChip.className = "result-chip pending";
    dashboardChip.textContent = "评测中";
    dashboardChip.className = "result-chip pending";
    resultState.textContent = message;
    dashboardState.textContent = message;
    dashboardContent.classList.add("hidden");
    lastPredictionPayload = null;
    resetTruthAnalysis();
}

function renderError(payload) {
    const message = payload.user_message || payload.error || "评测失败，请稍后重试。";
    resultChip.textContent = "评测失败";
    resultChip.className = "result-chip error";
    dashboardChip.textContent = "评测失败";
    dashboardChip.className = "result-chip error";
    resultState.textContent = message;
    dashboardState.textContent = payload.error_message || message;
    dashboardContent.classList.add("hidden");
    resetTruthAnalysis("在线评测失败，暂时无法进行真值分析。");
}

function renderSummary(payload) {
    summaryGrid.innerHTML = [
        createSummaryCard("多数投票结果", payload.summary.majority_label, `${payload.summary.majority_count}/${payload.summary.total_models} 个模型支持`),
        createSummaryCard("平均 Top-1 置信度", formatPercent(payload.summary.avg_top1_confidence), "反映整体自信程度"),
        createSummaryCard("平均在线耗时", formatMs(payload.summary.avg_latency_ms), "单模型平均单次推理耗时"),
        createSummaryCard("低置信度模型数", `${payload.summary.uncertain_model_count}`, "数量越少说明整体更稳定"),
    ].join("");

    const consensusText = payload.consensus.is_unanimous
        ? `全部模型一致预测为“${payload.summary.majority_label}”，该样本上模型意见高度统一。`
        : `当前多数投票为“${payload.summary.majority_label}”，但仍有 ${payload.consensus.disagreement_models.length} 个模型持不同意见：${payload.consensus.disagreement_models.join("、")}。`;
    consensusBox.textContent = consensusText;
}

function renderModelCards(models) {
    modelResultsGrid.innerHTML = models.map((item) => {
        const topRows = (item.top_k || [])
            .slice(0, 2)
            .map((scoreItem) => createBarRow(scoreItem.label, formatPercent(scoreItem.score), scoreItem.score))
            .join("");
        const roiText = item.roi?.roi_applied
            ? `已裁剪 (${item.roi.detector_used || "ROI"})`
            : `未裁剪 (${item.roi?.detector_used || "兜底"})`;
        const qualityClass = item.quality?.is_uncertain ? "pending" : "success";
        const qualityText = item.quality?.is_uncertain ? "低置信度" : "稳定";
        return `
            <article class="model-card">
                <div class="catalog-head">
                    <strong>${item.model.display_name}</strong>
                    <span class="catalog-badge ${qualityClass}">${qualityText}</span>
                </div>
                <div class="model-card-main">
                    <div>
                        <span class="summary-title">预测标签</span>
                        <strong class="model-prediction">${item.prediction.label}</strong>
                    </div>
                    <div>
                        <span class="summary-title">Top-1 置信度</span>
                        <strong class="model-confidence">${formatPercent(item.prediction.confidence)}</strong>
                    </div>
                </div>
                <div class="chart-stack">${topRows}</div>
                <div class="catalog-meta compact">
                    <span>在线延迟：${formatMs(item.meta?.inference_time_ms)}</span>
                    <span>ROI：${roiText}</span>
                    <span>test acc：${formatPercent(item.offline_metrics?.test_accuracy)}</span>
                    <span>macro f1：${formatPercent(item.offline_metrics?.macro_f1)}</span>
                    <span>离线延迟：${formatMs(item.offline_metrics?.avg_inference_latency_ms)}</span>
                    <span>参数量：${formatCount(item.offline_metrics?.parameter_count)}</span>
                </div>
                ${item.model?.training_remark ? `<div class="catalog-training-remark" title="该模型训练时使用的设置">训练条件：${item.model.training_remark}</div>` : ""}
                <div class="advice-inline">${item.quality?.advice || ""}</div>
            </article>
        `;
    }).join("");
}

function renderConfidenceChart(data) {
    confidenceChart.innerHTML = data
        .map((item) => createBarRow(`${item.display_name} · ${item.predicted_label}`, formatPercent(item.confidence), item.confidence, item.is_majority_vote ? "primary" : "secondary"))
        .join("");
}

function renderLatencyChart(data) {
    const maxLatency = Math.max(...data.map((item) => Number(item.latency_ms) || 0), 1);
    latencyChart.innerHTML = data
        .map((item) => createBarRow(item.display_name, formatMs(item.latency_ms), (Number(item.latency_ms) || 0) / maxLatency, "secondary"))
        .join("");
}

function renderVoteChart(data) {
    voteChart.innerHTML = data
        .map((item) => createBarRow(`${item.label} (${item.votes} 票)`, formatPercent(item.ratio), item.ratio, "accent"))
        .join("");
}

function renderProbabilityMatrix(rows) {
    probabilityMatrix.innerHTML = rows.map((row) => `
        <article class="matrix-card">
            <div class="catalog-head">
                <strong>${row.display_name}</strong>
            </div>
            <div class="chart-stack">
                ${row.scores.map((scoreItem) => createBarRow(scoreItem.label, formatPercent(scoreItem.score), scoreItem.score, "primary")).join("")}
            </div>
        </article>
    `).join("");
}

function renderPrediction(payload) {
    lastPredictionPayload = payload;
    resultChip.textContent = "评测完成";
    resultChip.className = "result-chip success";
    dashboardChip.textContent = payload.consensus?.is_unanimous ? "意见一致" : "存在分歧";
    dashboardChip.className = `result-chip ${payload.consensus?.is_unanimous ? "success" : "pending"}`;
    resultState.textContent = `本次输入已完成 ${payload.summary.total_models} 个模型的统一评测。`;
    dashboardState.textContent = payload.consensus?.is_unanimous
        ? `所有模型一致预测为“${payload.summary.majority_label}”。`
        : `多数投票结果为“${payload.summary.majority_label}”，建议结合真值进一步复盘。`;
    dashboardContent.classList.remove("hidden");
    renderSummary(payload);
    renderModelCards(payload.models || []);
    renderConfidenceChart(payload.chart_data?.confidence_comparison || []);
    renderProbabilityMatrix(payload.chart_data?.class_probability_matrix || []);
    renderVoteChart(payload.consensus?.vote_distribution || []);
    renderLatencyChart(payload.chart_data?.latency_comparison || []);
    truthState.textContent = "现在可以填写真实标签，查看正确性与高置信度误判分析。";
    truthChip.textContent = "可开始复盘";
    truthChip.className = "result-chip pending";
}

function renderTruthAnalysis(payload) {
    truthChip.textContent = "复盘完成";
    truthChip.className = "result-chip success";
    truthState.textContent = `真实标签为“${payload.truth_label}”，以下是各模型在该样本上的复盘结果。`;
    truthContent.classList.remove("hidden");
    truthSummaryGrid.innerHTML = [
        createSummaryCard("预测正确模型数", `${payload.summary.correct_models}/${payload.summary.total_models}`, `正确率 ${formatPercent(payload.summary.correct_ratio)}`),
        createSummaryCard("多数投票是否正确", payload.summary.majority_vote_correct ? "是" : "否", payload.summary.majority_vote_label),
        createSummaryCard("正确模型平均置信度", formatPercent(payload.summary.avg_confidence_correct), "越高说明正确预测更坚定"),
        createSummaryCard("错误模型平均置信度", formatPercent(payload.summary.avg_confidence_wrong), `高置信度误判 ${payload.summary.high_confidence_miss_count} 个`),
    ].join("");

    truthInsights.innerHTML = (payload.insights || []).map((item) => `<div class="insight-pill">${item}</div>`).join("");
    correctnessChart.innerHTML = (payload.chart_data?.correctness_comparison || [])
        .map((item) => createBarRow(`${item.display_name} · ${item.is_correct ? "正确" : "错误"}`, formatPercent(item.confidence), item.confidence, item.is_correct ? "primary" : "danger"))
        .join("");
    truthScoreChart.innerHTML = (payload.chart_data?.truth_score_comparison || [])
        .map((item) => createBarRow(item.display_name, formatPercent(item.truth_score), item.truth_score, "accent"))
        .join("");
    truthTable.innerHTML = (payload.per_model || []).map((item) => `
        <article class="truth-row ${item.is_correct ? "correct" : "wrong"}">
            <div class="catalog-head">
                <strong>${item.display_name}</strong>
                <span class="catalog-badge ${item.is_correct ? "success" : "error"}">${item.is_correct ? "正确" : "错误"}</span>
            </div>
            <div class="catalog-meta compact">
                <span>预测：${item.predicted_label}</span>
                <span>预测置信度：${formatPercent(item.confidence)}</span>
                <span>真实类别得分：${formatPercent(item.truth_score)}</span>
                <span>置信度偏差：${formatPercent(item.confidence_gap)}</span>
                <span>${item.is_high_confidence_miss ? "高置信度误判" : "无明显过度自信"}</span>
            </div>
            ${item.training_remark ? `<div class="catalog-training-remark">训练条件：${item.training_remark}</div>` : ""}
        </article>
    `).join("");
}

async function predictWithFormData(formData) {
    setLoading("正在调用全部模型进行统一评测，请稍候...");
    uploadSubmit.disabled = true;
    truthSubmit.disabled = true;
    captureFrameButton.disabled = true;
    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok || payload.status === "error") {
            renderError(payload);
            return;
        }
        renderPrediction(payload);
    } catch (error) {
        renderError({ user_message: "请求失败，请检查网络或服务状态。", error_message: error.message });
    } finally {
        uploadSubmit.disabled = false;
        truthSubmit.disabled = false;
        captureFrameButton.disabled = !cameraStream;
    }
}

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) {
        renderError({ user_message: "请先选择一张图片。", error_message: "Missing upload file." });
        return;
    }

    showPreview(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append("image", file);
    formData.append("input_source", "upload");
    await predictWithFormData(formData);
});

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        showPreview(URL.createObjectURL(file));
    }
});

startCameraButton.addEventListener("click", async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        camera.srcObject = cameraStream;
        captureFrameButton.disabled = false;
        cameraTip.textContent = "摄像头已开启，请将人物面部保持在中央后点击“抓帧评测”。";
    } catch (error) {
        renderError({ user_message: "摄像头打开失败，请检查权限。", error_message: error.message });
    }
});

captureFrameButton.addEventListener("click", async () => {
    if (!cameraStream) {
        renderError({ user_message: "请先打开摄像头。", error_message: "Camera stream not ready." });
        return;
    }

    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(camera, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg");
    showPreview(dataUrl);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
    const formData = new FormData();
    formData.append("image", blob, "camera_capture.jpg");
    formData.append("input_source", "camera");
    await predictWithFormData(formData);
});

truthForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!lastPredictionPayload?.models?.length) {
        truthState.textContent = "当前还没有可复盘的预测结果，请先完成一次在线评测。";
        return;
    }
    if (!truthSelect.value) {
        truthState.textContent = "请先选择真实标签。";
        return;
    }

    truthSubmit.disabled = true;
    truthChip.textContent = "分析中";
    truthChip.className = "result-chip pending";
    truthState.textContent = "正在根据真实标签生成复盘分析...";

    try {
        const response = await fetch("/analyze-truth", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                truth_class_name: truthSelect.value,
                predictions: lastPredictionPayload.models,
            }),
        });
        const payload = await response.json();
        if (!response.ok || payload.status === "error") {
            truthChip.textContent = "分析失败";
            truthChip.className = "result-chip error";
            truthState.textContent = payload.user_message || "真实标签分析失败。";
            truthContent.classList.add("hidden");
            return;
        }
        renderTruthAnalysis(payload);
    } catch (error) {
        truthChip.textContent = "分析失败";
        truthChip.className = "result-chip error";
        truthState.textContent = `真实标签分析失败：${error.message}`;
        truthContent.classList.add("hidden");
    } finally {
        truthSubmit.disabled = false;
    }
});

inputModeButtons.forEach((button) => {
    button.addEventListener("click", () => setInputMode(button.dataset.mode));
});

if (!classNames.length) {
    resultState.textContent = "当前没有可用类别信息，请先加载至少一个模型。";
}
if (!modelCatalog.length) {
    resultState.textContent = "当前没有可用模型，请先训练或准备权重。";
}

setInputMode("upload");
captureFrameButton.disabled = true;
resetTruthAnalysis();
