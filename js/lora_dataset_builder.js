import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "comfyui.lora.dataset.builder";
const CAPTION_PROVIDER_CLASS = "LoraDatasetCaptionProvider";
const RUN_SUMMARY_CLASS = "LoraDatasetRunSummary";

function installDatasetProgressPanel() {
  const panel = document.createElement("div");
  panel.style.cssText = [
    "display:none",
    "position:fixed",
    "top:76px",
    "right:24px",
    "z-index:10020",
    "width:min(420px,calc(100vw - 32px))",
    "padding:14px",
    "border:1px solid rgba(96,165,250,0.45)",
    "border-radius:12px",
    "background:rgba(15,23,42,0.96)",
    "box-shadow:0 14px 38px rgba(0,0,0,0.35)",
    "color:#e2e8f0",
    "font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace",
  ].join(";");

  const title = document.createElement("div");
  title.style.cssText = "font-weight:700;font-size:13px;margin-bottom:8px;color:#f8fafc;";
  title.textContent = "LoRA Dataset Builder";
  const status = document.createElement("div");
  const current = document.createElement("div");
  current.style.cssText = "margin-top:5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
  const track = document.createElement("div");
  track.style.cssText = "height:6px;margin-top:10px;border-radius:999px;overflow:hidden;background:#1e293b;";
  const fill = document.createElement("div");
  fill.style.cssText = "height:100%;width:0%;background:#3b82f6;transition:width 160ms ease;";
  track.appendChild(fill);
  panel.append(title, status, current, track);
  document.body.appendChild(panel);

  let hideTimer;
  app.api.addEventListener("lora_dataset.progress", (event) => {
    const detail = event?.detail || {};
    const processed = Number(detail.processed || 0);
    const total = Number(detail.total || 0);
    const state = String(detail.status || "running");
    clearTimeout(hideTimer);
    panel.style.display = "block";
    const percentage = total > 0 ? Math.min(100, Math.max(0, (processed / total) * 100)) : state === "complete" ? 100 : 0;
    fill.style.width = `${percentage}%`;
    fill.style.background = state === "error" ? "#f43f5e" : state === "complete" ? "#22c55e" : "#3b82f6";
    if (state === "error") {
      status.textContent = `Stopped: ${detail.message || "processing error"}`;
    } else if (state === "complete") {
      status.textContent = `Complete — ${processed}/${total} selected item(s)`;
    } else {
      status.textContent = `Processing ${processed}/${total} selected item(s) — ${Number(detail.failed || 0)} failed, ${Number(detail.excluded || 0)} excluded`;
    }
    current.textContent = detail.current_file ? `Current: ${detail.current_file}` : "Preparing manifest and dataset state…";
    if (state === "complete" || state === "error") {
      hideTimer = setTimeout(() => { panel.style.display = "none"; }, 6000);
    }
  });
}

const DEFAULT_URLS = {
  Ollama: "http://localhost:11434/v1",
  OpenRouter: "https://openrouter.ai/api/v1",
  NanoGPT: "https://nano-gpt.com/api/v1",
  Kobold: "http://localhost:5001/v1",
};

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  if (widget.inputEl) widget.inputEl.value = value;
}

function buildCaptionProviderPanel(node) {
  const panel = document.createElement("div");
  panel.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;width:100%;";

  const statusRow = document.createElement("div");
  statusRow.style.cssText = "display:flex;gap:6px;align-items:center;";

  const statusDot = document.createElement("span");
  statusDot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;background:#666;flex-shrink:0;";

  const statusText = document.createElement("span");
  statusText.style.cssText = "font-size:11px;opacity:0.8;flex:1;";
  statusText.textContent = "Not connected";

  const connectButton = document.createElement("button");
  connectButton.textContent = "🔌 Connect";
  connectButton.style.cssText = "cursor:pointer;font-size:11px;padding:3px 10px;border-radius:4px;border:1px solid rgba(100,180,255,0.4);background:rgba(100,180,255,0.1);color:rgba(150,210,255,1);white-space:nowrap;";

  statusRow.appendChild(statusDot);
  statusRow.appendChild(statusText);
  statusRow.appendChild(connectButton);

  const modelRow = document.createElement("div");
  modelRow.style.cssText = "display:none;flex-direction:column;gap:3px;";

  const modelLabel = document.createElement("div");
  modelLabel.style.cssText = "font-size:10px;opacity:0.7;";
  modelLabel.textContent = "Available API models:";

  const modelSelect = document.createElement("select");
  modelSelect.style.cssText = "width:100%;padding:4px 8px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);background:rgba(0,0,0,0.3);color:inherit;font-size:12px;";

  modelRow.appendChild(modelLabel);
  modelRow.appendChild(modelSelect);
  panel.appendChild(statusRow);
  panel.appendChild(modelRow);

  const setStatus = (state, text) => {
    statusDot.style.background = state === null ? "#aaa" : state ? "#4caf50" : "#f44336";
    statusText.textContent = text;
  };

  connectButton.addEventListener("click", async () => {
    const backendWidget = node.widgets?.find((widget) => widget.name === "backend");
    const apiUrlWidget = node.widgets?.find((widget) => widget.name === "api_url");
    const apiKeyWidget = node.widgets?.find((widget) => widget.name === "api_key");
    const openRouterKeyWidget = node.widgets?.find((widget) => widget.name === "openrouter_key");
    const nanoGptKeyWidget = node.widgets?.find((widget) => widget.name === "nanogpt_key");
    const modelNameWidget = node.widgets?.find((widget) => widget.name === "model_name");

    const backend = backendWidget?.value || "Ollama";
    const apiUrl = String(apiUrlWidget?.value || DEFAULT_URLS[backend] || "").trim();
    let apiKey = String(apiKeyWidget?.value || "").trim();
    if (!apiKey && backend === "OpenRouter") apiKey = String(openRouterKeyWidget?.value || "").trim();
    if (!apiKey && backend === "NanoGPT") apiKey = String(nanoGptKeyWidget?.value || "").trim();

    if (!apiUrl) {
      setStatus(false, "No API URL set");
      return;
    }
    if (!apiUrlWidget?.value) setWidgetValue(apiUrlWidget, apiUrl);

    connectButton.textContent = "⏳ Connecting...";
    connectButton.disabled = true;
    setStatus(null, "Connecting...");

    try {
      const response = await fetch("/lora_dataset_caption_suite/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_url: apiUrl, api_key: apiKey }),
      });
      const data = await response.json();
      if (!data.ok || !data.models?.length) {
        throw new Error(data.error || "No models found");
      }

      const currentModel = String(modelNameWidget?.value || "");
      modelSelect.innerHTML = "";
      for (const model of data.models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        option.selected = model === currentModel;
        modelSelect.appendChild(option);
      }
      if (currentModel && !data.models.includes(currentModel)) {
        const option = document.createElement("option");
        option.value = currentModel;
        option.textContent = `${currentModel} (current)`;
        option.selected = true;
        modelSelect.insertBefore(option, modelSelect.firstChild);
      } else if (!currentModel && modelSelect.value) {
        setWidgetValue(modelNameWidget, modelSelect.value);
      }

      modelSelect.onchange = () => {
        setWidgetValue(modelNameWidget, modelSelect.value);
        node.setDirtyCanvas(true, true);
      };
      modelRow.style.display = "flex";
      setStatus(true, `Connected — ${data.models.length} model(s) available`);
      node.setDirtyCanvas(true, true);
    } catch (error) {
      modelRow.style.display = "none";
      setStatus(false, `Error: ${error.message}`);
    } finally {
      connectButton.textContent = "🔌 Connect";
      connectButton.disabled = false;
    }
  });

  return panel;
}

app.registerExtension({
  name: EXTENSION_NAME,

  async setup() {
    installDatasetProgressPanel();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === RUN_SUMMARY_CLASS) {
      const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalOnNodeCreated?.apply(this, arguments);
        const panel = document.createElement("pre");
        panel.textContent = "Connect status_json and queue the Builder.";
        panel.style.cssText = [
          "box-sizing:border-box",
          "width:100%",
          "min-height:180px",
          "max-height:520px",
          "overflow:auto",
          "margin:0",
          "padding:10px",
          "border:1px solid rgba(255,255,255,0.15)",
          "border-radius:6px",
          "background:rgba(0,0,0,0.28)",
          "color:inherit",
          "font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace",
          "white-space:pre-wrap",
          "word-break:break-word",
        ].join(";");
        this.loraDatasetSummaryPanel = panel;
        const domWidget = this.addDOMWidget("run_summary", "LoraDatasetRunSummary", panel, {
          serialize: false,
          hideOnZoom: false,
          getValue: () => null,
          setValue: () => {},
        });
        if (domWidget) domWidget.computeSize = () => [520, Math.max(200, panel.scrollHeight + 20)];
        this.size = [Math.max(this.size?.[0] || 0, 540), Math.max(this.size?.[1] || 0, 280)];
        return result;
      };

      const originalOnExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        const text = Array.isArray(message?.text) ? message.text.join("\n") : String(message?.text || "");
        if (this.loraDatasetSummaryPanel && text) {
          this.loraDatasetSummaryPanel.textContent = text;
          this.setSize([Math.max(this.size?.[0] || 0, 540), Math.max(280, Math.min(600, this.loraDatasetSummaryPanel.scrollHeight + 100))]);
          this.setDirtyCanvas(true, true);
        }
      };
      return;
    }

    if (nodeData.name !== CAPTION_PROVIDER_CLASS) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      const node = this;
      const backendWidget = node.widgets?.find((widget) => widget.name === "backend");
      const apiUrlWidget = node.widgets?.find((widget) => widget.name === "api_url");
      const apiKeyWidget = node.widgets?.find((widget) => widget.name === "api_key");
      const openRouterKeyWidget = node.widgets?.find((widget) => widget.name === "openrouter_key");
      const nanoGptKeyWidget = node.widgets?.find((widget) => widget.name === "nanogpt_key");
      let initialized = false;
      setTimeout(() => { initialized = true; }, 500);

      if (backendWidget && apiUrlWidget) {
        const originalBackendCallback = backendWidget.callback;
        backendWidget.callback = function (value) {
          originalBackendCallback?.call(this, value);
          if (initialized && DEFAULT_URLS[value]) setWidgetValue(apiUrlWidget, DEFAULT_URLS[value]);
          if (value === "OpenRouter" && openRouterKeyWidget?.value) {
            setWidgetValue(apiKeyWidget, openRouterKeyWidget.value);
          } else if (value === "NanoGPT" && nanoGptKeyWidget?.value) {
            setWidgetValue(apiKeyWidget, nanoGptKeyWidget.value);
          } else if (initialized && (value === "Ollama" || value === "Kobold")) {
            setWidgetValue(apiKeyWidget, "");
          }
          node.setDirtyCanvas(true, true);
        };
      }

      if (apiKeyWidget) {
        const originalKeyCallback = apiKeyWidget.callback;
        apiKeyWidget.callback = function (value) {
          originalKeyCallback?.call(this, value);
          if (backendWidget?.value === "OpenRouter") setWidgetValue(openRouterKeyWidget, value);
          if (backendWidget?.value === "NanoGPT") setWidgetValue(nanoGptKeyWidget, value);
        };
      }

      const panel = buildCaptionProviderPanel(node);
      const domWidget = node.addDOMWidget("caption_provider_models", "LoraCaptionModels", panel, {
        serialize: false,
        hideOnZoom: true,
        getValue: () => null,
        setValue: () => {},
      });
      if (domWidget) {
        domWidget.computeSize = () => [panel.scrollWidth || 460, panel.scrollHeight || 60];
      }
      node.size = [Math.max(node.size?.[0] || 0, 460), Math.max(node.size?.[1] || 0, 430)];
      return result;
    };
  },
});
