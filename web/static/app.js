const i18n = {
  en: {
    title: "PHANES Counterfactual Demo",
    image: "Image",
    color_mode: "Color Mode",
    mask_thr: "Mask Threshold",
    topk: "Top-K %",
    aot_size: "AOT Input Size",
    viz_size: "Visualization Size",
    latent_step: "Latent Step",
    latent_max: "Latent Max",
    enable_topk: "Top-K Transfer",
    enable_adjust: "Latent Adjust",
    run: "Run",
  },
  zh: {
    title: "PHANES 反事实演示",
    image: "图像",
    color_mode: "色彩模式",
    mask_thr: "掩膜阈值",
    topk: "Top-K 百分比",
    aot_size: "AOT 输入尺寸",
    viz_size: "可视化尺寸",
    latent_step: "潜变量步长",
    latent_max: "潜变量最大系数",
    enable_topk: "启用 Top-K 迁移",
    enable_adjust: "启用潜变量调整序列",
    run: "执行",
  },
};

let currentLang = "en";
let layout = "vertical";
let progressBar = null;
const advBody = document.querySelector(".advanced-body");
const advToggle = document.getElementById("toggle-adv");
const form = document.getElementById("analyze-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsHeaderEl = document.getElementById("results-header");
const previewEl = document.getElementById("preview");
const modal = document.getElementById("modal");
const modalImg = document.getElementById("modal-img");
const modalLabel = document.getElementById("modal-label");
const modalClose = document.getElementById("modal-close");
const resetBtn = document.getElementById("reset-btn");
const sampleSelect = document.getElementById("sample-select");
let gallery = [];
let currentIdx = 0;
let saved = {};

function setLang(lang) {
  currentLang = lang;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (i18n[lang][key]) {
      el.textContent = i18n[lang][key];
    }
  });
}

async function initConfig() {
  try {
    const res = await fetch("/config");
    const cfg = await res.json();
    layout = cfg.layout || "vertical";
    document.body.classList.add(`layout-${layout}`);
  } catch (e) {
    document.body.classList.add("layout-vertical");
  }
  setLang("en");
  console.log("[UI] init complete, layout=", layout);
  // load sample list
  try {
    const ex = await fetch("/examples");
    const data = await ex.json();
    if (Array.isArray(data.examples)) {
      data.examples.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sampleSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn("[UI] failed to load examples", e);
  }
}
initConfig();

// restore saved params
try {
  saved = JSON.parse(localStorage.getItem("phanes_form") || "{}");
} catch (e) {
  console.warn("[UI] invalid saved form, clearing");
  localStorage.removeItem("phanes_form");
  saved = {};
}
for (const [k, v] of Object.entries(saved)) {
  if (!form[k]) continue;
  if (form[k].type === "checkbox") {
    form[k].checked = !!v;
  } else if (form[k].type === "file" || k === "sample_name") {
    continue; // do not restore file or sample to avoid stale preview
  } else {
    form[k].value = v;
  }
}
if (previewEl) previewEl.innerHTML = "";
if (sampleSelect) sampleSelect.value = "";

if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    localStorage.removeItem("phanes_form");
    form.reset();
    previewEl.innerHTML = "";
    gallery = [];
    currentIdx = 0;
    if (sampleSelect) sampleSelect.value = "";
  });
}

if (advToggle && advBody) {
  advToggle.addEventListener("click", () => {
    advBody.classList.toggle("open");
    advToggle.textContent = advBody.classList.contains("open") ? "▲" : "▼";
  });
}

form.image.addEventListener("change", (e) => {
  const file = e.target.files[0];
  // If user picks a file, clear sample selection
  if (sampleSelect) {
    sampleSelect.value = "";
  }
  if (!file) {
    previewEl.innerHTML = "";
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    previewEl.innerHTML = `
      <img src="${reader.result}" alt="preview">
      <span>${file.name}</span>
    `;
  };
  reader.readAsDataURL(file);
});

if (sampleSelect) {
  sampleSelect.addEventListener("change", (e) => {
    // Clear file input when choosing a sample
    try {
      form.image.value = "";
    } catch (_) {
      form.image.value = null;
    }
    if (!e.target.value) {
      previewEl.innerHTML = "";
      return;
    }
    previewEl.innerHTML = `<div class="preview-label">Sample: ${e.target.value}</div>`;
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  console.log("[UI] submit clicked");
  if (!form.image.files[0]) {
    if (!sampleSelect.value) {
      statusEl.textContent = currentLang === "zh" ? "请选择图像或示例" : "Please upload or choose a sample";
      return;
    }
  }
  statusEl.textContent = "Running...";
  resultsEl.innerHTML = "";
  resultsHeaderEl.innerHTML = "";
  document.body.classList.add("loading");
  updateProgress(5, currentLang === "zh" ? "准备中..." : "Preparing...");

  const formData = new FormData(form);
  // persist params
  const toSave = {};
  for (const el of Array.from(form.elements)) {
    if (!el.name) continue;
    toSave[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  // do not persist preview blob to avoid stale images on reload
  localStorage.setItem("phanes_form", JSON.stringify(toSave));
  // Checkboxes: ensure boolean sent
  formData.set("do_topk_transfer", form.do_topk_transfer.checked);
  formData.set("do_latent_adjust", form.do_latent_adjust.checked);
  // sample name
  formData.set("sample_name", sampleSelect.value || "");

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });
    updateProgress(50, currentLang === "zh" ? "推理中..." : "Inferencing...");
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = `Error: ${data.error || res.statusText}`;
      document.body.classList.remove("loading");
      updateProgress(0, "");
      return;
    }
    statusEl.textContent = currentLang === "zh" ? "完成" : "Done";
    document.body.classList.remove("loading");
    updateProgress(100, currentLang === "zh" ? "完成" : "Done");
    renderResults(data);
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err}`;
    document.body.classList.remove("loading");
    updateProgress(0, "");
  }
});

function renderResults(data) {
  resultsEl.innerHTML = "";
  resultsHeaderEl.innerHTML = "";
  gallery = [];
  currentIdx = 0;
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const inputName = form.image.files[0]?.name || (sampleSelect.value ? `sample_${sampleSelect.value}` : "result");
  const entries = [
    ["Input", data.input],
    ["Mask", data.mask],
    ["Pseudo", data.pseudo],
    ["Coarse", data.coarse],
    ["Overlay (mean)", data.overlay_mean],
    ["Overlay (max)", data.overlay_max],
    ["Latent Adjust Panel", data.adjust_panel],
    ["Latent Heatmap 7x7", data.latent_heatmap_7x7],
  ];
  if (entries.some(([, v]) => !!v)) {
    const bar = document.createElement("div");
    bar.style.display = "flex";
    bar.style.justifyContent = "flex-end";
    bar.style.gap = "8px";
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "clear-btn";
    clear.textContent = currentLang === "zh" ? "清空结果" : "Clear Results";
    clear.onclick = () => {
      resultsEl.innerHTML = "";
      resultsHeaderEl.innerHTML = "";
      gallery = [];
      currentIdx = 0;
    };
    bar.appendChild(clear);
    if (data.zip) {
      const btn = document.createElement("button");
      btn.className = "download-btn";
      btn.textContent = currentLang === "zh" ? "下载全部" : "Download All";
      btn.onclick = () => {
        const link = document.createElement("a");
        link.href = `data:application/zip;base64,${data.zip}`;
        link.download = "results.zip";
        link.click();
      };
      bar.appendChild(btn);
    }
    resultsHeaderEl.appendChild(bar);
  }
  entries.forEach(([label, b64]) => {
    if (!b64) return;
    const card = document.createElement("div");
    card.className = "card";
    if (label.includes("Adjust") || label.includes("Top-K") || label.includes("Heatmap")) {
      card.classList.add("wide");
    }
    const h = document.createElement("h3");
    h.textContent = label;
    const descMap = {
      "Input": "Original image",
      "Mask": "Auto-generated mask",
      "Pseudo": "Pseudo-healthy reconstruction",
      "Coarse": "Coarse AAE recon",
      "Overlay (mean)": "Latent diff mean",
      "Overlay (max)": "Latent diff max",
      "Latent Adjust Panel": "Latent interpolation variants",
      "Latent Heatmap 7x7": "Aggregated latent deviation map",
    };
    const desc = document.createElement("div");
    desc.className = "desc";
    desc.textContent = descMap[label] || "";
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${b64}`;
    img.onclick = () => openModal(label, img.src);
    card.appendChild(h);
    if (desc.textContent) card.appendChild(desc);
    card.appendChild(img);
    const dl = document.createElement("button");
    dl.className = "download-btn";
    dl.textContent = currentLang === "zh" ? "下载" : "Download";
    dl.onclick = () => {
      const link = document.createElement("a");
      link.href = img.src;
      link.download = `${inputName.replace(/\\s+/g, '_')}_${label.replace(/\\s+/g, '_').toLowerCase()}_${ts}.png`;
      link.click();
    };
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `...`;
    img.onload = () => {
      meta.textContent = `${img.naturalWidth}x${img.naturalHeight} | ${ts}`;
    };
    const footer = document.createElement("div");
    footer.style.display = "flex";
    footer.style.justifyContent = "space-between";
    footer.style.alignItems = "center";
    footer.appendChild(meta);
    footer.appendChild(dl);
    card.appendChild(footer);
    resultsEl.appendChild(card);
    gallery.push({ label, src: img.src });
  });
}

function updateProgress(pct, text) {
  if (!progressBar) {
    progressBar = document.createElement("div");
    progressBar.className = "progress";
    progressBar.innerHTML = `<div class="progress-inner"></div><span class="progress-text"></span>`;
    const anchor = document.querySelector(".progress-inline");
    (anchor || document.body).appendChild(progressBar);
  }
  const inner = progressBar.querySelector(".progress-inner");
  const label = progressBar.querySelector(".progress-text");
  inner.style.width = `${pct}%`;
  label.textContent = text || "";
  if (pct === 0 && !text) {
    progressBar.classList.remove("show");
  } else {
    progressBar.classList.add("show");
  }
}

function openModal(label, src) {
  const idx = gallery.findIndex(g => g.src === src);
  currentIdx = idx >= 0 ? idx : 0;
  const item = gallery[currentIdx];
  modalImg.src = item.src;
  modalLabel.textContent = `${item.label} (${currentIdx + 1}/${gallery.length})`;
  modal.classList.add("show");
}
modalClose.addEventListener("click", () => modal.classList.remove("show"));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.classList.remove("show");
});

document.addEventListener("keydown", (e) => {
  if (!modal.classList.contains("show")) return;
  if (e.key === "ArrowRight") {
    currentIdx = (currentIdx + 1) % gallery.length;
  } else if (e.key === "ArrowLeft") {
    currentIdx = (currentIdx - 1 + gallery.length) % gallery.length;
  } else {
    return;
  }
  const item = gallery[currentIdx];
  modalImg.src = item.src;
  modalLabel.textContent = `${item.label} (${currentIdx + 1}/${gallery.length})`;
});
