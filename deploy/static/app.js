/* ═══════════════════ State ═══════════════════ */
let currentFile = null;
let currentModel = '';
let compareFile = null;
let stream = null;
let autoInterval = null;
let modelsMeta = [];

const $ = document.getElementById.bind(document);

/* ═══════════════════ Tab Switching ═══════════════════ */
const switchTab = (name) => {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === 'panel-' + name));
  if (name !== 'webcam') { stopWebcam(); stopAuto(); }
};

document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => switchTab(t.dataset.tab)));

/* ═══════════════════ Model Select ═══════════════════ */
$('model-select').addEventListener('change', function () {
  currentModel = this.value;
});

/* ═══════════════════ File Handling ═══════════════════ */
const handleFile = (file) => {
  if (!file) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    $('preview').src = e.target.result;
    $('preview').style.display = 'block';
    $('upload-placeholder').style.display = 'none';
    $('upload-btn').disabled = false;
    $('upload-results').innerHTML = '';
    $('upload-time').innerHTML = '';
    $('upload-empty').style.display = 'block';
  };
  reader.readAsDataURL(file);
};

const resetUpload = () => {
  currentFile = null;
  $('preview').style.display = 'none';
  $('upload-placeholder').style.display = 'block';
  $('file-input').value = '';
  $('upload-btn').disabled = true;
  $('upload-results').innerHTML = '';
  $('upload-time').innerHTML = '';
  $('upload-empty').style.display = 'block';
};

$('file-input').addEventListener('change', function () {
  handleFile(this.files[0]);
});
$('upload-btn').addEventListener('click', () => classifyUpload());
$('upload-reset-btn').addEventListener('click', () => resetUpload());

/* ═══════════════════ Drag & Drop ═══════════════════ */
const setupDragDrop = (zoneId, handler) => {
  const zone = $(zoneId);
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handler(e.dataTransfer.files[0]);
  });
};

setupDragDrop('drop-zone', (f) => handleFile(f));
setupDragDrop('compare-drop-zone', (f) => handleCompareFile(f));

$('drop-zone').addEventListener('click', () => $('file-input').click());
$('compare-drop-zone').addEventListener('click', () => $('compare-file-input').click());

/* ═══════════════════ Spinner ═══════════════════ */
const showSpinner = (id) => $(id).classList.add('show');
const hideSpinner = (id) => $(id).classList.remove('show');

/* ═══════════════════ API Helper ═══════════════════ */
const apiClassify = async (file, model) => {
  const fd = new FormData();
  fd.append('image', file);
  const r = await fetch('/api/classify?model=' + encodeURIComponent(model), {
    method: 'POST', body: fd,
  });
  return r.json();
};

/* ═══════════════════ Upload Classification ═══════════════════ */
const classifyUpload = async () => {
  if (!currentFile) return;
  const btn = $('upload-btn');
  btn.disabled = true;
  btn.textContent = '\u23F3 识别中...';
  showSpinner('upload-spinner');
  try {
    const d = await apiClassify(currentFile, currentModel);
    if (d.error) { alert(d.error); return; }
    $('upload-empty').style.display = 'none';
    renderResults('upload-results', d.results);
    $('upload-time').textContent = d.model + ' | ' + d.time_ms + ' ms';
  } catch (e) {
    alert('上传失败: ' + (e.message || e));
  } finally {
    hideSpinner('upload-spinner');
    btn.disabled = false;
    btn.textContent = '\uD83D\uDD0D 开始识别';
  }
};

/* ═══════════════════ Webcam ═══════════════════ */
let cameraDevices = [];
let selectedCameraId = null;

const enumerateCameras = async () => {
  try {
    const all = await navigator.mediaDevices.enumerateDevices();
    cameraDevices = all.filter(d => d.kind === 'videoinput');
    const sel = $('webcam-select');
    sel.innerHTML = cameraDevices.map((d, i) =>
      `<option value="${d.deviceId}">${d.label || '摄像头 ' + (i + 1)}</option>`).join('');
    if (cameraDevices.length > 1) {
      sel.classList.remove('hidden');
      if (selectedCameraId && cameraDevices.find(d => d.deviceId === selectedCameraId)) {
        sel.value = selectedCameraId;
      } else {
        selectedCameraId = cameraDevices[0].deviceId;
      }
    } else {
      sel.classList.add('hidden');
    }
  } catch (e) {}
};

const toggleWebcam = async () => {
  const btn = $('webcam-btn');
  const cap = $('capture-btn');
  const autoBtn = $('auto-btn');
  const v = $('webcam-video');
  if (stream) {
    stopWebcam(); stopAuto();
    btn.textContent = '\uD83D\uDCF8 开启摄像头';
    cap.classList.add('hidden');
    autoBtn.classList.add('hidden');
    v.style.display = 'none';
    $('auto-badge').classList.remove('on');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
    });
    v.srcObject = stream;
    await v.play().catch(() => {});
    v.style.display = 'block';
    btn.textContent = '\u23F9 关闭摄像头';
    cap.classList.remove('hidden');
    autoBtn.classList.remove('hidden');
    enumerateCameras();  // 成功后异步枚举，填充下拉框
  } catch (e) {
    alert('摄像头访问失败:\n' +
      '1. 请确认已授予摄像头权限\n' +
      '2. 必须通过 localhost 或 HTTPS 访问\n' +
      '3. 确认摄像头未被其他应用占用');
  }
};

$('webcam-select').addEventListener('change', async function () {
  if (!stream) return;
  stopWebcam();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: this.value }, width: { ideal: 640 }, height: { ideal: 480 } },
    });
    $('webcam-video').srcObject = stream;
    await $('webcam-video').play().catch(() => {});
  } catch (e) {}
});

const stopWebcam = () => {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('webcam-video').srcObject = null;
};

const toggleAuto = () => {
  if (autoInterval) { stopAuto(); return; }
  const intervalMs = parseInt($('webcam-interval').dataset.value) || 1500;
  autoInterval = setInterval(captureFrame, intervalMs);
  $('auto-btn').textContent = '\u23F1 停止';
  $('auto-badge').classList.add('on');
  captureFrame();
};

const stopAuto = () => {
  if (autoInterval) { clearInterval(autoInterval); autoInterval = null; }
  $('auto-btn').textContent = '\u23F1 自动';
  $('auto-badge').classList.remove('on');
};

$('webcam-btn').addEventListener('click', () => toggleWebcam());
$('capture-btn').addEventListener('click', () => captureFrame());
$('auto-btn').addEventListener('click', () => toggleAuto());

const captureFrame = async () => {
  const v = $('webcam-video');
  const c = $('webcam-canvas');
  if (!v.videoWidth) return;
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const btn = $('capture-btn');
  if (!autoInterval) { btn.disabled = true; btn.textContent = '\u23F3 ...'; }
  c.toBlob(async (blob) => {
    try {
      const d = await apiClassify(blob, currentModel);
      if (d.error) return;
      renderResults('webcam-results', d.results);
      $('webcam-time').textContent = d.model + ' | ' + d.time_ms + ' ms';
    } catch (e) {}
    if (!autoInterval) { btn.disabled = false; btn.textContent = '\uD83D\uDCF7 拍照识别'; }
  }, 'image/jpeg', 0.85);
};

/* ═══════════════════ Result Rendering ═══════════════════ */
const resultItemHtml = (r, i) => {
  const pct = (r.confidence * 100).toFixed(1);
  const gold = i === 0 ? ' gold' : '';
  return (
    `<div class="result-item">
      <div class="result-rank${gold}">${i + 1}</div>
      <div class="result-name">${r.name}</div>
      <div class="result-bar-bg"><div class="result-bar${gold}" style="width:${pct}%"></div></div>
      <div class="result-conf">${pct}%</div>
    </div>`
  );
};

const renderResults = (containerId, results) => {
  $(containerId).innerHTML = results.map(resultItemHtml).join('');
};

/* ═══════════════════ Compare ═══════════════════ */
const handleCompareFile = (file) => {
  if (!file) return;
  compareFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    $('compare-preview').src = e.target.result;
    $('compare-preview').style.display = 'block';
    $('compare-placeholder').style.display = 'none';
    $('compare-btn').disabled = false;
    $('compare-results').innerHTML = '';
    $('compare-empty').style.display = 'block';
  };
  reader.readAsDataURL(file);
};

const resetCompare = () => {
  compareFile = null;
  $('compare-preview').style.display = 'none';
  $('compare-placeholder').style.display = 'block';
  $('compare-file-input').value = '';
  $('compare-btn').disabled = true;
  $('compare-results').innerHTML = '';
  $('compare-empty').style.display = 'block';
};

$('compare-file-input').addEventListener('change', function () {
  handleCompareFile(this.files[0]);
});
$('compare-btn').addEventListener('click', () => classifyCompare());
$('compare-reset-btn').addEventListener('click', () => resetCompare());

const getSelectedModels = () => {
  const checks = document.querySelectorAll('#model-checklist input[type=checkbox]:checked');
  return [...checks].map(c => c.value);
};

const selectAllModels = () => {
  document.querySelectorAll('#model-checklist input[type=checkbox]').forEach(c => c.checked = true);
};

const deselectAllModels = () => {
  document.querySelectorAll('#model-checklist input[type=checkbox]').forEach(c => c.checked = false);
};

$('select-all-btn').addEventListener('click', () => selectAllModels());
$('deselect-all-btn').addEventListener('click', () => deselectAllModels());

const classifyCompare = async () => {
  if (!compareFile) return;
  const selected = getSelectedModels();
  if (selected.length === 0) { alert('请至少勾选一个模型'); return; }

  const btn = $('compare-btn');
  btn.disabled = true;
  btn.textContent = '\u23F3 对比中...';
  showSpinner('compare-spinner');
  const fd = new FormData();
  fd.append('image', compareFile);

  try {
    const r = await fetch('/api/compare?models=' + encodeURIComponent(selected.join(',')), {
      method: 'POST', body: fd,
    });
    const d = await r.json();
    if (d.error) { alert(d.error); return; }
    $('compare-empty').style.display = 'none';
    renderCompareResults(d.models);
  } catch (e) {
    alert('对比失败: ' + (e.message || e));
  } finally {
    hideSpinner('compare-spinner');
    btn.disabled = false;
    btn.textContent = '\uD83D\uDD0D 对比识别';
  }
};

const renderCompareResults = (models) => {
  // 找出最高置信度模型作为 ⭐ winner
  let bestName = null, bestConf = 0;
  for (const [name, data] of Object.entries(models)) {
    if (data.predictions.length && data.predictions[0].confidence > bestConf) {
      bestConf = data.predictions[0].confidence;
      bestName = name;
    }
  }

  $('compare-results').innerHTML = Object.entries(models).map(([name, data]) => {
    const win = name === bestName;
    return (
      `<div class="compare-card${win ? ' winner' : ''}">
        <div class="compare-card-header">
          <strong>${win ? '\u2B50 ' : ''}${name}</strong>
          <span class="card-time">${data.time_ms} ms</span>
        </div>
        ${data.predictions.map(resultItemHtml).join('')}
      </div>`
    );
  }).join('');
};

/* ═══════════════════ Model Checklist ═══════════════════ */
const renderModelChecklist = () => {
  const container = $('model-checklist');
  if (!modelsMeta.length) {
    container.innerHTML = '<div class="module-empty">暂无模型</div>';
    return;
  }
  container.innerHTML = modelsMeta.map((m, i) => (
    `<div class="model-check-item">
      <input type="checkbox" value="${m.name}" id="cm-${m.name}"${i < 3 ? ' checked' : ''}>
      <label for="cm-${m.name}">${m.name}</label>
      <span class="size-tag">${m.size_mb}MB</span>
    </div>`
  )).join('');
};

/* ═══════════════════ Init ═══════════════════ */
const sel = $('model-select');
const defaultModel = sel.getAttribute('data-default') || '';

fetch('/api/models')
  .then(r => r.json())
  .then(models => {
    sel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
    if (models.length) {
      currentModel = (defaultModel && models.includes(defaultModel)) ? defaultModel : models[0];
      sel.value = currentModel;
    }
  }).catch(() => {});

fetch('/api/model-info')
  .then(r => r.json())
  .then(meta => { modelsMeta = meta; renderModelChecklist(); })
  .catch(() => {});

fetch('/api/classes')
  .then(r => r.json())
  .then(cs => {
    $('class-list').innerHTML = cs.map(c =>
      `<div class="class-tag"><span class="idx">${c.index + 1}</span>${c.name}</div>`
    ).join('');
  });
