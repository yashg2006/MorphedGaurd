/* ══════════════════════════════════════════════════════════
   MorphGuard — Frontend Application Logic
   ══════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────
const state = {
    selectedFiles: [],
    activeTasks: new Map(),    // task_id -> status
    allResults: [],
    monitorInterval: null,
    pollIntervals: new Map(),
};

// ── DOM Elements ─────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initTabs();
    initUpload();
    startOSMonitor();
});

// ══════════════════════════════════════════════════════════
//  Particles Background
// ══════════════════════════════════════════════════════════
function initParticles() {
    const container = $('#particles');
    if (!container) return;
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.top = (60 + Math.random() * 40) + '%';
        p.style.animationDelay = (Math.random() * 6) + 's';
        p.style.animationDuration = (4 + Math.random() * 4) + 's';
        const hue = Math.random() > 0.5 ? '170' : '260';
        p.style.background = `hsl(${hue}, 100%, 70%)`;
        container.appendChild(p);
    }
}

// ══════════════════════════════════════════════════════════
//  Tab Navigation
// ══════════════════════════════════════════════════════════
function initTabs() {
    $$('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            $$('.nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            $(`#tab-${target}`).classList.add('active');
        });
    });
}

// ══════════════════════════════════════════════════════════
//  Upload System
// ══════════════════════════════════════════════════════════
function initUpload() {
    const zone = $('#upload-zone');
    const fileInput = $('#file-input');
    const btnSelect = $('#btn-select-files');
    const btnBatch = $('#btn-batch');
    const btnAnalyze = $('#btn-analyze');
    const btnClear = $('#btn-clear');

    // Click to browse
    btnSelect.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    btnBatch.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.setAttribute('multiple', '');
        fileInput.click();
    });

    zone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(Array.from(e.target.files));
    });

    // Drag & drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleFiles(Array.from(e.dataTransfer.files));
    });

    // Analyze button
    btnAnalyze.addEventListener('click', startAnalysis);
    btnClear.addEventListener('click', clearFiles);
}

function handleFiles(files) {
    const valid = files.filter(f => {
        const ext = f.name.split('.').pop().toLowerCase();
        return ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'].includes(ext);
    });

    if (valid.length === 0) return;

    state.selectedFiles = [...state.selectedFiles, ...valid];
    renderFilePreviews();
    $('#analyze-controls').style.display = 'flex';
}

function renderFilePreviews() {
    const grid = $('#file-preview-grid');
    grid.innerHTML = '';

    state.selectedFiles.forEach((file, i) => {
        const card = document.createElement('div');
        card.className = 'file-preview-card';

        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = file.name;
            card.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.style.cssText = 'height:100px;display:flex;align-items:center;justify-content:center;font-size:2rem;background:rgba(255,255,255,0.02);';
            placeholder.textContent = '📄';
            card.appendChild(placeholder);
        }

        const name = document.createElement('div');
        name.className = 'file-name';
        name.textContent = file.name;
        card.appendChild(name);

        grid.appendChild(card);
    });
}

function clearFiles() {
    state.selectedFiles = [];
    $('#file-preview-grid').innerHTML = '';
    $('#analyze-controls').style.display = 'none';
    $('#file-input').value = '';
}

// ══════════════════════════════════════════════════════════
//  Analysis Pipeline
// ══════════════════════════════════════════════════════════
async function startAnalysis() {
    if (state.selectedFiles.length === 0) return;

    const progressContainer = $('#progress-container');
    progressContainer.style.display = 'block';
    $('#analyze-controls').style.display = 'none';

    const stages = ['preprocess', 'ela', 'noise', 'exif', 'copymove', 'cnn'];
    stages.forEach(s => {
        $(`#stage-${s}`).classList.remove('active', 'done');
    });

    if (state.selectedFiles.length === 1) {
        await analyzeSingle(state.selectedFiles[0]);
    } else {
        await analyzeBatch(state.selectedFiles);
    }
}

async function analyzeSingle(file) {
    const formData = new FormData();
    formData.append('file', file);

    updateProgress(5, 'Uploading image...');
    activateStage('preprocess');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.task_id) {
            updateProgress(15, 'Analysis started...');
            pollForResult(data.task_id, file.name);
        } else {
            updateProgress(0, `Error: ${data.error}`);
        }
    } catch (err) {
        updateProgress(0, `Upload failed: ${err.message}`);
    }
}

async function analyzeBatch(files) {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    updateProgress(5, 'Uploading batch...');
    activateStage('preprocess');

    try {
        const response = await fetch('/api/batch', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.task_ids) {
            updateProgress(15, `Batch started — ${data.count} images`);
            data.task_ids.forEach((tid, i) => {
                const name = files[i] ? files[i].name : `Image ${i + 1}`;
                pollForResult(tid, name);
            });
        }
    } catch (err) {
        updateProgress(0, `Batch upload failed: ${err.message}`);
    }
}

function pollForResult(taskId, filename) {
    let progress = 20;
    const stageList = ['preprocess', 'ela', 'noise', 'exif', 'copymove', 'cnn'];
    let stageIdx = 0;

    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${taskId}`);
            const data = await response.json();

            if (data.status === 'complete' || data.verdict) {
                clearInterval(interval);
                state.pollIntervals.delete(taskId);

                // Mark all stages done
                stageList.forEach(s => {
                    $(`#stage-${s}`).classList.remove('active');
                    $(`#stage-${s}`).classList.add('done');
                });

                updateProgress(100, `Complete — ${data.verdict}`);
                data.filename = filename;

                // Add to results
                state.allResults.unshift(data);
                renderResults();

                // Switch to results tab after a moment
                setTimeout(() => {
                    $('#nav-results').click();
                    $('#progress-container').style.display = 'none';
                    clearFiles();
                }, 1200);

            } else if (data.status === 'error') {
                clearInterval(interval);
                updateProgress(0, `Error: ${data.error || 'Analysis failed'}`);

            } else {
                // Advance progress simulation
                progress = Math.min(90, progress + 8);
                if (stageIdx < stageList.length) {
                    activateStage(stageList[stageIdx]);
                    if (stageIdx > 0) {
                        $(`#stage-${stageList[stageIdx - 1]}`).classList.remove('active');
                        $(`#stage-${stageList[stageIdx - 1]}`).classList.add('done');
                    }
                    stageIdx++;
                }
                updateProgress(progress, `Running ${stageList[Math.min(stageIdx, stageList.length - 1)].toUpperCase()} analysis...`);
            }
        } catch (err) {
            // Retry silently
        }
    }, 1500);

    state.pollIntervals.set(taskId, interval);
}

function updateProgress(percent, detail) {
    $('#progress-bar').style.width = percent + '%';
    $('#progress-detail').textContent = detail;
}

function activateStage(name) {
    $(`#stage-${name}`).classList.add('active');
}

// ══════════════════════════════════════════════════════════
//  Results Rendering
// ══════════════════════════════════════════════════════════
function renderResults() {
    const container = $('#results-container');

    if (state.allResults.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📋</div>
                <p>No results yet. Upload and analyze an image to see results here.</p>
            </div>`;
        return;
    }

    container.innerHTML = state.allResults.map(r => createResultCard(r)).join('');
}

function createResultCard(result) {
    const verdict = result.verdict || 'UNKNOWN';
    const score = result.overall_score || 0;
    const confidence = result.confidence || 0;
    const analyses = result.analyses || {};
    const elapsed = result.elapsed_seconds || 0;

    // Score circle calculations
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (score / 100) * circumference;
    const scoreColor = score < 40 ? '#00f5d4' : score < 70 ? '#fbbf24' : '#ef4444';

    // Bar data
    const bars = [
        { name: 'ELA', score: analyses.ela?.score || 0 },
        { name: 'Noise', score: analyses.noise?.score || 0 },
        { name: 'EXIF', score: analyses.exif?.score || 0 },
        { name: 'Copy-Move', score: analyses.copy_move?.score || 0 },
        { name: 'CNN AI', score: analyses.cnn?.score || 0 },
    ];

    const barHTML = bars.map(b => {
        const cls = b.score < 40 ? 'bar-low' : b.score < 70 ? 'bar-medium' : 'bar-high';
        return `
            <div class="analysis-bar-row">
                <span class="analysis-bar-label">${b.name}</span>
                <div class="analysis-bar-track">
                    <div class="analysis-bar-fill ${cls}" style="width: ${b.score}%"></div>
                </div>
                <span class="analysis-bar-score" style="color: ${b.score < 40 ? '#00f5d4' : b.score < 70 ? '#fbbf24' : '#ef4444'}">${b.score}</span>
            </div>`;
    }).join('');

    // Detail cards
    const detailCards = [];

    if (analyses.ela) {
        let imgHtml = analyses.ela.image
            ? `<img class="analysis-detail-image" src="data:image/png;base64,${analyses.ela.image}" alt="ELA Heatmap">`
            : '';
        detailCards.push(`
            <div class="analysis-detail-card">
                <div class="analysis-detail-header">
                    <span class="analysis-detail-title">🔥 Error Level Analysis</span>
                    <span class="analysis-detail-score" style="color: ${analyses.ela.score < 40 ? '#00f5d4' : '#fbbf24'}">${analyses.ela.score}/100</span>
                </div>
                <p class="analysis-detail-text">${analyses.ela.details || ''}</p>
                ${imgHtml}
            </div>`);
    }

    if (analyses.noise) {
        let imgHtml = analyses.noise.image
            ? `<img class="analysis-detail-image" src="data:image/png;base64,${analyses.noise.image}" alt="Noise Map">`
            : '';
        detailCards.push(`
            <div class="analysis-detail-card">
                <div class="analysis-detail-header">
                    <span class="analysis-detail-title">📡 Noise Analysis</span>
                    <span class="analysis-detail-score" style="color: ${analyses.noise.score < 40 ? '#00f5d4' : '#fbbf24'}">${analyses.noise.score}/100</span>
                </div>
                <p class="analysis-detail-text">${analyses.noise.details || ''}</p>
                ${imgHtml}
            </div>`);
    }

    if (analyses.exif) {
        const flagsHtml = (analyses.exif.flags || []).map(f =>
            `<span class="exif-flag">${f}</span>`
        ).join('');
        detailCards.push(`
            <div class="analysis-detail-card">
                <div class="analysis-detail-header">
                    <span class="analysis-detail-title">📋 EXIF Metadata</span>
                    <span class="analysis-detail-score" style="color: ${analyses.exif.score < 40 ? '#00f5d4' : '#fbbf24'}">${analyses.exif.score}/100</span>
                </div>
                <p class="analysis-detail-text">${analyses.exif.details || ''}</p>
                <div class="exif-flags">${flagsHtml}</div>
            </div>`);
    }

    if (analyses.copy_move) {
        let imgHtml = analyses.copy_move.image
            ? `<img class="analysis-detail-image" src="data:image/png;base64,${analyses.copy_move.image}" alt="Copy-Move Detection">`
            : '';
        detailCards.push(`
            <div class="analysis-detail-card">
                <div class="analysis-detail-header">
                    <span class="analysis-detail-title">🔍 Copy-Move Detection</span>
                    <span class="analysis-detail-score" style="color: ${analyses.copy_move.score < 40 ? '#00f5d4' : '#fbbf24'}">${analyses.copy_move.score}/100</span>
                </div>
                <p class="analysis-detail-text">${analyses.copy_move.details || ''}</p>
                ${imgHtml}
            </div>`);
    }

    if (analyses.cnn) {
        const cnnConf = analyses.cnn.confidence || {};
        const confHtml = Object.entries(cnnConf).map(([label, pct]) =>
            `<div class="analysis-bar-row">
                <span class="analysis-bar-label" style="width:120px">${label}</span>
                <div class="analysis-bar-track">
                    <div class="analysis-bar-fill bar-low" style="width: ${pct}%"></div>
                </div>
                <span class="analysis-bar-score">${pct}%</span>
            </div>`
        ).join('');
        detailCards.push(`
            <div class="analysis-detail-card">
                <div class="analysis-detail-header">
                    <span class="analysis-detail-title">🤖 CNN Classification</span>
                    <span class="analysis-detail-score">${analyses.cnn.label}</span>
                </div>
                <p class="analysis-detail-text">${analyses.cnn.details || ''}</p>
                ${confHtml}
            </div>`);
    }

    return `
    <div class="result-card">
        <div class="result-header">
            <div>
                <div class="result-filename">${result.filename || result.task_id}</div>
                <div class="result-time">Task: ${result.task_id} • ${elapsed}s</div>
            </div>
            <span class="verdict-badge verdict-${verdict}">${verdict}</span>
        </div>
        <div class="result-body">
            <div class="result-score-row">
                <div class="score-circle">
                    <svg width="120" height="120" viewBox="0 0 120 120">
                        <circle class="score-circle-bg" cx="60" cy="60" r="52"/>
                        <circle class="score-circle-fill" cx="60" cy="60" r="52"
                            stroke="${scoreColor}"
                            stroke-dasharray="${circumference}"
                            stroke-dashoffset="${offset}"/>
                    </svg>
                    <div style="text-align:center">
                        <div class="score-value" style="color:${scoreColor}">${score}</div>
                        <div class="score-label">Risk Score</div>
                    </div>
                </div>
                <div class="score-details">
                    <div class="analysis-bars">${barHTML}</div>
                </div>
            </div>
            <div class="analysis-details">${detailCards.join('')}</div>
        </div>
    </div>`;
}

// ══════════════════════════════════════════════════════════
//  OS Monitor
// ══════════════════════════════════════════════════════════
function startOSMonitor() {
    fetchOSStats();
    state.monitorInterval = setInterval(fetchOSStats, 3000);
}

async function fetchOSStats() {
    try {
        const response = await fetch('/api/os-stats');
        const stats = await response.json();
        updateMonitorUI(stats);
    } catch (err) {
        // Server not ready yet
    }
}

function updateMonitorUI(stats) {
    // ── Threads ──
    const threads = stats.threads || {};
    const stages = threads.stages || {};
    const pre = stages.preprocessing || {};
    const ext = stages.extraction || {};
    const det = stages.detection || {};

    setText('#thread-count', threads.current_thread_count || 0);
    setText('#thread-preprocess', `${pre.active || 0}/${pre.completed || 0}`);
    setText('#thread-extraction', `${ext.active || 0}/${ext.completed || 0}`);
    setText('#thread-detection', `${det.active || 0}/${det.completed || 0}`);
    setText('#thread-total', threads.total_threads_created || 0);

    // ── IPC ──
    const ipc = stats.ipc || {};
    const mqs = ipc.message_queues || {};
    const tq = mqs.task_queue || {};
    const rq = mqs.result_queue || {};
    const shm = ipc.shared_memory || {};
    const pipes = ipc.pipes || {};

    setText('#ipc-badge', `${tq.sent || 0} msgs`);
    setText('#ipc-task-queue', `${tq.sent || 0} / ${tq.received || 0}`);
    setText('#ipc-result-queue', `${rq.sent || 0} / ${rq.received || 0}`);
    setText('#ipc-progress', `${shm.progress || 0}%`);
    setText('#ipc-active', shm.active || 0);
    setText('#ipc-pipes', pipes.pipes_created || 0);

    // ── Sync ──
    const sync = stats.synchronization || {};
    const sem = sync.semaphore || {};
    const locks = sync.locks || {};
    const cv = sync.condition_variable || {};

    setText('#sync-badge', `${sem.total_acquisitions || 0} ops`);
    setText('#sync-semaphore', `${sem.current_acquisitions || 0}/${sem.max_permits || 3}`);
    setText('#sync-acquisitions', sem.total_acquisitions || 0);
    setText('#sync-results-lock', locks.results_lock_acquisitions || 0);
    setText('#sync-contentions', locks.contention_count || 0);
    setText('#sync-batch-signals', cv.signals_sent || 0);

    // ── Memory ──
    const mem = stats.memory || {};
    const cache = mem.cache || {};
    const buf = mem.buffer_pool || {};

    setText('#mem-badge', `${mem.process_memory_mb || 0} MB`);
    setText('#mem-hit-rate', `${cache.hit_rate || 0}%`);
    setText('#mem-cache-entries', `${cache.current_size || 0}/${cache.max_size || 32}`);
    setText('#mem-evictions', cache.evictions || 0);
    setText('#mem-buffer-pool', `${buf.in_use || 0}/${buf.pool_size || 8}`);
    setText('#mem-process', `${mem.process_memory_mb || 0} MB`);

    // ── Deadlock ──
    const dl = stats.deadlock || {};
    setText('#dl-badge', `${dl.deadlocks_detected || 0} detected`);
    setText('#dl-resources', dl.registered_resources || 0);
    setText('#dl-acquisitions', dl.total_acquisitions || 0);
    setText('#dl-detected', dl.deadlocks_detected || 0);
    setText('#dl-recovered', dl.deadlocks_recovered || 0);
    setText('#dl-retries', dl.retries || 0);

    // ── File System ──
    const fs = stats.file_system || {};
    setText('#fs-badge', `${fs.files_processed || 0} files`);
    setText('#fs-processed', fs.files_processed || 0);
    setText('#fs-data', `${fs.total_mb_processed || 0} MB`);
    setText('#fs-logs', fs.logs_written || 0);
    setText('#fs-results', fs.results_saved || 0);
}

function setText(selector, value) {
    const el = $(selector);
    if (el) el.textContent = value;
}
