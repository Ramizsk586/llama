// App state
let config = {};
let activeTab = 'providers';
let tokensChartInstance = null;
let modelChartInstance = null;
let timeRange = 30;

// Selectors
const navItems = document.querySelectorAll('.nav-item');
const tabPanes = document.querySelectorAll('.tab-pane');
const tabTitle = document.getElementById('tab-title');
const tabSubtitle = document.getElementById('tab-subtitle');
const providersList = document.getElementById('providers-list');
const aliasesTableBody = document.getElementById('aliases-table-body');
const toastEl = document.getElementById('toast');

// Modal Selectors
const modalProvider = document.getElementById('modal-provider');
const modalTitle = document.getElementById('modal-title');
const editProviderKey = document.getElementById('edit-provider-key');
const editProviderName = document.getElementById('edit-provider-name');
const editProviderType = document.getElementById('edit-provider-type');
const editProviderUrl = document.getElementById('edit-provider-url');
const editProviderKeyVal = document.getElementById('edit-provider-key-val');
const editProviderModel = document.getElementById('edit-provider-model');

// Load settings from backend
async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        if (!res.ok) throw new Error(await res.text());
        config = await res.json();
        
        // Ensure defaults if empty
        if (!config.providers) config.providers = {};
        if (!config.anthropic_models) config.anthropic_models = {};
        if (!config.pi) config.pi = {};
        if (!config.codex) config.codex = {};
        if (!config.server) config.server = {};
        if (!config.ngrok) config.ngrok = {};

        renderActiveTab();
    } catch (e) {
        showToast(`Failed to load config: ${e.message}`, 'error');
    }
}

// Check Llama Bridge status
async function checkBridgeStatus() {
    const statusText = document.getElementById('bridge-status-text');
    const indicator = document.getElementById('bridge-status-indicator');
    const urlText = document.getElementById('bridge-url');

    try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error();
        const data = await res.json();
        
        urlText.textContent = data.url;
        if (data.running) {
            statusText.textContent = "Bridge Online";
            indicator.className = "status-indicator online";
        } else {
            statusText.textContent = "Bridge Offline";
            indicator.className = "status-indicator offline";
        }
    } catch (e) {
        statusText.textContent = "Bridge Offline";
        indicator.className = "status-indicator offline";
    }
}

// Render active tab content
function renderActiveTab() {
    // Nav Items Active styling
    navItems.forEach(item => {
        if (item.dataset.tab === activeTab) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Tab Pane Active styling
    tabPanes.forEach(pane => {
        if (pane.id === `tab-${activeTab}`) {
            pane.classList.add('active');
        } else {
            pane.classList.remove('active');
        }
    });

    // Update Titles
    if (activeTab === 'providers') {
        tabTitle.textContent = "Manage Providers";
        tabSubtitle.textContent = "Configure connections to local and cloud LLM backends.";
        renderProviders();
    } else if (activeTab === 'aliases') {
        tabTitle.textContent = "Model Aliases";
        tabSubtitle.textContent = "Map Anthropic-style models (Haiku, Sonnet, Opus) to specific providers/models.";
        renderAliases();
    } else if (activeTab === 'integrations') {
        tabTitle.textContent = "Tool Integrations";
        tabSubtitle.textContent = "Manage models and configurations for Pi Coding Agent and Codex.";
        renderIntegrations();
    } else if (activeTab === 'settings') {
        tabTitle.textContent = "Server & Tunneling";
        tabSubtitle.textContent = "Configure FastAPI hosting settings and Ngrok tunnel endpoints.";
        renderSettings();
    } else if (activeTab === 'usage') {
        tabTitle.textContent = "Usage Stats";
        tabSubtitle.textContent = "Monitor model calls, token usage, sessions, and request heatmaps.";
        renderUsage();
    }
}

// ----------------------------------------------------
// Tab: Providers Logic
// ----------------------------------------------------
function renderProviders() {
    providersList.innerHTML = '';
    const providers = config.providers || {};
    
    Object.keys(providers).forEach(key => {
        const p = providers[key];
        const card = document.createElement('div');
        card.className = 'glass-card';
        card.innerHTML = `
            <div class="card-header">
                <div class="provider-info">
                    <h4>${key}</h4>
                    <span class="badge">${p.type}</span>
                </div>
                <div class="card-actions">
                    <button class="btn-icon btn-models" data-key="${key}" title="Manage Models">👁️</button>
                    <button class="btn-icon btn-edit" data-key="${key}" title="Edit Provider">✏️</button>
                    <button class="btn-icon btn-delete" data-key="${key}" title="Delete Provider">🗑️</button>
                </div>
            </div>
            <div class="card-body">
                <div class="card-row">
                    <span>Base URL</span>
                    <span>${p.base_url || 'None'}</span>
                </div>
                <div class="card-row">
                    <span>Default Model</span>
                    <span>${p.default_model || 'None'}</span>
                </div>
                <div class="card-row">
                    <span>Tools Support</span>
                    <span>${p.supports_tools !== false ? '✅ Enabled' : '❌ Disabled'}</span>
                </div>
            </div>
        `;
        providersList.appendChild(card);
    });

    // Attach actions
    document.querySelectorAll('.btn-edit').forEach(b => {
        b.onclick = () => openProviderModal(b.dataset.key);
    });
    document.querySelectorAll('.btn-delete').forEach(b => {
        b.onclick = () => deleteProvider(b.dataset.key);
    });
    document.querySelectorAll('.btn-models').forEach(b => {
        b.onclick = () => openModelsModal(b.dataset.key);
    });
}

function openProviderModal(key = '') {
    const isNew = key === '';
    editProviderKey.value = key;
    editProviderName.disabled = !isNew;
    
    if (isNew) {
        modalTitle.textContent = "Add Provider";
        editProviderName.value = '';
        editProviderType.value = 'openai_compatible';
        editProviderUrl.value = '';
        editProviderKeyVal.value = '';
        editProviderModel.value = '';
    } else {
        modalTitle.textContent = `Edit Provider: ${key}`;
        const p = config.providers[key];
        editProviderName.value = key;
        editProviderType.value = p.type || 'openai_compatible';
        editProviderUrl.value = p.base_url || '';
        editProviderKeyVal.value = p.api_key || '';
        editProviderModel.value = p.default_model || '';
    }
    
    modalProvider.classList.add('active');
}

function deleteProvider(key) {
    if (confirm(`Are you sure you want to delete provider '${key}'?`)) {
        delete config.providers[key];
        renderProviders();
        showToast(`Deleted provider ${key}`, 'success');
    }
}

// ----------------------------------------------------
// Tab: Aliases Logic
// ----------------------------------------------------
function renderAliases() {
    aliasesTableBody.innerHTML = '';
    const aliases = config.anthropic_models || {};
    const providerNames = Object.keys(config.providers || {});

    Object.keys(aliases).forEach(key => {
        const a = aliases[key] || {};
        const tr = document.createElement('tr');
        
        let providerSelectHTML = `<select class="form-control select-alias-provider" data-key="${key}">`;
        providerNames.forEach(pName => {
            providerSelectHTML += `<option value="${pName}" ${pName === a.provider ? 'selected' : ''}>${pName}</option>`;
        });
        providerSelectHTML += `</select>`;

        tr.innerHTML = `
            <td style="font-weight: 600;">${key}</td>
            <td>${providerSelectHTML}</td>
            <td>
                <input type="text" class="form-control input-alias-model" data-key="${key}" value="${a.model || ''}" placeholder="Model ID override">
            </td>
            <td>
                <button class="btn btn-danger btn-sm btn-delete-alias" data-key="${key}">Delete</button>
            </td>
        `;
        aliasesTableBody.appendChild(tr);
    });

    // Attach event listeners
    document.querySelectorAll('.select-alias-provider').forEach(sel => {
        sel.onchange = () => {
            const key = sel.dataset.key;
            config.anthropic_models[key].provider = sel.value;
        };
    });
    
    document.querySelectorAll('.input-alias-model').forEach(inp => {
        inp.oninput = () => {
            const key = inp.dataset.key;
            config.anthropic_models[key].model = inp.value;
        };
    });

    document.querySelectorAll('.btn-delete-alias').forEach(b => {
        b.onclick = () => {
            const key = b.dataset.key;
            delete config.anthropic_models[key];
            renderAliases();
        };
    });
}

function addAlias() {
    const aliasName = prompt("Enter new Anthropic model alias name (e.g. haiku, sonnet, opus, or custom):");
    if (!aliasName) return;
    const name = aliasName.trim().toLowerCase();
    if (!name) return;

    if (config.anthropic_models[name]) {
        showToast("Alias already exists!", "error");
        return;
    }

    const firstProvider = Object.keys(config.providers || {})[0] || "";
    config.anthropic_models[name] = {
        provider: firstProvider,
        model: ""
    };
    renderAliases();
}

// ----------------------------------------------------
// Tab: Integrations Logic
// ----------------------------------------------------
function renderIntegrations() {
    const providerNames = Object.keys(config.providers || {});
    
    // Pi configuration
    const pi = config.pi || {};
    const piProviderSelect = document.getElementById('pi-provider');
    piProviderSelect.innerHTML = '';
    providerNames.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        if (p === pi.provider) opt.selected = true;
        piProviderSelect.appendChild(opt);
    });
    document.getElementById('pi-model').value = pi.model || '';
    document.getElementById('pi-api').value = pi.api || 'openai-completions';
    document.getElementById('pi-web-search').checked = pi.web_search !== false;

    // Codex configuration
    const codex = config.codex || {};
    const codexProviderSelect = document.getElementById('codex-provider');
    codexProviderSelect.innerHTML = '';
    providerNames.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        if (p === codex.provider) opt.selected = true;
        codexProviderSelect.appendChild(opt);
    });
    document.getElementById('codex-model').value = codex.model || '';
}

function collectIntegrations() {
    config.pi = {
        provider: document.getElementById('pi-provider').value,
        model: document.getElementById('pi-model').value.trim(),
        api: document.getElementById('pi-api').value.trim(),
        web_search: document.getElementById('pi-web-search').checked
    };
    
    config.codex = {
        provider: document.getElementById('codex-provider').value,
        model: document.getElementById('codex-model').value.trim()
    };
}

// ----------------------------------------------------
// Tab: Settings Logic
// ----------------------------------------------------
function renderSettings() {
    // Server config
    const server = config.server || {};
    document.getElementById('server-host').value = server.host || '127.0.0.1';
    document.getElementById('server-port').value = server.port || 8089;
    document.getElementById('server-auth-token').value = server.auth_token || 'change-me';
    document.getElementById('server-idle-timeout').value = server.idle_timeout_seconds || 180;

    // Ngrok config
    const ngrok = config.ngrok || {};
    document.getElementById('ngrok-auth-token').value = ngrok.auth_token || '';
    document.getElementById('ngrok-region').value = ngrok.region || '';
}

function collectSettings() {
    config.server = {
        host: document.getElementById('server-host').value.trim(),
        port: parseInt(document.getElementById('server-port').value) || 8089,
        auth_token: document.getElementById('server-auth-token').value.trim(),
        idle_timeout_seconds: parseInt(document.getElementById('server-idle-timeout').value) || 0
    };

    config.ngrok = {
        auth_token: document.getElementById('ngrok-auth-token').value.trim(),
        region: document.getElementById('ngrok-region').value.trim() || null
    };
}

// ----------------------------------------------------
// Saving Logic
// ----------------------------------------------------
async function saveAllConfig() {
    if (activeTab === 'integrations') {
        collectIntegrations();
    } else if (activeTab === 'settings') {
        collectSettings();
    }

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error(await res.text());
        showToast("Configuration saved successfully!", "success");
        // Reload settings
        await fetchConfig();
    } catch (e) {
        showToast(`Failed to save configuration: ${e.message}`, "error");
    }
}

// ----------------------------------------------------
// Provider testing
// ----------------------------------------------------
async function testProviderConnection() {
    const providerCfg = {
        type: editProviderType.value,
        base_url: editProviderUrl.value.trim(),
        api_key: editProviderKeyVal.value.trim(),
        default_model: editProviderModel.value.trim()
    };

    if (!providerCfg.base_url) {
        showToast("Base URL is required to test connection.", "error");
        return;
    }

    const testBtn = document.getElementById('btn-test-provider');
    const oldText = testBtn.textContent;
    testBtn.textContent = "Testing...";
    testBtn.disabled = true;

    try {
        const res = await fetch('/api/test-provider', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(providerCfg)
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        
        if (data.success) {
            showToast("Connection Successful!", "success");
        } else {
            showToast(`Connection Failed: ${data.message}`, "error");
        }
    } catch (e) {
        showToast(`Failed to check provider: ${e.message}`, "error");
    } finally {
        testBtn.textContent = oldText;
        testBtn.disabled = false;
    }
}

// ----------------------------------------------------
// Toast and UI Helpers
// ----------------------------------------------------
function showToast(message, type = 'success') {
    toastEl.textContent = message;
    toastEl.className = `toast show ${type}`;
    setTimeout(() => {
        toastEl.classList.remove('show');
    }, 4000);
}

// Event Listeners
navItems.forEach(item => {
    item.addEventListener('click', () => {
        // Collect current input fields first if leaving inputs
        if (activeTab === 'integrations') collectIntegrations();
        if (activeTab === 'settings') collectSettings();
        
        activeTab = item.dataset.tab;
        renderActiveTab();
    });
});

document.getElementById('btn-add-provider').onclick = () => openProviderModal();
document.getElementById('btn-add-alias').onclick = addAlias;
document.getElementById('btn-save-all').onclick = saveAllConfig;
document.getElementById('btn-test-provider').onclick = testProviderConnection;

// Modal Close logic
const closeModal = () => modalProvider.classList.remove('active');
document.getElementById('btn-close-modal').onclick = closeModal;
document.getElementById('btn-cancel-provider').onclick = closeModal;

// Modal Save logic
document.getElementById('btn-save-provider').onclick = () => {
    const key = editProviderName.value.trim().toLowerCase();
    if (!key) {
        showToast("Provider Name key is required.", "error");
        return;
    }

    config.providers[key] = {
        type: editProviderType.value,
        base_url: editProviderUrl.value.trim(),
        api_key: editProviderKeyVal.value.trim(),
        default_model: editProviderModel.value.trim(),
        supports_tools: true
    };

    closeModal();
    renderProviders();
    showToast(`Saved provider ${key} locally (Click Save Changes to persist).`, "success");
};

// ----------------------------------------------------
// Tab: Usage Stats Logic
// ----------------------------------------------------
async function renderUsage() {
    let stats = {
        token_usage: "0",
        sessions: 0,
        messages: 0,
        active_days: 0,
        streak: 0,
        favorite_model: "None",
        favorite_model_share: 0,
        heatmap: [],
        tokens_per_day: {},
        model_usage: []
    };

    try {
        const res = await fetch('/api/usage');
        if (res.ok) {
            stats = await res.json();
        }
    } catch (e) {
        showToast("Failed to load usage statistics", "error");
    }

    document.getElementById('usage-total-tokens').textContent = stats.token_usage;
    document.getElementById('usage-sessions').textContent = stats.sessions;
    document.getElementById('usage-messages').textContent = stats.messages;
    document.getElementById('usage-active-days').textContent = stats.active_days;
    document.getElementById('usage-streak').textContent = stats.streak;

    const favModelEl = document.getElementById('usage-fav-model');
    favModelEl.textContent = stats.favorite_model;
    favModelEl.title = stats.favorite_model;
    document.getElementById('usage-fav-model-share').textContent = `${stats.favorite_model_share}% share`;

    renderHeatmap(stats.heatmap);
    renderCharts(stats.tokens_per_day, stats.model_usage);
    setupTimeRangeButtons();
}

function renderHeatmap(heatmapData) {
    const container = document.getElementById('heatmap-grid-container');
    container.innerHTML = '';

    const totalCells = 182; // 26 weeks
    const today = new Date();

    const activityMap = new Map();
    if (heatmapData) {
        heatmapData.forEach(d => {
            activityMap.set(d.date, d.count);
        });
    }

    const datesArray = [];
    for (let i = totalCells - 1; i >= 0; i--) {
        const d = new Date();
        d.setDate(today.getDate() - i);
        datesArray.push(d);
    }

    datesArray.forEach(date => {
        const dateStr = date.toISOString().split('T')[0];
        const count = activityMap.get(dateStr) || 0;

        let level = 0;
        if (count > 30) level = 4;
        else if (count > 15) level = 3;
        else if (count > 5) level = 2;
        else if (count > 0) level = 1;

        const cell = document.createElement('div');
        cell.className = `heatmap-cell level-${level}`;

        const options = { month: 'short', day: 'numeric', year: 'numeric' };
        cell.title = `${date.toLocaleDateString('en-US', options)}: ${count} requests`;
        container.appendChild(cell);
    });
}

function renderCharts(tokensPerDay, modelUsage) {
    if (tokensChartInstance) tokensChartInstance.destroy();
    if (modelChartInstance) modelChartInstance.destroy();

    const allDates = Object.keys(tokensPerDay).sort();
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - timeRange);
    const cutoffStr = cutoffDate.toISOString().split('T')[0];

    const filteredDates = allDates.filter(d => d >= cutoffStr);

    if (filteredDates.length === 0) {
        for (let i = timeRange - 1; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            filteredDates.push(d.toISOString().split('T')[0]);
        }
    }

    const uniqueModels = new Set();
    filteredDates.forEach(date => {
        const dateData = tokensPerDay[date] || {};
        Object.keys(dateData).forEach(model => uniqueModels.add(model));
    });

    const colors = [
        '#3b82f6',
        '#10b981',
        '#8b5cf6',
        '#ef4444',
        '#f59e0b',
        '#06b6d4',
        '#ec4899',
    ];

    const datasets = Array.from(uniqueModels).map((model, idx) => {
        const data = filteredDates.map(date => {
            const dateData = tokensPerDay[date] || {};
            return dateData[model] || 0;
        });

        return {
            label: model,
            data: data,
            backgroundColor: colors[idx % colors.length],
            borderRadius: 4,
            borderSkipped: false
        };
    });

    const ctxBar = document.getElementById('tokens-chart').getContext('2d');
    tokensChartInstance = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: filteredDates.map(d => {
                const parts = d.split('-');
                const date = new Date(parts[0], parts[1] - 1, parts[2]);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit' } }
                },
                y: {
                    stacked: true,
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: 'Outfit' },
                        callback: function(val) {
                            if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
                            if (val >= 1000) return (val / 1000).toFixed(0) + 'K';
                            return val;
                        }
                    }
                }
            }
        }
    });

    const donutLabels = modelUsage.map(m => m.model);
    const donutData = modelUsage.map(m => m.tokens);
    const donutColors = donutLabels.map((_, idx) => colors[idx % colors.length]);

    const legendContainer = document.getElementById('donut-legend-container');
    legendContainer.innerHTML = '';
    modelUsage.forEach((m, idx) => {
        const item = document.createElement('div');
        item.className = 'donut-legend-item';
        item.innerHTML = `
            <div class="legend-label-group">
                <span class="legend-color-dot" style="background-color: ${colors[idx % colors.length]}"></span>
                <span class="legend-model-name" title="${m.model}">${m.model}</span>
            </div>
            <span class="legend-percentage">${m.percentage}%</span>
        `;
        legendContainer.appendChild(item);
    });

    const ctxDonut = document.getElementById('model-donut-chart').getContext('2d');
    modelChartInstance = new Chart(ctxDonut, {
        type: 'doughnut',
        data: {
            labels: donutLabels,
            datasets: [{
                data: donutData,
                backgroundColor: donutColors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            let formatted = val;
                            if (val >= 1000000) formatted = (val / 1000000).toFixed(1) + 'M';
                            else if (val >= 1000) formatted = (val / 1000).toFixed(0) + 'K';
                            return ` ${context.label}: ${formatted} tokens`;
                        }
                    }
                }
            }
        }
    });
}

function setupTimeRangeButtons() {
    const rangeButtons = document.querySelectorAll('.btn-time-range');
    rangeButtons.forEach(btn => {
        btn.onclick = () => {
            rangeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            timeRange = parseInt(btn.dataset.range);
            renderUsage();
        };
    });
}


// ----------------------------------------------------
// Models Visibility Logic
// ----------------------------------------------------
const modalModels = document.getElementById('modal-models');
const modelsProviderKey = document.getElementById('models-provider-key');
const modelsCheckboxList = document.getElementById('models-checkbox-list');

async function openModelsModal(providerKey) {
    modelsProviderKey.value = providerKey;
    document.getElementById('models-modal-title').textContent = `Manage Models: ${providerKey}`;
    modelsCheckboxList.innerHTML = '<p style="color: hsl(var(--text-muted)); font-style: italic;">Fetching models from provider...</p>';
    modalModels.classList.add('active');

    try {
        const res = await fetch(`/api/provider/${providerKey}/models`);
        if (!res.ok) throw new Error(await res.text());
        const models = await res.json();
        
        modelsCheckboxList.innerHTML = '';
        if (models.length === 0) {
            modelsCheckboxList.innerHTML = '<p style="color: hsl(var(--red));">No models found or provider connection timed out.</p>';
            return;
        }

        models.forEach(m => {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '10px';
            row.style.padding = '8px 12px';
            row.style.backgroundColor = 'rgba(255,255,255,0.02)';
            row.style.borderRadius = '6px';
            row.style.border = '1px solid rgba(255,255,255,0.05)';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `model-vis-${m.id}`;
            checkbox.className = 'model-vis-chk';
            checkbox.value = m.id;
            checkbox.checked = m.visible;
            checkbox.style.width = '16px';
            checkbox.style.height = '16px';
            checkbox.style.accentColor = 'hsl(var(--primary))';

            const label = document.createElement('label');
            label.htmlFor = `model-vis-${m.id}`;
            label.textContent = m.id;
            label.style.fontSize = '14px';
            label.style.fontWeight = '500';
            label.style.cursor = 'pointer';
            label.style.flexGrow = '1';

            row.appendChild(checkbox);
            row.appendChild(label);
            modelsCheckboxList.appendChild(row);
        });

    } catch (e) {
        modelsCheckboxList.innerHTML = `<p style="color: hsl(var(--red));">Failed to fetch models: ${e.message}</p>`;
    }
}

// Select All / Deselect All
document.getElementById('btn-select-all-models').onclick = () => {
    document.querySelectorAll('.model-vis-chk').forEach(chk => chk.checked = true);
};

document.getElementById('btn-deselect-all-models').onclick = () => {
    document.querySelectorAll('.model-vis-chk').forEach(chk => chk.checked = false);
};

// Cancel
const closeModelsModal = () => modalModels.classList.remove('active');
document.getElementById('btn-close-models-modal').onclick = closeModelsModal;
document.getElementById('btn-cancel-models').onclick = closeModelsModal;

// Save visibility
document.getElementById('btn-save-models').onclick = async () => {
    const key = modelsProviderKey.value;
    const chks = document.querySelectorAll('.model-vis-chk');
    
    const disabledModels = [];
    chks.forEach(chk => {
        if (!chk.checked) {
            disabledModels.push(chk.value);
        }
    });

    const saveBtn = document.getElementById('btn-save-models');
    const oldText = saveBtn.textContent;
    saveBtn.textContent = 'Saving...';
    saveBtn.disabled = true;

    try {
        const res = await fetch(`/api/provider/${key}/models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(disabledModels)
        });
        if (!res.ok) throw new Error(await res.text());
        
        showToast("Model visibility saved successfully!", "success");
        closeModelsModal();
        await fetchConfig();
    } catch (e) {
        showToast(`Failed to save model visibility: ${e.message}`, "error");
    } finally {
        saveBtn.textContent = oldText;
        saveBtn.disabled = false;
    }
};


// Initial Setup
fetchConfig();
checkBridgeStatus();
setInterval(checkBridgeStatus, 5000);
