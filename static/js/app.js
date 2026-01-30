document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // THEME SETUP & SWITCHER
    // =========================================================================
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const THEME_KEY = 'memtool_theme';

    let chartsInitialized = false; // Flag to check if charts are ready

    const getPreferredTheme = () => localStorage.getItem(THEME_KEY) || 'dark';

    const updateThemeToggleButtonIcon = (theme) => {
        if (themeToggleButton) {
            themeToggleButton.innerHTML = theme === 'dark' ? '<i class="bi bi-sun-fill"></i>' : '<i class="bi bi-moon-stars-fill"></i>';
        }
    };

    const setTheme = (theme) => {
        document.documentElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        updateThemeToggleButtonIcon(theme); // Update icon immediately
        if (chartsInitialized) { // Only update chart colors if charts are already initialized
            updateAllChartsTheme(theme);
        }
    };

    setTheme(getPreferredTheme()); // Set initial theme and icon on page load

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const currentTheme = getPreferredTheme();
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        });
    }

    // =========================================================================
    // CHART SETUP
    // =========================================================================
    let timelineChart, typeDistChart, tagTopChart;

    const THEME_COLORS = {
        dark: {
            cyan: '#00f2ff', cyanAlpha: 'rgba(0, 242, 255, 0.1)',
            purple: '#7000ff', purpleAlpha: 'rgba(112, 0, 255, 0.2)',
            text: '#8b949e', grid: 'rgba(255, 255, 255, 0.05)',
            surface: '#0d1117',
            palette: ['#00f2ff', '#7000ff', '#39ff14', '#ff00c1', '#ffff00', '#ff4d00', '#0070ff']
        },
        light: {
            cyan: '#0d6efd', cyanAlpha: 'rgba(13, 110, 253, 0.1)',
            purple: '#6f42c1', purpleAlpha: 'rgba(111, 66, 193, 0.2)',
            text: '#6c757d', grid: 'rgba(0, 0, 0, 0.05)',
            surface: '#ffffff',
            palette: ['#0d6efd', '#6f42c1', '#198754', '#d63384', '#ffc107', '#fd7e14', '#dc3545']
        }
    };

    function getChartDefaults(theme) {
        const colors = THEME_COLORS[theme];
        return {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false, labels: { color: colors.text, font: { family: 'JetBrains Mono' } } } },
            scales: {
                x: { grid: { color: colors.grid }, ticks: { color: colors.text, font: { family: 'JetBrains Mono', size: 10 } } },
                y: { grid: { color: colors.grid }, ticks: { color: colors.text, font: { family: 'JetBrains Mono', size: 10 } } }
            }
        };
    }
    
    function renderChart(canvasId, chartType, data, options = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        const theme = getPreferredTheme();
        return new Chart(ctx, { type: chartType, data: data, options: { ...getChartDefaults(theme), ...options } });
    }
    
    function updateAllChartsTheme(theme) {
        // Ensure all chart instances are defined before attempting to update
        if (!timelineChart || !typeDistChart || !tagTopChart) {
            console.warn("[WARN] Charts not fully initialized, skipping theme update.");
            return;
        }

        const colors = THEME_COLORS[theme];
        
        // Update timelineChart
        timelineChart.options = { ...getChartDefaults(theme), ...timelineChart.config.options.userDefined };
        timelineChart.data.datasets[0].borderColor = colors.cyan;
        timelineChart.data.datasets[0].backgroundColor = colors.cyanAlpha;
        timelineChart.data.datasets[0].pointBackgroundColor = colors.cyan;
        timelineChart.update();

        // Update typeDistChart
        typeDistChart.options = { ...getChartDefaults(theme), ...typeDistChart.config.options.userDefined };
        typeDistChart.data.datasets[0].backgroundColor = colors.palette;
        typeDistChart.data.datasets[0].borderColor = colors.surface;
        typeDistChart.options.plugins.legend.labels.color = colors.text; // Specific legend color for pie/doughnut
        typeDistChart.update();

        // Update tagTopChart
        tagTopChart.options = { ...getChartDefaults(theme), ...tagTopChart.config.options.userDefined };
        tagTopChart.data.datasets[0].backgroundColor = colors.purpleAlpha;
        tagTopChart.data.datasets[0].borderColor = colors.purple;
        tagTopChart.update();
    }


    // =========================================================================
    // API & DATA HANDLING
    // =========================================================================
    const API_BASE = '/api';
    async function fetchData(url) {
        try {
            console.log(`[SYS] 正在获取数据: ${url}`);
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP_状态_${response.status}`);
            const data = await response.json();
            console.log(`[SYS] 数据已接收:`, data);
            return data;
        } catch (error) {
            console.error(`[ERR] 数据获取失败: ${url}`, error);
            return null;
        }
    }
    
    const modalBodyContent = document.getElementById('modal-body-content');
    const memoriesTbody = document.querySelector('#memories-table tbody');
    let memoryModal;
    const modalEl = document.getElementById('memoryDetailModal');
    if (modalEl) memoryModal = new bootstrap.Modal(modalEl);


    async function loadSummaryStats() {
        const data = await fetchData(`${API_BASE}/stats/summary`);
        if (data) {
            totalMemoriesEl.textContent = data.total_memories.toString().padStart(4, '0');
            latestUpdateEl.textContent = data.latest_update ? new Date(data.latest_update).toLocaleTimeString('zh-CN', { hour12: false }) : '00:00:00';
        }
    }

    async function loadTimelineChart() {
        const data = await fetchData(`${API_BASE}/stats/timeseries?period=day`);
        if (data && data.labels) {
            if (timelineChart) timelineChart.destroy();
            const theme = getPreferredTheme();
            const colors = THEME_COLORS[theme];
            timelineChart = renderChart('timeline-chart', 'line', {
                labels: data.labels,
                datasets: [{ label: '新增记忆', data: data.data, borderColor: colors.cyan, backgroundColor: colors.cyanAlpha, fill: true, tension: 0.4, pointRadius: 4, pointBackgroundColor: colors.cyan, borderWidth: 2 }]
            });
            timelineChart.config.options.userDefined = { /* Store specific options here if needed for re-theming */ };
        }
    }
    async function loadTypeDistChart() {
        const data = await fetchData(`${API_BASE}/stats/type-dist`);
        if (data) {
            if (typeDistChart) typeDistChart.destroy();
            const theme = getPreferredTheme();
            const colors = THEME_COLORS[theme];
            const options = {
                plugins: { legend: { display: true, position: 'bottom', labels: { color: colors.text, boxWidth: 10, font: { size: 10, family: 'JetBrains Mono' } } } },
                scales: { x: { display: false }, y: { display: false } }, cutout: '75%'
            };
            typeDistChart = renderChart('type-dist-chart', 'doughnut', {
                labels: Object.keys(data),
                datasets: [{ data: Object.values(data), backgroundColor: colors.palette, borderColor: colors.surface, borderWidth: 2, hoverOffset: 15 }]
            }, options);
            typeDistChart.config.options.userDefined = options; // Store specific options
        }
    }
    async function loadTopTagsChart() {
        const data = await fetchData(`${API_BASE}/stats/tag-top?n=10`);
        if (data) {
            if (tagTopChart) tagTopChart.destroy();
            const theme = getPreferredTheme();
            const colors = THEME_COLORS[theme];
            const options = { indexAxis: 'y', scales: { x: { grid: { display: false } }, y: { grid: { display: false } } } };
            tagTopChart = renderChart('tag-top-chart', 'bar', {
                labels: data.map(item => item.tag),
                datasets: [{ label: '出现频率', data: data.map(item => item.count), backgroundColor: colors.purpleAlpha, borderColor: colors.purple, borderWidth: 1, borderRadius: 2 }]
            }, options);
            tagTopChart.config.options.userDefined = options; // Store specific options
        }
    }
    
    async function showMemoryDetails(memoryId) {
        const data = await fetchData(`${API_BASE}/memories/${memoryId}`);
        if (data && modalBodyContent) {
            renderMemoryDetails(data);
            if (memoryModal) memoryModal.show();
        }
    }

    function renderMemoryDetails(data) {
        let content;
        try { content = JSON.parse(data.content); } catch (e) { content = data.content; }
        const coreDetails = { 'UID': data.id, '分类': data.type, '标识符': data.key, '更新于': new Date(data.updated_at).toLocaleString('zh-CN', { hour12: false }), '标签': data.tags || [] };
        const contextDetails = { '任务ID': data.task_id, '步骤ID': data.step_id, '来源': data.source, };
        let html = '<h3 class="detail-section-title">核心信息</h3>' + renderJsonAsGrid(coreDetails);
        html += '<h3 class="detail-section-title">上下文</h3>' + renderJsonAsGrid(contextDetails);
        html += '<h3 class="detail-section-title">内容</h3>';
        if (typeof content === 'object' && content !== null) html += renderJsonAsGrid(content); else html += `<pre>${content}</pre>`;
        modalBodyContent.innerHTML = html;
    }
    
    function renderJsonAsGrid(data) {
        let html = '';
        for (const key in data) html += `<div class="detail-grid"><div class="detail-key">${key}</div><div class="detail-value">${renderDetailValue(data[key])}</div></div>`;
        return html;
    }

    function renderDetailValue(value) {
        if (Array.isArray(value)) return value.map(t => `<span class="tag-badge">${t}</span>`).join(' ');
        if (typeof value === 'object' && value !== null) return `<div class="nested-content">${renderJsonAsGrid(value)}</div>`;
        return value;
    }

    async function loadMemories() {
        const params = new URLSearchParams();
        Object.keys(filterInputs).forEach(key => {
            if (!filterInputs[key]) return;
            const value = filterInputs[key].value.trim();
            if (value) params.append(key === 'from' ? 'from' : (key === 'to' ? 'to' : key), value);
        });
        const data = await fetchData(`${API_BASE}/memories?${params.toString()}`);
        memoriesTbody.innerHTML = ''; 
        if (data && data.length > 0) {
            data.forEach(mem => {
                const row = document.createElement('tr');
                const tagsHtml = mem.tags ? `<div class="tags-container">${mem.tags.map(t => `<span class="tag-badge">${t}</span>`).join('')}</div>` : '';
                const taskStep = (mem.task_id || mem.step_id) ? `<div class="text-secondary small">任务: ${mem.task_id || '无'}</div><div class="text-cyan small">步骤: ${mem.step_id || '无'}</div>` : '<span class="text-muted opacity-50">无上下文信息</span>';
                row.innerHTML = `
                    <td class="ps-4 font-monospace text-secondary">0x${mem.id.toString().slice(-4)}</td>
                    <td><span class="text-uppercase fw-bold text-cyan">${mem.type}</span></td>
                    <td class="text-truncate" style="max-width: 250px;" title="${mem.key}">${mem.key}</td>
                    <td>${taskStep}</td>
                    <td class="font-monospace small text-secondary">${new Date(mem.updated_at).toLocaleString('zh-CN', { hour12: false })}</td>
                    <td class="text-end pe-4"><button class="btn btn-cyber-sm view-btn" data-id="${mem.id}"><i class="bi bi-eye me-1"></i>查看详情</button></td>
                `;
                memoriesTbody.appendChild(row);
            });
        } else {
             memoriesTbody.innerHTML = '<tr><td colspan="7" class="text-center py-5 text-muted font-monospace">缓冲区为空: 未找到相关记录</td></tr>';
        }
    }

    const filterBtn = document.getElementById('filter-btn');
    const resetBtn = document.getElementById('reset-btn');
    const filterInputs = { type: document.getElementById('filter-type'), key: document.getElementById('filter-key'), tag: document.getElementById('filter-tag'), task_id: document.getElementById('filter-task-id') };
    if (filterBtn) filterBtn.addEventListener('click', loadMemories);
    if (resetBtn) resetBtn.addEventListener('click', () => {
        Object.values(filterInputs).forEach(input => { if (input) input.value = ''; });
        loadMemories();
    });
    memoriesTbody.addEventListener('click', (e) => {
        const btn = e.target.closest('.view-btn');
        if (btn) showMemoryDetails(btn.getAttribute('data-id'));
    });

    function initializeDashboard() {
        console.log("[SYS] 正在初始化仪表板...");
        loadSummaryStats();
        loadTimelineChart();
        loadTypeDistChart();
        loadTopTagsChart();
        loadMemories();
        chartsInitialized = true; // Mark charts as initialized
        updateAllChartsTheme(getPreferredTheme()); // Apply theme colors to charts now that they exist
    }

    if (typeof Chart === 'undefined') console.error("[CRIT] Chart.js 未找到。可视化功能已禁用。"); else initializeDashboard();
});