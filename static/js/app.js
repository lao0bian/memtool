document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // 工具函数 & 时钟 (改为北京时间)
    // =========================================================================
    function updateClock() {
        const now = new Date();
        const el = document.getElementById('sys-clock');
        if (el) el.textContent = now.toLocaleTimeString('zh-CN', { hour12: false }) + " 系统时间";
    }
    setInterval(updateClock, 1000);
    updateClock();

    // =========================================================================
    // 主题设置
    // =========================================================================
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const THEME_KEY = 'memtool_theme';
    let chartsInitialized = false;

    const getPreferredTheme = () => localStorage.getItem(THEME_KEY) || 'dark';

    const setTheme = (theme) => {
        document.documentElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        if (themeToggleButton) {
            themeToggleButton.innerHTML = theme === 'dark' ? '<i class="bi bi-sun-fill"></i>' : '<i class="bi bi-moon-stars-fill"></i>';
        }
        if (chartsInitialized) updateAllChartsTheme(theme);
    };

    setTheme(getPreferredTheme());

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            setTheme(getPreferredTheme() === 'dark' ? 'light' : 'dark');
        });
    }

    // =========================================================================
    // 图表配置 (高科技单色系)
    // =========================================================================
    let timelineChart, typeDistChart;

    const THEME_COLORS = {
        dark: {
            primary: '#00f2ff',
            primaryAlpha: 'rgba(0, 242, 255, 0.1)',
            primaryStroke: 'rgba(0, 242, 255, 0.8)',
            text: '#666',
            grid: 'rgba(255, 255, 255, 0.05)',
            surface: '#0a0a0a',
            palette: ['#00f2ff', '#00c3ff', '#0095ff', '#0066ff', '#2d4eff', '#5635ff']
        },
        light: {
            primary: '#0066ff',
            primaryAlpha: 'rgba(0, 102, 255, 0.1)',
            primaryStroke: 'rgba(0, 102, 255, 0.8)',
            text: '#888',
            grid: 'rgba(0, 0, 0, 0.05)',
            surface: '#ffffff',
            palette: ['#0066ff', '#2979ff', '#448aff', '#2962ff', '#0091ea', '#00b0ff']
        }
    };

    function getChartDefaults(theme) {
        const colors = THEME_COLORS[theme];
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: colors.surface,
                    titleColor: colors.primary,
                    bodyColor: '#e0e0e0',
                    borderColor: colors.primaryStroke,
                    borderWidth: 1,
                    cornerRadius: 0,
                    displayColors: false,
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' }
                }
            },
            scales: {
                x: { 
                    grid: { color: colors.grid, drawBorder: false }, 
                    ticks: { color: colors.text, font: { family: 'JetBrains Mono', size: 9 } } 
                },
                y: { 
                    grid: { color: colors.grid, borderDash: [2, 2] }, 
                    ticks: { color: colors.text, font: { family: 'JetBrains Mono', size: 9 } } 
                }
            },
            elements: {
                line: { tension: 0 },
                point: { radius: 2, hoverRadius: 5 }
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
        if (!timelineChart || !typeDistChart) return;
        
        const colors = THEME_COLORS[theme];
        const defaults = getChartDefaults(theme);
        
        timelineChart.options = defaults;
        timelineChart.data.datasets[0].borderColor = colors.primaryStroke;
        timelineChart.data.datasets[0].backgroundColor = colors.primaryAlpha;
        timelineChart.data.datasets[0].pointBackgroundColor = colors.primary;
        timelineChart.update();

        typeDistChart.options = { 
            ...defaults, 
            scales: { x: { display: false }, y: { display: false } },
            plugins: { legend: { display: true, position: 'right', labels: { color: colors.text, font: { size: 10 }, boxWidth: 10, usePointStyle: true } } }
        };
        typeDistChart.data.datasets[0].backgroundColor = colors.palette;
        typeDistChart.data.datasets[0].borderColor = colors.surface;
        typeDistChart.update();
    }

    // =========================================================================
    // API 数据处理
    // =========================================================================
    const API_BASE = '/api';
    async function fetchData(url) {
        try {
            const r = await fetch(url);
            if (!r.ok) throw new Error(r.status);
            return await r.json();
        } catch (e) {
            console.error(e);
            return null;
        }
    }

    const statTotal = document.getElementById('stat-total');
    const statLast = document.getElementById('stat-last');
    const statTypes = document.getElementById('stat-types');
    const memoriesTbody = document.getElementById('memories-tbody');
    
    async function loadStats() {
        const s = await fetchData(`${API_BASE}/stats/summary`);
        if (s && statTotal) {
            statTotal.textContent = s.total_memories.toString().padStart(4, '0');
            statLast.textContent = s.latest_update ? new Date(s.latest_update).toLocaleTimeString('zh-CN', {hour12:false}) : '--:--:--';
        }
        const t = await fetchData(`${API_BASE}/stats/type-dist`);
        if (t && statTypes) statTypes.textContent = Object.keys(t).length;
    }

    async function loadCharts() {
        const tData = await fetchData(`${API_BASE}/stats/timeseries?period=day`);
        if (tData) {
            if (timelineChart) timelineChart.destroy();
            const theme = getPreferredTheme();
            const colors = THEME_COLORS[theme];
            
            timelineChart = renderChart('timeline-chart', 'line', {
                labels: tData.labels,
                datasets: [{
                    label: '新增记忆',
                    data: tData.data,
                    borderColor: colors.primaryStroke,
                    backgroundColor: colors.primaryAlpha,
                    fill: true,
                    pointBackgroundColor: colors.primary
                }]
            });
        }

        const dData = await fetchData(`${API_BASE}/stats/type-dist`);
        if (dData) {
            if (typeDistChart) typeDistChart.destroy();
            const theme = getPreferredTheme();
            const colors = THEME_COLORS[theme];
            
            typeDistChart = renderChart('type-dist-chart', 'doughnut', {
                labels: Object.keys(dData),
                datasets: [{
                    data: Object.values(dData),
                    backgroundColor: colors.palette,
                    borderColor: colors.surface,
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            }, {
                cutout: '60%',
                scales: { x: { display: false }, y: { display: false } },
                plugins: { legend: { display: true, position: 'right', labels: { color: colors.text, font: { size: 10 }, boxWidth: 10, usePointStyle: true } } }
            });
        }
    }

    async function loadTable() {
        const params = new URLSearchParams();
        const inputs = {
            type: document.getElementById('filter-type'),
            key: document.getElementById('filter-key'),
            tag: document.getElementById('filter-tag')
        };
        for (const [k, v] of Object.entries(inputs)) {
            if (v && v.value.trim()) params.append(k, v.value.trim());
        }

        const data = await fetchData(`${API_BASE}/memories?${params}`);
        memoriesTbody.innerHTML = '';

        if (data && data.length) {
            data.forEach(m => {
                const tr = document.createElement('tr');
                const context = m.task_id ? `<span class="text-dim">任务:</span> ${m.task_id.slice(0,8)}` : '<span class="text-dim opacity-25">--</span>';
                
                tr.innerHTML = `
                    <td class="mono-col">0x${m.id.slice(-4)}</td>
                    <td><span class="tech-badge">${m.type.toUpperCase()}</span></td>
                    <td class="text-truncate" style="max-width: 250px;">${m.key}</td>
                    <td class="small">${context}</td>
                    <td class="mono-col small">${new Date(m.updated_at).toLocaleString('zh-CN')}</td>
                    <td style="text-align: right;">
                        <button class="btn btn-tech py-0 px-2 view-btn" data-id="${m.id}" style="font-size: 0.7rem;">查看</button>
                    </td>
                `;
                memoriesTbody.appendChild(tr);
            });
        } else {
            memoriesTbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-dim font-monospace">> 未发现相关记忆数据流</td></tr>`;
        }
    }

    const modalEl = document.getElementById('memoryDetailModal');
    const modalBody = document.getElementById('modal-body-content');
    const bsModal = modalEl ? new bootstrap.Modal(modalEl) : null;

    async function openModal(id) {
        const m = await fetchData(`${API_BASE}/memories/${id}`);
        if (!m) return;
        
        let contentStr = m.content;
        try { contentStr = JSON.stringify(JSON.parse(m.content), null, 2); } catch {}

        modalBody.innerHTML = `
            <div class="json-view">
<span class="text-accent">// 元数据 (META)</span>
唯一标识:   ${m.id}
分类类型:   ${m.type}
标识符键:   ${m.key}
创建时间:   ${m.created_at}
最后更新:   ${m.updated_at}
关联标签:   [${(m.tags||[]).join(', ')}]

<span class="text-accent">// 上下文 (CONTEXT)</span>
任务 ID:    ${m.task_id || '无'}
步骤 ID:    ${m.step_id || '无'}
数据来源:   ${m.source || '未知'}

<span class="text-accent">// 核心数据体 (PAYLOAD)</span>
${contentStr}
            </div>
        `;
        bsModal.show();
    }

    document.getElementById('filter-btn').onclick = loadTable;
    document.getElementById('reset-btn').onclick = () => {
        document.querySelectorAll('.tech-input').forEach(i => i.value = '');
        loadTable();
    };
    memoriesTbody.onclick = (e) => {
        if (e.target.classList.contains('view-btn')) openModal(e.target.dataset.id);
    };

    function init() {
        loadStats();
        loadCharts();
        loadTable();
        chartsInitialized = true;
    }

    if (typeof Chart !== 'undefined') init();
});