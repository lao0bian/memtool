
document.addEventListener('DOMContentLoaded', () => {
    // Chart instances
    let timelineChart, typeDistChart, tagTopChart;

    // DOM Elements
    const totalMemoriesEl = document.getElementById('total-memories');
    const latestUpdateEl = document.getElementById('latest-update');
    const memoriesTbody = document.querySelector('#memories-table tbody');
    const modalBackdrop = document.getElementById('modal-backdrop');
    const modalBody = document.getElementById('modal-body');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // Filter Elements
    const filterBtn = document.getElementById('filter-btn');
    const resetBtn = document.getElementById('reset-btn');
    const filterInputs = {
        type: document.getElementById('filter-type'),
        key: document.getElementById('filter-key'),
        tag: document.getElementById('filter-tag'),
        task_id: document.getElementById('filter-task-id'),
        from: document.getElementById('filter-from'),
        to: document.getElementById('filter-to'),
    };

    const API_BASE = '/api';

    /**
     * Fetches data from a URL and returns it as JSON.
     * @param {string} url - The URL to fetch.
     */
    async function fetchData(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Failed to fetch ${url}:`, error);
            return null;
        }
    }

    /**
     * Renders a chart using Chart.js.
     * @param {HTMLCanvasElement} canvasId - The canvas element ID.
     * @param {string} chartType - The type of chart (e.g., 'bar', 'line', 'pie').
     * @param {object} data - The data object for the chart.
     * @param {object} options - Chart.js options.
     * @returns {Chart} - The new Chart instance.
     */
    function renderChart(canvasId, chartType, data, options = {}) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: chartType,
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options,
            },
        });
    }

    /**
     * Loads the high-level summary stats.
     */
    async function loadSummaryStats() {
        const data = await fetchData(`${API_BASE}/stats/summary`);
        if (data) {
            totalMemoriesEl.textContent = data.total_memories;
            latestUpdateEl.textContent = data.latest_update ? new Date(data.latest_update).toLocaleString() : 'N/A';
        }
    }

    /**
     * Loads and renders the timeline chart.
     */
    async function loadTimelineChart() {
        const data = await fetchData(`${API_BASE}/stats/timeseries?period=day`);
        if (data) {
            if (timelineChart) timelineChart.destroy();
            timelineChart = renderChart('timeline-chart', 'line', {
                labels: data.labels,
                datasets: [{
                    label: 'Memories per Day',
                    data: data.data,
                    borderColor: 'rgba(0, 123, 255, 0.8)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                }],
            });
        }
    }
    
    /**
     * Loads and renders the type distribution chart.
     */
    async function loadTypeDistChart() {
        const data = await fetchData(`${API_BASE}/stats/type-dist`);
        if (data) {
            if (typeDistChart) typeDistChart.destroy();
            typeDistChart = renderChart('type-dist-chart', 'pie', {
                labels: Object.keys(data),
                datasets: [{
                    label: 'Memory Types',
                    data: Object.values(data),
                     backgroundColor: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6c757d'],
                }],
            });
        }
    }

    /**
     * Loads and renders the top tags chart.
     */
    async function loadTopTagsChart() {
        const data = await fetchData(`${API_BASE}/stats/tag-top?n=10`);
        if (data) {
            if (tagTopChart) tagTopChart.destroy();
            tagTopChart = renderChart('tag-top-chart', 'bar', {
                labels: data.map(item => item.tag),
                datasets: [{
                    label: 'Tag Frequency',
                    data: data.map(item => item.count),
                    backgroundColor: 'rgba(23, 162, 184, 0.7)',
                    borderColor: 'rgba(23, 162, 184, 1)',
                    borderWidth: 1
                }],
            }, { indexAxis: 'y' });
        }
    }
    
    /**
     * Loads memories from the API and populates the table.
     */
    async function loadMemories() {
        const params = new URLSearchParams();
        Object.keys(filterInputs).forEach(key => {
            const value = filterInputs[key].value.trim();
            if (value) {
                // The API expects 'from' and 'to' aliases for dates
                const paramKey = key === 'from' ? 'from' : (key === 'to' ? 'to' : key);
                params.append(paramKey, value);
            }
        });

        const data = await fetchData(`${API_BASE}/memories?${params.toString()}`);
        memoriesTbody.innerHTML = ''; // Clear existing rows
        if (data && data.length > 0) {
            data.forEach(mem => {
                const row = document.createElement('tr');
                const tagsHtml = mem.tags ? `<ul class="tags-list">${mem.tags.map(t => `<li>${t}</li>`).join('')}</ul>` : '';
                row.innerHTML = `
                    <td>${mem.id}</td>
                    <td>${mem.type}</td>
                    <td>${mem.key}</td>
                    <td>${tagsHtml}</td>
                    <td>${mem.task_id || ''} / ${mem.step_id || ''}</td>
                    <td>${new Date(mem.updated_at).toLocaleString()}</td>
                    <td><button class="action-btn view-btn" data-id="${mem.id}">View</button></td>
                `;
                memoriesTbody.appendChild(row);
            });
        } else {
             memoriesTbody.innerHTML = '<tr><td colspan="7">No memories found.</td></tr>';
        }
    }
    
    /**
     * Shows the modal with details for a specific memory.
     * @param {number} memoryId - The ID of the memory to show.
     */
    async function showMemoryDetails(memoryId) {
        const data = await fetchData(`${API_BASE}/memories/${memoryId}`);
        if (data) {
            modalBody.textContent = JSON.stringify(data, null, 2);
            modalBackdrop.classList.remove('hidden');
        }
    }

    // Event Listeners
    filterBtn.addEventListener('click', loadMemories);
    
    resetBtn.addEventListener('click', () => {
        Object.values(filterInputs).forEach(input => input.value = '');
        loadMemories();
    });

    memoriesTbody.addEventListener('click', (e) => {
        if (e.target.classList.contains('view-btn')) {
            const id = e.target.getAttribute('data-id');
            showMemoryDetails(id);
        }
    });

    modalCloseBtn.addEventListener('click', () => modalBackdrop.classList.add('hidden'));
    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            modalBackdrop.classList.add('hidden');
        }
    });


    /**
     * Initial data load
     */
    function initializeDashboard() {
        loadSummaryStats();
        loadTimelineChart();
        loadTypeDistChart();
        loadTopTagsChart();
        loadMemories();
    }

    initializeDashboard();
});
