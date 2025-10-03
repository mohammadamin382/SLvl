// Chess SL Training Dashboard - JavaScript

// Global state
const state = {
    currentPage: 'new-training',
    gpuInfo: null,
    jobs: [],
    activeStreams: {},
    charts: {},
};

// API Base URL
const API_BASE = '';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
function initializeApp() {
    // Load GPU info
    loadGPUInfo();

    // Setup navigation
    setupNavigation();

    // Setup forms
    setupForms();

    // Load jobs
    loadJobs();

    // Setup auto-refresh
    setInterval(loadJobs, 5000); // Refresh every 5 seconds

    // Setup system stats
    initializeSystemStats();
    setInterval(updateSystemStats, 2000);
}

// Load GPU information
async function loadGPUInfo() {
    try {
        const response = await fetch('/api/gpu_info');
        const data = await response.json();
        state.gpuInfo = data;

        const gpuInfoEl = document.getElementById('gpu-info');
        if (data.cuda_available) {
            gpuInfoEl.innerHTML = `
                <span class="gpu-info-badge">
                    <i class="bi bi-gpu-card"></i> ${data.gpu_type || 'Unknown'}
                    (${data.memory_gb?.toFixed(1) || '?'} GB)
                </span>
            `;
        } else {
            gpuInfoEl.innerHTML = '<span class="badge bg-warning">CPU Only</span>';
        }
    } catch (error) {
        console.error('Error loading GPU info:', error);
    }
}

// Setup navigation
function setupNavigation() {
    const navLinks = document.querySelectorAll('[data-page]');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            // Show page
            const pageName = this.dataset.page;
            showPage(pageName);
        });
    });
}

// Show page
function showPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.content-page').forEach(page => {
        page.style.display = 'none';
    });

    // Show selected page
    const page = document.getElementById(pageName);
    if (page) {
        page.style.display = 'block';
        state.currentPage = pageName;

        // Load page-specific data
        if (pageName === 'active-training') {
            loadActiveJobs();
        } else if (pageName === 'all-jobs') {
            loadAllJobs();
        }
    }
}

// Setup forms
function setupForms() {
    // Check data path button
    document.getElementById('check-data-btn').addEventListener('click', checkDataPath);

    // New training form
    document.getElementById('new-training-form').addEventListener('submit', createTrainingJob);
}

// Check data path
async function checkDataPath() {
    const path = document.getElementById('data-path').value;
    const statusEl = document.getElementById('data-path-status');

    if (!path) {
        statusEl.innerHTML = '<div class="alert alert-warning">Please enter a path</div>';
        return;
    }

    statusEl.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Checking...';

    try {
        const response = await fetch('/api/check_data_path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });

        const data = await response.json();

        if (data.valid) {
            statusEl.innerHTML = `
                <div class="data-path-valid">
                    <i class="bi bi-check-circle-fill"></i>
                    Found ${data.total_files} PGN file(s)
                    ${data.total_files > 10 ? `<br><small>Showing first 10:</small>` : ''}
                    <ul class="mt-2 mb-0">
                        ${data.pgn_files.map(f => `<li><small>${f}</small></li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            statusEl.innerHTML = `
                <div class="data-path-invalid">
                    <i class="bi bi-x-circle-fill"></i>
                    ${!data.exists ? 'Path does not exist' :
                      !data.is_directory ? 'Path is not a directory' :
                      'No PGN files found in directory'}
                </div>
            `;
        }
    } catch (error) {
        statusEl.innerHTML = '<div class="alert alert-danger">Error checking path</div>';
        console.error('Error:', error);
    }
}

// Create training job
async function createTrainingJob(e) {
    e.preventDefault();

    const button = e.target.querySelector('button[type="submit"]');
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Creating...';

    // Collect form data
    const settings = {
        model: {
            dim: parseInt(document.getElementById('model-dim').value),
            depth: parseInt(document.getElementById('model-depth').value),
            num_heads: parseInt(document.getElementById('model-heads').value),
            dropout: parseFloat(document.getElementById('model-dropout').value),
        },
        training: {
            total_steps: parseInt(document.getElementById('train-steps').value),
            learning_rate: parseFloat(document.getElementById('train-lr').value),
            warmup_steps: parseInt(document.getElementById('train-warmup').value),
            policy_weight: 1.0,
            value_weight: 0.5,
            use_ema: true,
            checkpoint_interval: 5000,
            log_interval: 100,
            keep_checkpoints: 5,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
        },
        data: {
            pgn_directory: document.getElementById('data-path').value,
            batch_size: parseInt(document.getElementById('train-batch').value),
            min_elo: parseInt(document.getElementById('data-min-elo').value) || null,
            max_elo: parseInt(document.getElementById('data-max-elo').value) || null,
            min_moves: parseInt(document.getElementById('data-min-moves').value),
            max_moves: parseInt(document.getElementById('data-max-moves').value),
            max_games: null,
            result_filter: ['1-0', '0-1', '1/2-1/2'],
        },
        optimizer: {
            learning_rate: parseFloat(document.getElementById('train-lr').value),
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
        },
        gpu: {
            force_gpu_type: null,
            enable_compile: false,
            compile_mode: 'max-autotune',
        },
        advanced: {
            seed: 42,
            num_workers: null,
            pin_memory: true,
            non_blocking: true,
            detect_anomaly: false,
        },
        paths: {
            checkpoint_dir: 'checkpoints',
            log_dir: 'logs',
        },
    };

    const dataSettings = {
        tag_mapping: {
            white_elo: 'WhiteElo',
            black_elo: 'BlackElo',
            result: 'Result',
            time_control: 'TimeControl',
            event: 'Event',
            date: 'Date',
        },
        auto_detection: {
            enabled: true,
            scan_games: 10,
        },
        quality_filters: {
            require_elo: true,
            required_tags: ['WhiteElo', 'BlackElo', 'Result'],
        },
    };

    try {
        const response = await fetch('/api/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: document.getElementById('model-name').value,
                data_path: document.getElementById('data-path').value,
                settings: settings,
                data_settings: dataSettings,
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Start dry run automatically
            await startDryRun(data.job_id);

            // Show success message
            alert(`Job created successfully! Starting dry run...\nJob ID: ${data.job_id}`);

            // Reset form
            e.target.reset();

            // Navigate to active jobs
            showPage('active-training');
            document.querySelector('[data-page="active-training"]').click();
        } else {
            alert('Error creating job: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error creating job: ' + error.message);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

// Start dry run
async function startDryRun(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/dry_run`, {
            method: 'POST'
        });

        if (!response.ok) {
            console.error('Failed to start dry run');
        }
    } catch (error) {
        console.error('Error starting dry run:', error);
    }
}

// Load all jobs
async function loadJobs() {
    try {
        const response = await fetch('/api/jobs');
        const jobs = await response.json();
        state.jobs = jobs;

        // Update active jobs if on that page
        if (state.currentPage === 'active-training') {
            loadActiveJobs();
        } else if (state.currentPage === 'all-jobs') {
            loadAllJobs();
        }
    } catch (error) {
        console.error('Error loading jobs:', error);
    }
}

// Load active jobs
function loadActiveJobs() {
    const container = document.getElementById('active-jobs-container');
    const activeJobs = state.jobs.filter(j => j.status === 'training' || j.status === 'created');

    if (activeJobs.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted">
                <i class="bi bi-hourglass-split" style="font-size: 3rem;"></i>
                <p>No active training jobs</p>
            </div>
        `;
        return;
    }

    container.innerHTML = activeJobs.map(job => createJobCard(job, true)).join('');

    // Setup click handlers
    activeJobs.forEach(job => {
        const card = document.getElementById(`job-${job.id}`);
        if (card) {
            card.addEventListener('click', () => showJobDetail(job.id));
        }
    });
}

// Load all jobs
function loadAllJobs() {
    const container = document.getElementById('all-jobs-container');

    if (state.jobs.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted">
                <i class="bi bi-inbox" style="font-size: 3rem;"></i>
                <p>No training jobs yet</p>
            </div>
        `;
        return;
    }

    container.innerHTML = state.jobs.map(job => createJobCard(job, false)).join('');

    // Setup click handlers
    state.jobs.forEach(job => {
        const card = document.getElementById(`job-${job.id}`);
        if (card) {
            card.addEventListener('click', () => showJobDetail(job.id));
        }
    });
}

// Create job card HTML
function createJobCard(job, detailed = false) {
    const statusClass = {
        'created': 'secondary',
        'training': 'success',
        'stopped': 'warning',
        'completed': 'info',
        'failed': 'danger',
    }[job.status] || 'secondary';

    const progress = job.total_steps > 0 ?
        (job.current_step / job.total_steps * 100).toFixed(1) : 0;

    const metrics = job.metrics || {};
    const latestLoss = metrics.loss?.length > 0 ?
        metrics.loss[metrics.loss.length - 1][1].toFixed(4) : 'N/A';
    const latestAcc = metrics.accuracy?.length > 0 ?
        metrics.accuracy[metrics.accuracy.length - 1][1].toFixed(3) : 'N/A';

    return `
        <div class="card job-card mb-3" id="job-${job.id}">
            <div class="card-body position-relative">
                <span class="badge bg-${statusClass} status-badge">${job.status}</span>

                <h5 class="card-title">
                    <span class="status-indicator ${job.status}"></span>
                    ${job.model_name}
                </h5>
                <p class="text-muted mb-2">
                    <small>
                        <i class="bi bi-folder"></i> ${job.data_path}<br>
                        <i class="bi bi-clock"></i> Created: ${new Date(job.created_at).toLocaleString()}
                    </small>
                </p>

                ${detailed ? `
                    <div class="progress mb-3">
                        <div class="progress-bar progress-bar-striped ${job.status === 'training' ? 'progress-bar-animated' : ''}"
                             role="progressbar"
                             style="width: ${progress}%"
                             aria-valuenow="${progress}"
                             aria-valuemin="0"
                             aria-valuemax="100">
                            ${progress}%
                        </div>
                    </div>

                    <div class="real-time-metrics">
                        <div class="mini-metric">
                            <div class="mini-metric-value">${job.current_step}</div>
                            <div class="mini-metric-label">Step</div>
                        </div>
                        <div class="mini-metric">
                            <div class="mini-metric-value">${latestLoss}</div>
                            <div class="mini-metric-label">Loss</div>
                        </div>
                        <div class="mini-metric">
                            <div class="mini-metric-value">${latestAcc}</div>
                            <div class="mini-metric-label">Accuracy</div>
                        </div>
                        ${job.memory_mb ? `
                            <div class="mini-metric">
                                <div class="mini-metric-value">${(job.memory_mb / 1024).toFixed(1)}</div>
                                <div class="mini-metric-label">GB RAM</div>
                            </div>
                        ` : ''}
                    </div>
                ` : `
                    <div class="row">
                        <div class="col-md-4">
                            <small class="text-muted">Step:</small> ${job.current_step} / ${job.total_steps}
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Loss:</small> ${latestLoss}
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Accuracy:</small> ${latestAcc}
                        </div>
                    </div>
                `}

                <div class="training-controls mt-2">
                    <button class="btn btn-sm btn-primary" onclick="showJobDetail('${job.id}'); event.stopPropagation();">
                        <i class="bi bi-info-circle"></i> Details
                    </button>
                    ${job.status === 'training' ? `
                        <button class="btn btn-sm btn-warning" onclick="stopTraining('${job.id}'); event.stopPropagation();">
                            <i class="bi bi-stop-circle"></i> Stop
                        </button>
                    ` : ''}
                    ${job.status === 'created' && job.dry_run_status === 'success' ? `
                        <button class="btn btn-sm btn-success" onclick="startTraining('${job.id}'); event.stopPropagation();">
                            <i class="bi bi-play-circle"></i> Start Training
                        </button>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
}

// Show job detail in modal
async function showJobDetail(jobId) {
    const modal = new bootstrap.Modal(document.getElementById('jobDetailModal'));
    const titleEl = document.getElementById('jobDetailTitle');
    const contentEl = document.getElementById('jobDetailContent');

    titleEl.textContent = 'Loading...';
    contentEl.innerHTML = '<div class="spinner-container"><div class="spinner-border"></div></div>';
    modal.show();

    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const job = await response.json();

        titleEl.textContent = job.model_name;

        const metrics = job.metrics || {};
        const progress = job.total_steps > 0 ?
            (job.current_step / job.total_steps * 100).toFixed(1) : 0;

        contentEl.innerHTML = `
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${job.current_step}</div>
                        <div class="metric-label">Current Step</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${progress}%</div>
                        <div class="metric-label">Progress</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${metrics.loss?.length > 0 ?
                            metrics.loss[metrics.loss.length - 1][1].toFixed(4) : 'N/A'}</div>
                        <div class="metric-label">Latest Loss</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">${metrics.accuracy?.length > 0 ?
                            metrics.accuracy[metrics.accuracy.length - 1][1].toFixed(3) : 'N/A'}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                </div>
            </div>

            <div class="row mb-4">
                <div class="col-md-6">
                    <h6>Training Metrics</h6>
                    <canvas id="trainingChart"></canvas>
                </div>
                <div class="col-md-6">
                    <h6>Memory Usage</h6>
                    <canvas id="memoryChart"></canvas>
                </div>
            </div>

            <h6>Training Logs</h6>
            <div class="log-container" id="jobLogs">
                <div class="spinner-container"><div class="spinner-border spinner-border-sm"></div></div>
            </div>

            <div class="mt-3">
                <button class="btn btn-primary" onclick="downloadLogs('${jobId}')">
                    <i class="bi bi-download"></i> Download Full Logs
                </button>
            </div>
        `;

        // Load logs
        loadJobLogs(jobId);

        // Create charts
        createJobCharts(jobId, metrics);

        // Start streaming if training
        if (job.status === 'training') {
            startLogStreaming(jobId);
        }

    } catch (error) {
        console.error('Error loading job details:', error);
        contentEl.innerHTML = '<div class="alert alert-danger">Error loading job details</div>';
    }
}

// Load job logs
async function loadJobLogs(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/logs?lines=50`);
        const data = await response.json();

        const logsEl = document.getElementById('jobLogs');
        if (logsEl) {
            logsEl.innerHTML = data.logs.map(line =>
                `<div class="log-line">${escapeHtml(line)}</div>`
            ).join('');
            logsEl.scrollTop = logsEl.scrollHeight;
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    }
}

// Start log streaming
function startLogStreaming(jobId) {
    // Clean up existing stream
    if (state.activeStreams[jobId]) {
        state.activeStreams[jobId].close();
    }

    const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.log) {
            const logsEl = document.getElementById('jobLogs');
            if (logsEl) {
                const lines = data.log.split('\n');
                lines.forEach(line => {
                    if (line.trim()) {
                        const div = document.createElement('div');
                        div.className = 'log-line';
                        div.textContent = line;
                        logsEl.appendChild(div);
                    }
                });
                logsEl.scrollTop = logsEl.scrollHeight;
            }
        }

        if (data.status) {
            // Update job status
            updateJobInList(data.status);
        }
    };

    eventSource.onerror = function() {
        eventSource.close();
        delete state.activeStreams[jobId];
    };

    state.activeStreams[jobId] = eventSource;
}

// Create job charts
function createJobCharts(jobId, metrics) {
    // Training chart
    const trainingCtx = document.getElementById('trainingChart');
    if (trainingCtx && metrics.loss) {
        new Chart(trainingCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Loss',
                        data: metrics.loss,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    },
                    {
                        label: 'Accuracy',
                        data: metrics.accuracy,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Step' } },
                    y: { type: 'linear', title: { display: true, text: 'Loss' } },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Accuracy' }
                    }
                }
            }
        });
    }

    // Memory chart
    const memoryCtx = document.getElementById('memoryChart');
    if (memoryCtx && metrics.memory_gb) {
        new Chart(memoryCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Memory (GB)',
                    data: metrics.memory_gb,
                    borderColor: 'rgb(153, 102, 255)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Step' } },
                    y: { type: 'linear', title: { display: true, text: 'Memory (GB)' } }
                }
            }
        });
    }
}

// Start training
async function startTraining(jobId) {
    if (!confirm('Start full training for this job?')) {
        return;
    }

    try {
        const response = await fetch(`/api/jobs/${jobId}/train`, {
            method: 'POST'
        });

        if (response.ok) {
            alert('Training started!');
            loadJobs();
        } else {
            const data = await response.json();
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error starting training');
    }
}

// Stop training
async function stopTraining(jobId) {
    if (!confirm('Stop training for this job?')) {
        return;
    }

    try {
        const response = await fetch(`/api/jobs/${jobId}/stop`, {
            method: 'POST'
        });

        if (response.ok) {
            alert('Training stopped!');
            loadJobs();
        } else {
            const data = await response.json();
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error stopping training');
    }
}

// Initialize system stats
function initializeSystemStats() {
    const cpuCtx = document.getElementById('cpu-chart');
    const gpuCtx = document.getElementById('gpu-chart');

    if (cpuCtx) {
        state.charts.cpu = new Chart(cpuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { min: 0, max: 100 }
                }
            }
        });
    }

    if (gpuCtx) {
        state.charts.gpu = new Chart(gpuCtx, {
            type: 'bar',
            data: {
                labels: ['GPU Memory'],
                datasets: [
                    {
                        label: 'Allocated',
                        data: [0],
                        backgroundColor: 'rgba(75, 192, 192, 0.5)'
                    },
                    {
                        label: 'Reserved',
                        data: [0],
                        backgroundColor: 'rgba(255, 206, 86, 0.5)'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { min: 0 }
                }
            }
        });
    }
}

// Update system stats
async function updateSystemStats() {
    if (state.currentPage !== 'system') return;

    try {
        const response = await fetch('/api/system_stats');
        const stats = await response.json();

        // Update CPU chart
        if (state.charts.cpu) {
            const chart = state.charts.cpu;
            const now = new Date().toLocaleTimeString();

            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(stats.cpu_percent);
            chart.data.datasets[1].data.push(stats.memory_percent);

            // Keep only last 20 points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }

            chart.update('none');
        }

        // Update GPU chart
        if (state.charts.gpu && stats.gpu_memory_total) {
            const chart = state.charts.gpu;
            chart.data.datasets[0].data = [stats.gpu_memory_allocated];
            chart.data.datasets[1].data = [stats.gpu_memory_reserved];
            chart.options.scales.y.max = stats.gpu_memory_total;
            chart.update('none');
        }
    } catch (error) {
        console.error('Error updating system stats:', error);
    }
}

// Helper functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateJobInList(job) {
    const index = state.jobs.findIndex(j => j.id === job.id);
    if (index >= 0) {
        state.jobs[index] = job;
    }
}

function downloadLogs(jobId) {
    window.open(`/api/jobs/${jobId}/logs?lines=10000`, '_blank');
}
