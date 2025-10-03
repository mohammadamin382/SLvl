## Chess SL Training - Web Interface Guide

### Overview

The web interface provides a powerful, user-friendly dashboard for managing and monitoring chess model training.

### Features

✅ **New Training Creation**
- Configure all model and training parameters via GUI
- Auto-detect GPU and optimize settings
- Validate data paths before starting
- Automatic dry-run before full training

✅ **Real-Time Monitoring**
- Live training progress with progress bars
- Real-time metrics (loss, accuracy, memory)
- Live log streaming
- Interactive charts and visualizations

✅ **Job Management**
- View all training jobs
- Start, stop, and monitor jobs
- Download logs and checkpoints
- Job status tracking

✅ **System Statistics**
- CPU and memory monitoring
- GPU memory usage tracking
- Real-time system stats

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start Web Server

```bash
# Default (0.0.0.0:5000)
./start_web_ui.sh

# Custom port
./start_web_ui.sh --port 8080

# Custom host and port
./start_web_ui.sh --host 0.0.0.0 --port 8000

# Debug mode
./start_web_ui.sh --debug
```

#### 3. Access Dashboard

Open your browser and navigate to:
```
http://SERVER_IP:5000
```

For local access:
```
http://localhost:5000
```

### Using the Web Interface

#### Creating a New Training Job

1. **Navigate to "New Training"**
   - Click "New Training" in the left sidebar

2. **Enter Basic Information**
   - **Model Name**: Choose a unique name (e.g., "chess_model_v1")
   - **Data Path**: Enter path to PGN files (e.g., "/root/datas/pgns/")
   - Click "Check" to validate the path

3. **Configure Model Architecture**
   - **Dimension**: Model size (256/512/768/1024)
     - 256: Small, fast, less powerful
     - 512: Medium, balanced (recommended)
     - 768: Large, slower, more powerful
     - 1024: XLarge, very slow, maximum power
   - **Depth**: Number of transformer layers (6/12/18/24)
   - **Attention Heads**: Number of attention heads (4/8/12/16)
   - **Dropout**: Regularization (0.0-0.5, default 0.1)

4. **Configure Training**
   - **Total Steps**: How many training steps (default 100000)
   - **Batch Size**: Samples per batch (auto-adjusted by GPU)
   - **Learning Rate**: Training speed (default 0.0003)
   - **Warmup Steps**: LR warmup period (default 2000)

5. **Configure Data Filters**
   - **Min ELO**: Minimum player rating (default 2000)
   - **Max ELO**: Maximum player rating (optional)
   - **Min/Max Moves**: Game length filters

6. **Create Job**
   - Click "Create Training Job"
   - System will automatically:
     - Create job directory
     - Generate config files
     - Start dry-run
     - If dry-run succeeds, prompt to start training

#### Monitoring Active Training

1. **Navigate to "Active Training"**
   - Click "Active Training" in the left sidebar

2. **View Running Jobs**
   - See all currently training models
   - Real-time metrics:
     - Current step
     - Loss value
     - Accuracy
     - Memory usage
   - Progress bar shows completion percentage

3. **Job Controls**
   - **Details**: View detailed information
   - **Stop**: Stop the training process

#### Viewing Job Details

Click on any job to see:

1. **Metrics Overview**
   - Current step
   - Progress percentage
   - Latest loss
   - Latest accuracy

2. **Training Charts**
   - Loss over time
   - Accuracy over time
   - Memory usage over time
   - Interactive zoom and pan

3. **Live Training Logs**
   - Real-time log streaming
   - Color-coded messages
   - Auto-scroll to latest
   - Download full logs

4. **Checkpoints**
   - List of saved checkpoints
   - Download checkpoints
   - Checkpoint metadata

#### Managing All Jobs

1. **Navigate to "All Jobs"**
   - See complete job history
   - Filter by status
   - Sort by date

2. **Job Actions**
   - View details
   - Resume stopped jobs
   - Delete old jobs
   - Download logs

#### System Statistics

1. **Navigate to "System"**
   - Real-time CPU usage chart
   - Memory usage monitoring
   - GPU memory visualization
   - Disk usage stats

### API Endpoints

The web interface exposes a REST API:

#### GPU Information
```
GET /api/gpu_info
```
Returns GPU details and capabilities.

#### Check Data Path
```
POST /api/check_data_path
Body: { "path": "/path/to/pgns" }
```
Validates PGN directory.

#### List Jobs
```
GET /api/jobs
```
Returns all training jobs.

#### Get Job Details
```
GET /api/jobs/{job_id}
```
Returns detailed job information.

#### Create Job
```
POST /api/jobs
Body: {
  "model_name": "my_model",
  "data_path": "/path/to/data",
  "settings": {...},
  "data_settings": {...}
}
```
Creates a new training job.

#### Start Dry Run
```
POST /api/jobs/{job_id}/dry_run
```
Starts dry-run validation.

#### Start Training
```
POST /api/jobs/{job_id}/train
```
Starts full training.

#### Stop Training
```
POST /api/jobs/{job_id}/stop
```
Stops training process.

#### Get Logs
```
GET /api/jobs/{job_id}/logs?lines=100
```
Returns recent log lines.

#### Stream Logs (SSE)
```
GET /api/jobs/{job_id}/stream
```
Server-Sent Events stream for real-time logs.

#### System Stats
```
GET /api/system_stats
```
Returns system resource usage.

### Architecture

#### Backend (Flask)
- **app.py**: Main Flask application
- **TrainingJobManager**: Manages job lifecycle
- **REST API**: Exposes functionality
- **SSE Streaming**: Real-time log updates

#### Frontend (HTML/JS)
- **index.html**: Main dashboard
- **app.js**: Client-side logic
- **style.css**: Styling
- **Chart.js**: Data visualization
- **Bootstrap**: UI framework

#### Data Flow
1. User creates job via UI
2. Backend generates config files
3. Backend starts dry-run process
4. If successful, starts training with nohup
5. Backend monitors process
6. Frontend polls/streams updates
7. User sees real-time progress

### File Structure

```
src_frontend/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Main dashboard
└── static/
    ├── css/
    │   └── style.css      # Styles
    └── js/
        └── app.js         # Client logic

jobs/                       # Training jobs (auto-created)
├── job_xxx_model1/
│   ├── settings.yaml
│   ├── data_settings_format.yaml
│   ├── training.log
│   ├── dry_run.log
│   ├── checkpoints/
│   └── logs/
└── job_yyy_model2/
    └── ...

training_jobs.json          # Job metadata
```

### Advanced Usage

#### Programmatic Access

```python
import requests

# Create job
response = requests.post('http://localhost:5000/api/jobs', json={
    'model_name': 'my_model',
    'data_path': '/data/pgns',
    'settings': {...},
    'data_settings': {...}
})

job_id = response.json()['job_id']

# Start training
requests.post(f'http://localhost:5000/api/jobs/{job_id}/train')

# Monitor progress
response = requests.get(f'http://localhost:5000/api/jobs/{job_id}')
job = response.json()
print(f"Step: {job['current_step']}")
```

#### Remote Access

To access from another machine:

1. **Start server on all interfaces:**
   ```bash
   ./start_web_ui.sh --host 0.0.0.0 --port 5000
   ```

2. **Configure firewall (if needed):**
   ```bash
   sudo ufw allow 5000
   ```

3. **Access from remote:**
   ```
   http://SERVER_IP:5000
   ```

#### Production Deployment

For production use:

1. **Use Gunicorn:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 src_frontend.app:app
   ```

2. **Use Nginx reverse proxy:**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /api/jobs/ {
           proxy_pass http://127.0.0.1:5000;
           proxy_buffering off;
           proxy_cache off;
           proxy_set_header Connection '';
           chunked_transfer_encoding off;
       }
   }
   ```

3. **Use systemd service:**
   ```ini
   [Unit]
   Description=Chess SL Training Web UI
   After=network.target

   [Service]
   User=your-user
   WorkingDirectory=/path/to/SLvl
   ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 src_frontend.app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

### Troubleshooting

#### Port Already in Use
```bash
# Find process using port
lsof -i :5000

# Kill process
kill -9 PID

# Or use different port
./start_web_ui.sh --port 8080
```

#### Cannot Access Remotely
- Check firewall settings
- Ensure server binds to 0.0.0.0
- Verify network connectivity

#### Training Not Starting
- Check dry-run logs
- Verify PGN files exist
- Check GPU availability
- Review training.log

#### Logs Not Streaming
- Check browser console for errors
- Verify SSE support in browser
- Check network connection
- Try refreshing page

### Security Considerations

⚠️ **Important**: This web interface has NO authentication by default.

For production:
1. Add authentication (Flask-Login, OAuth)
2. Use HTTPS (SSL/TLS)
3. Implement rate limiting
4. Validate all inputs
5. Use firewall rules
6. Run behind reverse proxy

### Performance Tips

1. **Batch Operations**: Create multiple jobs, let them queue
2. **Monitor Resources**: Use system stats page
3. **Log Rotation**: Clean old logs periodically
4. **Checkpoint Management**: Keep only recent checkpoints

### Support

For issues with the web interface:
1. Check console logs (F12 in browser)
2. Check server logs
3. Verify API endpoints work
4. Review job logs

### Examples

#### Example 1: Quick Start
```bash
# Start server
./start_web_ui.sh

# In browser: http://localhost:5000
# 1. Go to "New Training"
# 2. Name: quick_test
# 3. Path: data/pgn
# 4. Keep defaults
# 5. Click "Create Training Job"
```

#### Example 2: Large Model
```bash
# In web UI:
# - Model Name: large_model_v1
# - Dimension: 1024
# - Depth: 24
# - Heads: 16
# - Total Steps: 500000
# - Min ELO: 2500
```

#### Example 3: Monitor Multiple Jobs
```bash
# Create 3 jobs with different configs
# Navigate to "Active Training"
# See all running side-by-side
# Click any for details
```

Enjoy training! 🚀
