#!/usr/bin/env python3
"""
Chess SL Training - Web Interface
Advanced web panel for managing and monitoring chess model training
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch

from gpus import detect_gpu, get_gpu_optimizer

app = Flask(__name__)
CORS(app)

# Configuration
TRAINING_JOBS_FILE = 'training_jobs.json'
TRAINING_LOGS_DIR = Path('training_logs')
TRAINING_LOGS_DIR.mkdir(exist_ok=True)


class TrainingJobManager:
    """Manage training jobs"""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.load_jobs()

    def load_jobs(self):
        """Load jobs from file"""
        if Path(TRAINING_JOBS_FILE).exists():
            with open(TRAINING_JOBS_FILE, 'r') as f:
                self.jobs = json.load(f)

    def save_jobs(self):
        """Save jobs to file"""
        with open(TRAINING_JOBS_FILE, 'w') as f:
            json.dump(self.jobs, f, indent=2)

    def create_job(
        self,
        model_name: str,
        data_path: str,
        settings: Dict,
        data_settings: Dict
    ) -> str:
        """Create a new training job"""
        job_id = f"job_{int(time.time())}_{model_name.replace(' ', '_')}"

        # Create job directory
        job_dir = Path('jobs') / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save configurations
        settings_path = job_dir / 'settings.yaml'
        data_settings_path = job_dir / 'data_settings_format.yaml'

        with open(settings_path, 'w') as f:
            yaml.dump(settings, f)

        with open(data_settings_path, 'w') as f:
            yaml.dump(data_settings, f)

        # Create job entry
        job = {
            'id': job_id,
            'model_name': model_name,
            'data_path': data_path,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'finished_at': None,
            'job_dir': str(job_dir),
            'settings_path': str(settings_path),
            'data_settings_path': str(data_settings_path),
            'log_file': str(job_dir / 'training.log'),
            'pid': None,
            'dry_run_status': None,
            'current_step': 0,
            'total_steps': settings.get('training', {}).get('total_steps', 100000),
            'metrics': {},
        }

        self.jobs[job_id] = job
        self.save_jobs()

        return job_id

    def start_dry_run(self, job_id: str) -> bool:
        """Start dry run for a job"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        job_dir = Path(job['job_dir'])

        # Start dry run
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / 'train.py'),
            '--config', job['settings_path'],
            '--dry-run'
        ]

        log_file = open(job_dir / 'dry_run.log', 'w')

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=job_dir
        )

        job['dry_run_pid'] = process.pid
        job['dry_run_status'] = 'running'
        self.save_jobs()

        # Wait for dry run to complete
        def wait_for_dry_run():
            process.wait()
            log_file.close()

            if process.returncode == 0:
                self.jobs[job_id]['dry_run_status'] = 'success'
            else:
                self.jobs[job_id]['dry_run_status'] = 'failed'

            self.save_jobs()

        thread = threading.Thread(target=wait_for_dry_run)
        thread.daemon = True
        thread.start()

        return True

    def start_training(self, job_id: str) -> bool:
        """Start full training for a job"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        job_dir = Path(job['job_dir'])

        # Create training script with nohup
        cmd = [
            'nohup',
            sys.executable,
            str(Path(__file__).parent.parent / 'train.py'),
            '--config', job['settings_path'],
            '&'
        ]

        # Use subprocess to start in background
        log_file = open(job_dir / 'training.log', 'w')

        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent.parent / 'train.py'),
             '--config', job['settings_path']],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=job_dir,
            start_new_session=True  # Detach from parent
        )

        job['pid'] = process.pid
        job['status'] = 'training'
        job['started_at'] = datetime.now().isoformat()
        self.save_jobs()

        return True

    def stop_training(self, job_id: str) -> bool:
        """Stop training for a job"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        if job.get('pid'):
            try:
                os.kill(job['pid'], signal.SIGTERM)
                job['status'] = 'stopped'
                job['finished_at'] = datetime.now().isoformat()
                self.save_jobs()
                return True
            except ProcessLookupError:
                job['status'] = 'stopped'
                self.save_jobs()
                return True

        return False

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a job"""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id].copy()

        # Check if process is still running
        if job.get('pid'):
            try:
                process = psutil.Process(job['pid'])
                if process.is_running():
                    # Get CPU and memory usage
                    job['cpu_percent'] = process.cpu_percent(interval=0.1)
                    job['memory_mb'] = process.memory_info().rss / (1024 * 1024)
                else:
                    job['status'] = 'completed'
                    job['finished_at'] = datetime.now().isoformat()
                    self.save_jobs()
            except psutil.NoSuchProcess:
                job['status'] = 'completed'
                job['finished_at'] = datetime.now().isoformat()
                self.save_jobs()

        # Parse log file for metrics
        log_path = Path(job['log_file'])
        if log_path.exists():
            job['metrics'] = self.parse_log_metrics(log_path)

        return job

    def parse_log_metrics(self, log_path: Path) -> Dict:
        """Parse training metrics from log file"""
        metrics = {
            'current_step': 0,
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'accuracy': [],
            'learning_rate': [],
            'memory_gb': [],
        }

        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Parse step information
                if '[Step ' in line:
                    try:
                        # Extract step number
                        step_str = line.split('[Step ')[1].split(']')[0]
                        step = int(step_str)
                        metrics['current_step'] = max(metrics['current_step'], step)

                        # Extract metrics
                        if 'Loss:' in line:
                            loss = float(line.split('Loss: ')[1].split(',')[0])
                            metrics['loss'].append([step, loss])

                        if 'Policy:' in line:
                            policy = float(line.split('Policy: ')[1].split(',')[0])
                            metrics['policy_loss'].append([step, policy])

                        if 'Value:' in line:
                            value = float(line.split('Value: ')[1].split(',')[0])
                            metrics['value_loss'].append([step, value])

                        if 'Accuracy:' in line:
                            acc = float(line.split('Accuracy: ')[1].split()[0])
                            metrics['accuracy'].append([step, acc])

                        if 'Allocated:' in line:
                            mem_str = line.split('Allocated: ')[1].split('GB')[0]
                            mem = float(mem_str)
                            metrics['memory_gb'].append([step, mem])

                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"Error parsing log: {e}")

        return metrics

    def get_job_log(self, job_id: str, lines: int = 100) -> List[str]:
        """Get recent log lines for a job"""
        if job_id not in self.jobs:
            return []

        log_path = Path(self.jobs[job_id]['log_file'])

        if not log_path.exists():
            return []

        try:
            with open(log_path, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except Exception:
            return []

    def list_jobs(self) -> List[Dict]:
        """List all jobs"""
        jobs = []
        for job_id in self.jobs:
            status = self.get_job_status(job_id)
            if status:
                jobs.append(status)

        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)


# Global job manager
job_manager = TrainingJobManager()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/gpu_info')
def gpu_info():
    """Get GPU information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_type': None,
        'memory_gb': None,
        'compute_capability': None,
    }

    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_type'] = detect_gpu()

        gpu_opt = get_gpu_optimizer()
        info['memory_gb'] = gpu_opt.memory_gb
        info['compute_capability'] = gpu_opt.compute_capability

    return jsonify(info)


@app.route('/api/check_data_path', methods=['POST'])
def check_data_path():
    """Check if data path is valid"""
    data = request.json
    path = data.get('path', '')

    result = {
        'valid': False,
        'exists': False,
        'is_directory': False,
        'pgn_files': [],
        'total_files': 0,
    }

    if path:
        p = Path(path)
        result['exists'] = p.exists()
        result['is_directory'] = p.is_dir()

        if p.exists() and p.is_dir():
            pgn_files = list(p.glob('*.pgn'))
            result['pgn_files'] = [str(f) for f in pgn_files[:10]]  # First 10
            result['total_files'] = len(pgn_files)
            result['valid'] = len(pgn_files) > 0

    return jsonify(result)


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all training jobs"""
    jobs = job_manager.list_jobs()
    return jsonify(jobs)


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job details"""
    job = job_manager.get_job_status(job_id)
    if job:
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404


@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new training job"""
    data = request.json

    model_name = data.get('model_name')
    data_path = data.get('data_path')
    settings = data.get('settings')
    data_settings = data.get('data_settings')

    if not model_name or not data_path:
        return jsonify({'error': 'Missing required fields'}), 400

    # Update data path in settings
    if 'data' not in settings:
        settings['data'] = {}
    settings['data']['pgn_directory'] = data_path

    job_id = job_manager.create_job(model_name, data_path, settings, data_settings)

    return jsonify({'job_id': job_id, 'status': 'created'})


@app.route('/api/jobs/<job_id>/dry_run', methods=['POST'])
def start_dry_run(job_id):
    """Start dry run for a job"""
    success = job_manager.start_dry_run(job_id)

    if success:
        return jsonify({'status': 'started'})
    return jsonify({'error': 'Failed to start dry run'}), 400


@app.route('/api/jobs/<job_id>/train', methods=['POST'])
def start_training(job_id):
    """Start training for a job"""
    success = job_manager.start_training(job_id)

    if success:
        return jsonify({'status': 'started'})
    return jsonify({'error': 'Failed to start training'}), 400


@app.route('/api/jobs/<job_id>/stop', methods=['POST'])
def stop_training(job_id):
    """Stop training for a job"""
    success = job_manager.stop_training(job_id)

    if success:
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Failed to stop training'}), 400


@app.route('/api/jobs/<job_id>/logs')
def get_job_logs(job_id):
    """Get job logs"""
    lines = request.args.get('lines', 100, type=int)
    logs = job_manager.get_job_log(job_id, lines)

    return jsonify({'logs': logs})


@app.route('/api/jobs/<job_id>/stream')
def stream_job_logs(job_id):
    """Stream job logs in real-time (SSE)"""

    def generate():
        job = job_manager.jobs.get(job_id)
        if not job:
            return

        log_path = Path(job['log_file'])
        last_size = 0

        while True:
            try:
                if log_path.exists():
                    current_size = log_path.stat().st_size

                    if current_size > last_size:
                        with open(log_path, 'r') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            last_size = current_size

                            if new_content:
                                yield f"data: {json.dumps({'log': new_content})}\n\n"

                # Also send status update
                status = job_manager.get_job_status(job_id)
                if status:
                    yield f"data: {json.dumps({'status': status})}\n\n"

                time.sleep(1)

            except Exception as e:
                print(f"Error streaming: {e}")
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/system_stats')
def system_stats():
    """Get system statistics"""
    stats = {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
    }

    if torch.cuda.is_available():
        stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
        stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
        stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return jsonify(stats)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Chess Training Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print("=" * 60)
    print("Chess Supervised Learning - Web Interface")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
