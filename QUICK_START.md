# Quick Start Guide - Chess SL Training

## 🎯 Get Training in 3 Minutes!

### Method 1: Web Interface (Easiest!) 🌐

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start web server
./start_web_ui.sh

# 3. Open browser
# Visit: http://localhost:5000
```

**In the browser:**
1. Click "New Training"
2. Enter model name: `my_first_model`
3. Enter data path: `data/pgn` (or your path)
4. Click "Check" to validate
5. Adjust settings if desired (or keep defaults)
6. Click "Create Training Job"
7. Wait for dry-run to complete
8. System will auto-start training!
9. Click "Active Training" to monitor in real-time

**That's it!** ✅

### Method 2: Command Line 💻

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare data
mkdir -p data/pgn
# Copy your .pgn files to data/pgn/

# 3. Quick validation
python validate_setup.py

# 4. Dry run (test setup)
python train.py --dry-run

# 5. Full training
python train.py
```

## 🎨 Web Interface Features

### Dashboard Pages:
- **New Training**: Create and configure jobs
- **Active Training**: Monitor running jobs with real-time metrics
- **All Jobs**: View job history
- **System Stats**: CPU/GPU monitoring

### What You See:
- ✅ Real-time progress bars
- ✅ Live loss/accuracy charts
- ✅ Streaming training logs
- ✅ GPU memory usage
- ✅ Current step / total steps
- ✅ Start/stop controls

## 📊 Monitoring Training

### In Web UI:
1. Click "Active Training"
2. See all running models
3. Click any job card for details
4. Watch live metrics update

### Key Metrics:
- **Loss**: Should decrease over time
- **Accuracy**: Should increase over time
- **Step**: Current training step
- **Memory**: GPU memory usage

## ⚙️ Configuration Tips

### Small Model (Fast Training)
```
Dimension: 256
Depth: 6
Heads: 4
Batch: 32
```

### Medium Model (Recommended)
```
Dimension: 512
Depth: 12
Heads: 8
Batch: 64
```

### Large Model (Best Quality)
```
Dimension: 1024
Depth: 24
Heads: 16
Batch: 128
```

### Data Quality:
- **Min ELO 1800**: Amateur to master games
- **Min ELO 2200**: Strong player games
- **Min ELO 2500**: Super-GM level

## 🔥 Hot Tips

1. **Always run dry-run first** - Catches issues early
2. **Start small** - Test with small model first
3. **Monitor memory** - Watch GPU usage
4. **Higher ELO = better** - Quality over quantity
5. **Use web UI** - Easiest to manage multiple jobs

## 🚨 Troubleshooting

### Problem: No PGN files found
**Solution**:
```bash
# Check path
ls data/pgn/*.pgn

# Or use different path in settings
```

### Problem: Out of memory
**Solution**:
- Reduce batch size (try 32 or 16)
- Reduce model size (try 256 dim, 6 depth)
- Close other programs

### Problem: Can't access web UI
**Solution**:
```bash
# Try different port
./start_web_ui.sh --port 8080

# For remote access
./start_web_ui.sh --host 0.0.0.0 --port 5000
```

## 📱 Remote Access

**On Server:**
```bash
./start_web_ui.sh --host 0.0.0.0 --port 5000
```

**From Laptop:**
```
Open browser to: http://SERVER_IP:5000
```

Now you can manage training from anywhere! 🚀

## 🎓 Next Steps

1. ✅ Complete a dry-run
2. ✅ Start a small training job
3. ✅ Monitor in web UI
4. ✅ Try different configurations
5. ✅ Scale up to larger models

## 📚 More Info

- **Detailed Usage**: See USAGE_GUIDE.md
- **Web Interface**: See WEB_UI_GUIDE.md
- **Complete Info**: See README.md

## ⏱️ Expected Timeline

- **Setup**: 5 minutes
- **Dry Run**: 1-2 minutes
- **Full Training**: Hours to days (depends on data and model size)

## 🎉 You're Ready!

Everything is configured and ready to go. Just start the web UI or run training!

```bash
# The easiest way:
./start_web_ui.sh
```

Happy training! 🚀🎯
