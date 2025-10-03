#!/bin/bash
# Start Chess SL Training Web Interface

# Default values
HOST="0.0.0.0"
PORT=5000
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--port PORT] [--debug]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Chess SL Training - Web Interface"
echo "=============================================="
echo "Starting server on http://${HOST}:${PORT}"
echo "=============================================="
echo ""

cd "$(dirname "$0")"

python3 src_frontend/app.py --host "${HOST}" --port "${PORT}" ${DEBUG}
