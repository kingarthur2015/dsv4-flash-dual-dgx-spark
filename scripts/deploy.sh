#!/bin/bash
# =============================================================================
# 🦐 DSV4 Flash Dual DGX Spark — Deployment Script
# 双机DSV4 Flash部署脚本
# =============================================================================
# Bilingual / 中英双语
# 
# Usage · 用法:
#   ./deploy.sh              # Deploy both nodes
#   ./deploy.sh --worker     # Worker only (Node B)
#   ./deploy.sh --head       # Head only (Node A)
#   ./deploy.sh --stop       # Stop both
#   ./deploy.sh --status     # Check status
# =============================================================================

set -euo pipefail

# ---- Configuration - change these! · 配置——改成你的！ ----
HEAD_SSH="xiaowan"          # SSH hostname for Head (Node A)
WORKER_SSH="xiaowan_b"      # SSH hostname for Worker (Node B)
WORKDIR="~/spark_vllm_docker"  # Working directory (bjk110 repo)

# ---- Colors · 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Help · 帮助 ----
if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
    echo "🦐 DSV4 Flash Dual DGX Spark Deploy Script"
    echo ""
    echo "Usage · 用法:"
    echo "  ./deploy.sh              Full deploy · 完整部署 (Worker → Head)"
    echo "  ./deploy.sh --worker     Deploy Worker only · 只启动Worker"
    echo "  ./deploy.sh --head       Deploy Head only · 只启动Head"
    echo "  ./deploy.sh --stop       Stop all · 停止所有"
    echo "  ./deploy.sh --status     Check status · 检查状态"
    exit 0
fi

# ---- Functions · 函数 ----
deploy_worker() {
    info "Clearing page cache on Worker (B端清理缓存)..."
    ssh "$WORKER_SSH" "echo 3 | sudo tee /proc/sys/vm/drop_caches" 2>/dev/null || warn "Could not clear cache on Worker"
    
    info "Starting Worker container on ${WORKER_SSH} (启动Worker容器)..."
    ssh "$WORKER_SSH" "cd ${WORKDIR} && docker compose --profile worker up -d" 2>&1
    
    info "Worker started (Worker已启动). Waiting 5s for initialization..."
    sleep 5
}

deploy_head() {
    info "Clearing page cache on Head (A端清理缓存)..."
    echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || warn "Could not clear cache on Head"
    
    info "Starting Head container (启动Head容器)..."
    cd "$WORKDIR" && docker compose --profile head up -d
    
    info "Head started (Head已启动). Monitoring logs... (监控日志...)"
    echo ""
    info "Key milestones (关键里程碑):"
    echo "  [0:10] Loading safetensors: 0% | 0/46"
    echo "  [2:00] Loading safetensors: 100% | 46/46  ← 权重加载完成"
    echo "  [4:00] init engine took 161s"
    echo "  [4:16] Application startup complete  ← 🎉 成功"
    echo ""
    docker logs -f vllm-spark-head 2>/dev/null || docker logs -f vllm-spark-head
}

stop_all() {
    info "Stopping Head (停止Head)..."
    cd "$WORKDIR" && docker compose --profile head down 2>/dev/null || true
    
    info "Stopping Worker (停止Worker)..."
    ssh "$WORKER_SSH" "cd ${WORKDIR} && docker compose --profile worker down" 2>/dev/null || true
    
    info "All stopped (全部已停止)."
}

check_status() {
    echo "=== Head (Node A) ==="
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep vllm || echo "Not running (未运行)"
    
    echo ""
    echo "=== Worker (Node B) ==="
    ssh "$WORKER_SSH" "docker ps --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null | grep vllm || echo "Not running (未运行)"
    
    echo ""
    echo "=== GPU (Node A) ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
    
    echo ""
    echo "=== Inference Test (推理测试) ==="
    curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/health 2>/dev/null || echo "Service not reachable (服务不可达)"
}

# ---- Main · 主流程 ----
case "${1:-}" in
    --worker)
        deploy_worker
        ;;
    --head)
        deploy_head
        ;;
    --stop)
        stop_all
        ;;
    --status)
        check_status
        ;;
    *)
        info "Full deployment (完整部署): Worker → Head"
        deploy_worker
        deploy_head
        ;;
esac

echo ""
info "🦐 Done! (完成!)"
