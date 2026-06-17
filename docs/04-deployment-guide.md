# 🚀 Deployment Guide · 部署指南

## Prerequisites · 前置条件

| # | Chinese 中文 | English |
|---|-------------|---------|
| 1 | 双机Docker + nvidia-container-toolkit已装好 | Docker + nvidia-container-toolkit on both nodes |
| 2 | 模型权重在相同路径（149GB官方FP8） | Model weights at same path on both nodes (149GB official FP8) |
| 3 | Docker镜像已加载 `dsv4-d568` | Docker image `ghcr.io/bjk110/vllm-spark:dsv4-d568` loaded |
| 4 | RoCE IP已配置 | RoCE IPs configured (10.65.100.1 & 10.65.100.2) |
| 5 | SSH免密互通 | Passwordless SSH between nodes |
| 6 | bjk110 repo已克隆到双机 | [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) cloned on both |

### FlashInfer Patch Setup · 补丁设置

```bash
# On both nodes! 双机都要！
cp patches/flashinfer/flashinfer_arch_parse.patch ~/spark_vllm_docker/patches/flashinfer/
```

The entrypoint.sh in the bjk110 repo already has auto-apply logic for this patch.

bjk110 repo的entrypoint.sh已有此补丁的自动应用逻辑。

## Start Order · 启动顺序

⚠️ **Worker FIRST, then Head!** · **先Worker，再Head！**

### Step 1: Clear Page Cache (Both Nodes)
### 第一步：清Page缓存（双机）

```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
# Repeat on Node B · B端也要做
```

### Step 2: Start Worker (Node B)
### 第二步：启动Worker（B端）

```bash
ssh xiaowan_b "cd ~/spark_vllm_docker && docker compose --profile worker up -d"
sleep 5
```

### Step 3: Start Head (Node A)
### 第三步：启动Head（A端）

```bash
cd ~/spark_vllm_docker && docker compose --profile head up -d
```

### Step 4: Monitor · 第四步：监控

```bash
docker logs -f vllm-spark-head
```

## Expected Timeline · 预计时间线

| Time 时间 | Event 事件 | Chinese 中文 |
|:---------:|-----------|-------------|
| 0:00 | FlashInfer patch applied | 打上架构解析补丁 |
| 0:10 | Loading safetensors: 0% | 开始加载46个分片权重 |
| **2:00** | **Loading safetensors: 100%** | **权重加载完成！74GB** |
| 2:50 | TileLang kernels compiling | 编译sparse MLA kernel |
| 4:00 | KV cache: 44,226 tokens | KV缓存分配完成 |
| 4:10 | CUDA Graph PIECEWISE 4/4 | 混合模式CUDA图捕获 |
| 4:12 | CUDA Graph FULL 3/3 | 纯解码模式CUDA图捕获 |
| **4:15** | **init engine took 161.94 s** | **引擎初始化完成！** |
| **4:16** | **Application startup complete** | **🎉 服务上线！** |

## Verification · 验证

```bash
# Health check · 健康检查
curl http://localhost:8000/health
# → HTTP 200

# List models · 模型列表
curl http://localhost:8000/v1/models
# → {"data":[{"id":"deepseek-v4-flash",...}]}

# Chat · 对话测试
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":10}'
# → "Hello!"

# Resource check · 资源检查
free -h        # Expect ~100GB used / 121GB total
nvidia-smi     # Expect 0% GPU util (idle)
docker ps      # Expect vllm-spark-head + vllm-spark-worker
```
