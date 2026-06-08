# 🔧 Troubleshooting · 故障排查

## Error Quick Reference · 错误速查

| Error 错误 | Root Cause 根因 | Fix 修复 | Discovered 发现于 |
|-----------|----------------|---------|:--------------:|
| `ValueError: not enough values to unpack` | FlashInfer arch empty string | Verify `flashinfer_arch_parse.patch` applied | Round 1 |
| `NCCL error: invalid usage` | NCCL NET plugin missing | `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1` | Round 10 |
| `ValueError: int() with base 10: ''` | Empty `VLLM_MEMORY_PROFILER...` | Set `=0` in .env AND docker-compose.yml | Round 12 |
| FlashInfer sm75 check | SM12x not in whitelist | `VLLM_USE_FLASHINFER_SAMPLER=0` | Round 12 |
| Container crashes on start | Env var not propagated | Check docker-compose `environment:` section | Round 6 |
| First inference slow | Triton JIT warmup | Normal — subsequent requests are fast | Round 12 |
| `invalid device ordinal` | TP beyond available GPUs | Each Spark has 1 GPU only | Round 9 |
| Weight load hangs at 74% | Disk full / page cache | `df -h`, `drop_caches`, retry | Round 11 |
| Worker→Head connection fail | Wrong start order | Start Worker FIRST, then Head | Round 5 |
| `NVRM: NV_ERR_NO_MEMORY` | Unified memory OOM | Lower `--gpu-memory-utilization` | Round 11 |

## Diagnostic Commands · 诊断命令

```bash
# Check container env vars · 检查容器环境变量
docker exec vllm-spark-head env | grep -E 'NCCL|VLLM|FLASHINEF'

# Check NCCL NET plugin · 检查NCCL网络插件
docker exec vllm-spark-head ldconfig -p | grep nccl-net

# Check FlashInfer patch · 检查FlashInfer补丁
docker exec vllm-spark-head grep -n "parts = arch.split" /usr/local/lib/python3.12/dist-packages/flashinfer/compilation_context.py

# Check GPU memory · 检查GPU内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check system memory · 检查系统内存
free -h

# Check startup log tail · 查看启动日志尾部
docker logs vllm-spark-head --tail 50

# Test inference · 测试推理
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'

# Check NCCL connectivity between nodes · 检查双机NCCL连通性
# (Inside head container)
docker exec vllm-spark-head python3 -c "import torch; print(torch.cuda.nccl.version())"
```

## Startup Failure Root Cause Analysis · 启动失败根因分析

When vLLM Engine initialization fails, the error message is often generic:

当vLLM引擎初始化失败时，错误信息通常是笼统的：

```
RuntimeError: Engine core initialization failed
```

**Always scroll UP in the logs to find the real error above this message.**
**一定要往上翻日志，找这个信息上面的真正错误。**

Common patterns to search for in logs 日志中常见的搜索关键词：
- `NCCL` — network communication issues 网络通信问题
- `FlashInfer` / `FlashInfer` — attention/sampler issues attention/采样器问题
- `ValueError` — env var parsing issues 环境变量解析问题
- `OOM` / `memory` — memory budget exceeded 内存超预算
- `NVRM` — GPU driver errors GPU驱动错误
- `Traceback` — Python exceptions Python异常

## Memory Emergency Procedure · 内存应急流程

If the system freezes or snow-screens (OOM):
如果系统死机或雪花屏（OOM）：

1. Wait 2-3 minutes — OOM Killer may recover
   等待2-3分钟——OOM Killer可能自动恢复
2. If still frozen → **hard power cycle** (hold power button 10s)
   如果仍然死机→**硬重启**（长按电源键10秒）
3. After reboot, verify RoCE IPs are up
   重启后，检查RoCE IP是否正常
4. Drop caches before next attempt
   下次启动前清缓存
