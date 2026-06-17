# 💡 6 Key Lessons · 六大经验教训

## Lesson 1: Docker Compose Env Vars Don't Auto-Propagate
## 教训1：Docker Compose环境变量不会自动传播

**Chinese**: `.env`文件设置了不等于容器能读到。所有环境变量必须在`docker-compose.yml`的`environment:`段**显式声明**。我们第6轮才发现——之前所有NCCL参数都白设了。

**English**: Setting a variable in `.env` doesn't mean the container sees it. You must explicitly declare every env var in docker-compose.yml's `environment:` section. We learned this on Round 6.

```bash
# Verify with: 用这个检查
docker exec vllm-spark-head env | grep NCCL_P2P
# Must return NCCL_P2P_DISABLE=1, not empty!
```

## Lesson 2: SM12x Is a New Architecture
## 教训2：SM12x是全新架构

**Chinese**: GB10 (SM 12.1) 是NVIDIA新一代Blackwell架构在消费级市场的首发。所有检查sm75/sm80/sm90的库都会失败。做好打补丁或禁用冗余模块的准备。

**English**: GB10 (SM 12.1) is the first consumer Blackwell GPU. Libraries checking for sm75/sm80/sm90 will fail. Be ready to patch or disable non-essential modules.

Components affected: FlashInfer sampler, FlashInfer autotune, DeepGEMM, CUDA Graph (in upstream vLLM), some Triton kernels.

## Lesson 3: NCCL in Docker Needs Shared Libraries
## 教训3：Docker里的NCCL需要共享库

**Chinese**: 我们以为挂载`/sys/class/infiniband`就够了。错了。NCCL的NET插件`libnccl-net.so`必须在容器**内部**存在。只挂载设备不挂载库文件是没用的。

**English**: We thought mounting `/sys/class/infiniband` was sufficient. Wrong. The NCCL NET plugin `libnccl-net.so` must exist **inside** the container. Device mounts without library files don't work.

Check with: 用这个检查：
```bash
docker exec <container> ldconfig -p | grep nccl-net
```

## Lesson 4: GB10 Unified Memory Has No Hardware Boundary
## 教训4：GB10统一内存没有硬件边界

**Chinese**: 普通GPU上显存OOM只崩进程。在GB10上，CPU和GPU共享同一121GB物理内存——OOM会杀掉gnome-shell、sshd、journald，导致雪花屏、死机、硬重启。这是最危险的坑。必须保守设`--gpu-memory-utilization`。

**English**: On discrete GPUs, VRAM OOM just kills the process. On GB10, CPU and GPU share the same 121GB physical RAM — OOM kills gnome-shell, sshd, causing snow screen, system freeze, hard reboot. This is the most dangerous pitfall. Always set `--gpu-memory-utilization` conservatively.

## Lesson 5: Weight Loading Peak Exceeds Steady-State
## 教训5：权重加载峰值超过最终稳态

**Chinese**: 加载权重时，safetensors解码到CPU临时内存 + GPU参数初始化**同时**占用统一内存池。这个峰值远大于最终稳态大小。即使最终内存够，加载峰值也可能爆掉系统。

**English**: During weight loading, safetensors decoding to CPU temp memory + GPU parameter initialization use the SAME unified memory pool simultaneously. Peak >> steady-state. Even if your final budget fits, the loading peak can blow the system.

## Lesson 6: Don't Trust AI-Suggested Parameters
## 教训6：不要盲信AI建议的参数

**Chinese**: AI助手频繁推荐不存在的参数。以下**不存在**，不要加：

**English**: AI assistants frequently suggest non-existent parameters. These DO NOT EXIST:

| Non-existent 不存在 | Why 原因 |
|-------------------|---------|
| `--quantization fp4_fp8_mixed` | Does not exist in vLLM |
| `--expert-parallel-size N` | Only `--enable-expert-parallel` (no size) |
| `--distributed-executor-backend nccl` | Valid options: mp, ray, uni, external_launcher |
| `--offload-backend uva` on GB10 | Useless — CPU and GPU share same RAM |

Always verify with ````bash
vllm serve --help | grep <parameter>
```` and source code! 一定要用`--help`和源码验证！

## Bonus: The docker-compose.yml Trap
## 额外：docker-compose.yml陷阱

**Chinese**: `.env`文件和`docker-compose.yml`的关系要注意——**变量在`.env`中有不代表容器能读到**。必须在compose的`environment:`段用`${VAR:-default}`语法重新声明一遍。两个profile都要写（head和worker）。

**English**: The `.env` file and `docker-compose.yml` relationship is tricky — **having a variable in .env doesn't mean the container sees it**. You must re-declare it in the `environment:` section with `${VAR:-default}` syntax. For BOTH head and worker profiles.

```yaml
# docker-compose.yml 必须写！
environment:
  - NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}  # ← 从.env读取，容器才收得到！
```
