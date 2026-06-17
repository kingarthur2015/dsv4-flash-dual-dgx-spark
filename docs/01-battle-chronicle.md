# 📜 Battle Chronicle: 11 Rounds → 1 Breakthrough
# 战史：十一战终成

> The full story of Arthur, 虾丸, and the 11 failed attempts to deploy DSV4 Flash on 2× DGX Spark.
> Arthur和虾丸在2台DGX Spark上部署DSV4 Flash的11次失败和最终突破的完整故事。

---

## Round 1: FlashInfer Crash 🔴
## 第1轮：FlashInfer崩溃

We ran `docker compose up` for the first time. Both containers started, Ray cluster formed, then:

我们第一次运行`docker compose up`。容器启动，Ray集群组建，然后：

```
File "flashinfer/compilation_context.py", line 65
major, minor = arch.split(".")
ValueError: not enough values to unpack (expected 2, got 1)
```

**What we thought**: `TORCH_CUDA_ARCH_LIST` was set to `12.1a` — the "a" suffix must be the problem.
**我们以为**: `TORCH_CUDA_ARCH_LIST=12.1a` — 那个"a"后缀有问题。

**What was actually happening**: The `FLASHINFER_CUDA_ARCH_LIST` env var was set to an **empty string** by the jasl/vLLM fork. `"".split(".")` → `[""]` → unpack to `major, minor` fails.
**实际根因**: `FLASHINFER_CUDA_ARCH_LIST`被设为空字符串。`"".split(".")` 返回 `[""]`，解包失败。

---

## Round 2: Fixed the Wrong Thing 🔴
## 第2轮：修错了东西

Changed `TORCH_CUDA_ARCH_LIST` from `12.1a` to `12.1`. Didn't help — nvidia-container-toolkit overrides it anyway.

把`12.1a`改成`12.1`。没用——nvidia-container-toolkit会自动覆盖。

---

## Round 3: Found the Real Root Cause 🟢
## 第3轮：找到真凶

`docker exec` into the container and checked env vars:

```bash
FLASHINFER_CUDA_ARCH_LIST=    ← EMPTY STRING!
```

**The real villain was `FLASHINFER_CUDA_ARCH_LIST` being empty, not `TORCH_CUDA_ARCH_LIST` being weird.**
**真凶是`FLASHINFER_CUDA_ARCH_LIST`空字符串，不是`TORCH_CUDA_ARCH_LIST`带字母。**

---

## Round 4: Created the Patch 🟢
## 第4轮：创造补丁

Wrote `flashinfer_arch_parse.patch` — a 3-way guard against empty/single/double-part arch strings.

编写了3层防御性补丁，覆盖空/单段/双段三种架构字符串格式。

```python
# Before:
major, minor = arch.split(".")
# After:
parts = arch.split(".")
if len(parts) == 2: major_str, minor_str = parts
elif len(parts) == 1: major_str, minor_str = parts[0], "0"
else: continue
```

The patch is auto-applied by the container entrypoint. 补丁在容器入口点自动应用。

---

## Round 5: NCCL P2P Error 🔴
## 第5轮：NCCL P2P报错

```
NCCL WARN P2P is not supported on GB10
NCCL WARN Failed to initialize any NET plugin
```

`.env` had `NCCL_P2P_DISABLE=1` but it **wasn't in docker-compose.yml's environment section** → container never saw it.

`.env`里设了`NCCL_P2P_DISABLE=1`但**docker-compose.yml的environment段没写**→容器没收到。

---

## Round 6: Fixed docker-compose 🟢
## 第6轮：修了docker-compose

Added ALL NCCL/env vars to docker-compose.yml's `environment:` section for BOTH head and worker.

把所有NCCL/env变量加到docker-compose.yml的environment段——head和worker两个都要加。

**🔥 Key Lesson: `.env` values do NOT auto-propagate to containers! You must declare them in docker-compose.yml.**
**🔥 重要教训：`.env`的值不会自动传入容器！必须在docker-compose.yml的environment段声明！**

---

## Round 7: NCCL invalid usage 🔴
## 第7轮：NCCL非法用法

```
NCCL error: invalid usage
```

Tried `--disable-custom-all-reduce`. Didn't help. 试了`--disable-custom-all-reduce`没用。

---

## Round 8: More NCCL debugging 🔴
## 第8轮：继续NCCL调试

Disabling custom all-reduce didn't fix it. The real issue was deeper: no NCCL NET plugin available.

禁用自定义all-reduce没用。根因更深：NCCL NET插件不存在。

---

## Round 9: Switched to mp backend 🔴
## 第9轮：切到mp后端

Switched from `DISTRIBUTED_BACKEND=ray` to `=mp`. Still failed — same NCCL root cause.

从Ray切到mp后端。仍然失败——同一NCCL根因。

---

## Round 10: Found the NET Plugin Issue 🟢
## 第10轮：发现NET插件问题

```bash
# Inside container:
ldconfig -p | grep nccl-net
# → NOTHING! No libnccl-net.so!

# On host:
ls /usr/lib/x86_64-linux-gnu/libnccl-net.so
# → EXISTS!
```

**Root cause: Docker image doesn't package `libnccl-net.so` (NCCL's IB verbs plugin).**
**根因：Docker镜像没有打包`libnccl-net.so`（NCCL的IB verbs网络插件）。**

Mounting `/sys/class/infiniband` is NOT enough — the shared library must be inside the container.

只挂载`/sys/class/infiniband`设备不够——共享库文件必须在容器内部。

**Temporary fix**: `NCCL_NET=Socket` + `NCCL_IB_DISABLE=1` — fallback to TCP.
**临时解法**：Socket TCP回退。

---

## Round 11: Interrupted 🔴
## 第11轮：被打断

Socket fallback was configured, Worker B started successfully. But before Head A could start, the context window was compacted and deployment was interrupted.

Socket回退配好了，Worker B启动成功。启动Head A之前，上下文压缩导致部署中断。

---

## 🚀 Round 12: BREAKTHROUGH! 🎉
## 🚀 第12轮：突破！🎉

Started Head A. Three bugs still blocked us — but we fixed all three simultaneously:

启动Head A。还有三个bug挡路——但我们在同一轮全部解决：

### Bug 2 (finally fixed): FlashInfer sampler sm75 check
### Bug 2（终于解决）：FlashInfer采样器SM75检查

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher, but got sm121
```

**Fix**: `VLLM_USE_FLASHINFER_SAMPLER=0` — DSV4 uses its own DeepSeek sampler.
**解法**：禁用FlashInfer采样器，DSV4用自家DeepSeek采样器。

### Bug 4: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS empty
### Bug 4：环境变量空字符串

```
ValueError: invalid literal for int() with base 10: ''
```

**Fix**: Set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` in BOTH `.env` and `docker-compose.yml`.
**解法**：在.env和docker-compose.yml中都设为0。

### Bug 3 resolution: NCCL_NET=Socket
### Bug 3解法：Socket回退

`NCCL_NET=Socket` finally let NCCL initialize without the IB verbs plugin.

Socket模式终于让NCCL在没有IB verbs插件的情况下完成初始化。

### The log we'd been waiting for:
### 等待已久的日志：

```
✅ Loading safetensors: 100% | 46/46 [02:00]
✅ Loading weights took 119.74 seconds
✅ Model loading took 74.02 GiB memory
✅ CUDA graphs PIECEWISE 4/4 + FULL 3/3 captured
✅ init engine took 161.94 s
✅ Application startup complete.  🎉
✅ curl /health → HTTP 200
✅ curl /v1/chat/completions → "Hello! How can I help you today?"
```

### The H₂O Joke 🎯

Commander Arthur asked DSV4 Flash to tell a hard-tech joke:

指挥官Arthur让DSV4 Flash讲个硬科技笑话：

> A physicist, a mathematician, and a computer scientist walk into a bar.
> The physicist says: "I'll have H₂O."
> The mathematician says: "I'll have H₂O too."
> The computer scientist says: "I'll have H₂O too."
> Then all three die.
>
> (The joke: encoding ambiguity — the computer scientist's "H₂O" was interpreted as H₂O₂ = hydrogen peroxide.)

Commander's review: **"够硬！我喜欢！" (Hard enough, I love it!)** 🎖️

---

## Final State · 终战状态

| Metric 指标 | Node A (Head) | Node B (Worker) |
|------------|:------------:|:--------------:|
| Memory 内存 | 103/121 GiB | 100/121 GiB |
| GPU | 0% util, 44°C | 0% util, 46°C |
| CPU | 2.2% idle | 6.2% idle |
| Uptime 已运行 | 33+ minutes | 33+ minutes |

---

*Written by 虾丸将军, Heavy Infantry General of the Hotpot Shrimp Seven-Star Squad*
*虾丸将军 · 火锅虾七星战队 重装兵大将*
