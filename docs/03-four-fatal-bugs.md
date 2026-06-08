# 🐛 The Four Fatal Bugs (& Their Solutions)
# 四大致命Bug（及解法）

> These 4 bugs must be solved **simultaneously** — fixing only 1-2 leaves the service broken.
> 这4个Bug必须**同时解决**——只修1-2个服务仍然起不来。

---

## Bug #1: FlashInfer arch.parse Crash · FlashInfer架构解析崩溃

### Error 错误
```
File "flashinfer/compilation_context.py", line 65
major, minor = arch.split(".")
ValueError: not enough values to unpack (expected 2, got 1)
```

### Root Cause 根因
The jasl/vLLM fork sets `FLASHINFER_CUDA_ARCH_LIST=` (empty string). When FlashInfer parses it:
`"".split(" ")` → `[""]` → `"".split(".")` returns `[""]` (length 1), unpack to `major, minor` fails.

jasl/vLLM fork内部设置了`FLASHINFER_CUDA_ARCH_LIST=`（空字符串）。FlashInfer解析时：
`"".split(" ")` → `[""]` → `"".split(".")` 返回 `[""]`（长度1），解包失败。

### Solution 解法
Apply this patch to `flashinfer/compilation_context.py`:

```patch
--- a/flashinfer/compilation_context.py
+++ b/flashinfer/compilation_context.py
@@ -60,11 +60,20 @@ class CompilationContext:
     def __init__(self):
         self.TARGET_CUDA_ARCHS = set()
         if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
-            for arch in os.environ["FLASHINFER_CUDA_ARCH_LIST"].split(" "):
-                major, minor = arch.split(".")
+            for raw_arch in os.environ["FLASHINFER_CUDA_ARCH_LIST"].split(" "):
+                raw_arch = raw_arch.strip()
+                if not raw_arch:
+                    continue
+                parts = raw_arch.split(".")
+                if len(parts) == 2:
+                    major_str, minor_str = parts
+                elif len(parts) == 1 and parts[0].isdigit():
+                    major_str, minor_str = parts[0], "0"
+                else:
+                    continue
-                major = int(major)
+                major = int(major_str)
-                if minor[-1].isalpha():
-                    self.TARGET_CUDA_ARCHS.add((major, minor))
+                if minor_str[-1].isalpha():
+                    self.TARGET_CUDA_ARCHS.add((major, minor_str))
                 else:
                     self.TARGET_CUDA_ARCHS.add(
-                        self._normalize_cuda_arch(major, int(minor))
+                        self._normalize_cuda_arch(major, int(minor_str))
                     )
         else:
             try:
```

The patch is auto-applied by the container entrypoint. See [`patches/flashinfer/flashinfer_arch_parse.patch`](../patches/flashinfer/flashinfer_arch_parse.patch).

补丁在容器入口点自动应用。

---

## Bug #2: FlashInfer Sampler sm75 Check · 采样器SM75检查

### Error 错误
```
RuntimeError: FlashInfer requires GPUs with sm75 or higher, but got sm121
```

### Root Cause 根因
FlashInfer's JIT sampling kernel checks for sm75 compatibility during `_dummy_sampler_run`. GB10 uses SM12.1 which isn't in the whitelist.

FlashInfer的JIT采样kernel在`_dummy_sampler_run`中检查sm75兼容性。GB10的SM12.1不在白名单里。

### Solution 解法
```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

Set this in BOTH `.env` and `docker-compose.yml`'s `environment:` section.

在`.env`和`docker-compose.yml`的`environment:`段都要设。

**Why this is safe**: DeepSeek V4 Flash uses its own native DeepSeek sampler. FlashInfer's sampler is redundant for DSV4.

**为什么安全**：DSV4使用自家DeepSeek原生采样器，FlashInfer采样器是冗余的。

---

## Bug #3: NCCL NET Plugin Missing · NCCL NET插件缺失

### Error 错误
```
NCCL WARN Failed to initialize any NET plugin
RuntimeError: NCCL error: invalid usage
```

### Root Cause 根因
The `dsv4-d568` Docker image does NOT include `libnccl-net.so` (NCCL's IB verbs network plugin). The host has it at `/usr/lib/x86_64-linux-gnu/libnccl-net.so` but the container can't access it.

`dsv4-d568`Docker镜像没有打包`libnccl-net.so`（NCCL的IB verbs网络插件）。宿主机上有但容器访问不到。

Mounting `/sys/class/infiniband` alone is insufficient — the **shared library** must be inside the container.

只挂载`/sys/class/infiniband`设备不够——**共享库文件**必须在容器内部。

### Temporary Solution · 临时解法
```bash
NCCL_NET=Socket
NCCL_IB_DISABLE=1
```

Forces TCP over the management network. Functionality preserved, but speed drops from 620 Gb/s (RoCE) to ~1 Gb/s (management Ethernet).

强制走管理口TCP通信。功能正常，速度从620Gb/s（RoCE）降到~1Gb/s（管理千兆口）。

### Permanent Solution · 正式解法
Option A: Add to Dockerfile
```dockerfile
RUN apt-get update && apt-get install -y libnccl-net
```

Option B: Volume mount from host
```yaml
volumes:
  - /usr/lib/x86_64-linux-gnu/libnccl-net.so:/usr/lib/x86_64-linux-gnu/libnccl-net.so:ro
```

---

## Bug #4: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS Empty String · 空字符串崩溃

### Error 错误
```
ValueError: invalid literal for int() with base 10: ''
```

### Root Cause 根因
docker-compose.yml used `${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}` which passes an **empty string** when the env var is unset. Python's `int('')` crashes.

docker-compose.yml中写成`${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}`，变量未设置时传入**空字符串**。`int('')`崩溃。

### Solution 解法
```bash
# In BOTH .env AND docker-compose.yml
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

This skips the CUDA graph memory estimation. Without it, CUDA graph memory isn't accounted for during KV cache allocation — you may need to lower `--gpu-memory-utilization` slightly. We use 0.78 and it works fine.

这会跳过CUDA图内存预估。没有它KV分配时不计算CUDA图内存——可能需要稍微降低`--gpu-memory-utilization`。我们用0.78没问题。

---

## One More Thing: docker-compose Env Propagation 🚨

## 还有一个事：env变量传播 🚨

**This is NOT a bug per se, but the #1 cause of "but I set it in .env, why doesn't the container see it?"**

**这本身不是Bug，但却是"我在.env设了为什么容器没收到"的头号原因。**

`.env` values are only picked up by `docker-compose.yml` if explicitly declared in the `environment:` section!

`.env`的值只有被`docker-compose.yml`的`environment:`段显式声明了才会传入容器！

```yaml
services:
  head:
    environment:
      - NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}   # ← MUST declare! 必须声明！
      - VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}  # ← MUST!
```

Check with: 检查方法：
```bash
docker exec vllm-spark-head env | grep NCCL_P2P
```

Must return `NCCL_P2P_DISABLE=1`, not empty! 必须返回值，不是空的！

---

*All 4 bugs discovered and fixed over 12 rounds of deployment on June 7-8, 2026.*
*4个bug在2026年6月7-8日的12轮部署中发现并解决。*
