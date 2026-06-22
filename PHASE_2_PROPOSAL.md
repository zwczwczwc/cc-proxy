# codewhale-proxy 多实例隔离 + Docker 化 + 模型配置化方案

> **日期**: 2026-06-08
> **状态**: 待评审

---

## 1. 现状与问题

### 1.1 当前拓扑

```
本机 (PXT-Ubuntu-Internet35147, 100.64.0.9)
  └─ codewhale-proxy :11435 → eswitch :11434 → 阿里百炼 → DeepSeek
  └─ cc-connect :11435 (ANTHROPIC_BASE_URL)

GPU 工作站 (zhengweicheng-Default-string, 100.64.0.6)
  └─ codewhale-proxy :11439 → eswitch :11434 → 阿里百炼 → DeepSeek
  └─ CCR :3456 (备)
  └─ cc-connect → ANTHROPIC_BASE_URL=http://127.0.0.1:11440
```

### 1.2 四个待解决问题

| # | 问题 | 现状 | 影响 |
|:--|:---|:---|:---|
| **1** | PREFIX_MANAGER 无多实例隔离 | 全局 `LazyLock<Mutex<>>` 单例 | 多 CC 共用时 stability_ratio 失真 |
| **2** | GPU 代理需独立维护 | 两套独立部署、独立升级 | 维护成本 ×2 |
| **3** | 无容器化管理 | 裸进程运行 | 无健康检查、无自动重启、无版本管理 |
| **4** | 模型映射硬编码 | `converter.rs` 中硬编码 match 分支 | 换模型需重新编译部署 |

---

## 2. 方案设计

### 2.1 PREFIX_MANAGER 多实例隔离

**问题**: 全局单例导致不同 CC 实例的 prefix 指纹混在一起，`stability_ratio` 失真。

**方案**: 移除全局单例，改为**请求级 PrefixStabilityManager**。由于 codewhale-proxy 是无状态代理（不持有会话），没有 conversation_id 来分组，最优策略是每次请求独立计算 fingerprint 并随日志输出，不做跨请求对比。

```rust
// 改造前（converter.rs:13）
static PREFIX_MANAGER: LazyLock<Mutex<PrefixStabilityManager>> = ...;

// 改造后：每次请求独立计算，不对比历史
let fingerprint = compute_prefix_fingerprint(sys_prompt, tools);
tracing::info!(prefix_fingerprint = %fingerprint, ...);
```

**跨请求对比的替代方案**: 如果有 conversation_id（Anthropic Messages API 无此字段），可以按 conversation_id 分组。当前协议不支持，因此只做单次请求的 fingerprint 输出。外部监控系统（如日志聚合）可以按 system prompt hash 或 tool hash 分组分析 stability。

**影响**: 移除 `stability_ratio` 和 `consecutive_stable` 字段（在多实例场景下本来就已经不准确）。

### 2.2 GPU 工作站代理切换

**现状**: GPU 的 cc-connect 指向 `http://127.0.0.1:11440`，需改为本机代理 `http://100.64.0.9:11435`。

**可行性**: ✅ 完全可行
- 两台机器通过 Tailscale mesh 互联（100.64.0.9 ↔ 100.64.0.6），延迟通常 <5ms
- 代理是无状态 HTTP 服务，跨网络无状态问题
- GPU 的 eswitch 端口也是 11434（本地），代理转发路径不变

**带宽/延迟评估**:
- Tailscale 直连延迟: ~2-5ms（同机房）
- 单次 API 调用额外延迟: 2×2-5ms = 4-10ms（请求+响应各一次转发）
- 对 Claude Code 的交互体验几乎无感知影响

**改造步骤**:
1. 确保本机代理监听 `0.0.0.0:11435`（已在 listen，确认 Tailscale 可达）
2. 修改 GPU 的 cc-connect 配置: `ANTHROPIC_BASE_URL = "http://100.64.0.9:11435"`
3. 重启 GPU 的 cc-connect
4. GPU 的 codewhale-proxy (11439) 保留作为回退，CCR (3456) 保留不变

### 2.3 Docker 化

**方案**: 多阶段 Docker 构建，最小化镜像体积。

```dockerfile
FROM rust:1.85-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/codewhale-proxy /usr/local/bin/
EXPOSE 11435
ENV LISTEN_ADDR="0.0.0.0:11435"
ENV ESWITCH_URL="http://127.0.0.1:11434"
ENV DEEPSEEK_API_KEY="***"
ENV RUST_LOG="info"
HEALTHCHECK --interval=30s CMD curl -f http://localhost:11435/health || exit 1
ENTRYPOINT ["/usr/local/bin/codewhale-proxy"]
```

**优势**:
- `docker restart` 自动重启
- `HEALTHCHECK` 自动健康监控
- `docker logs` 统一日志
- `docker compose` 管理多服务
- 镜像仓库版本管理（推送到私有 registry）

**需解决的问题**:
- eswitch 在宿主机 `127.0.0.1:11434`，容器内 `127.0.0.1` 指向容器自己
- 方案: 使用 `host.docker.internal:11434` 或 `--network host`

### 2.4 模型映射配置化

**方案**: TOML 配置文件 + 环境变量覆盖。

```toml
# /etc/codewhale-proxy/config.toml
[models]
default = "deepseek-v4-pro"

[models.mapping]
"claude-opus-4-7" = "deepseek-v4-pro"
"claude-opus-4-6" = "deepseek-v4-pro"
"claude-opus-4-5" = "deepseek-v4-pro"
"claude-opus-4"   = "deepseek-v4-pro"
"claude-sonnet-4-6" = "deepseek-v4-pro"
"claude-sonnet-4-5" = "deepseek-v4-pro"
"claude-sonnet-4"   = "deepseek-v4-pro"
"claude-haiku-4-5" = "deepseek-v4-pro"
"claude-3-haiku"   = "deepseek-v4-pro"
"claude-haiku-4"   = "deepseek-v4-pro"
```

**加载优先级**: 环境变量 `MODEL_CONFIG_PATH` → `/etc/codewhale-proxy/config.toml` → 内置默认值

**热重载**: 不支持（避免运行时模型切换导致的不一致）。需 `docker restart` 生效。

**实现量**: ~50 行 Rust（`serde` + `toml` 反序列化 + HashMap 查找替代 match）

---

## 3. 改动范围估算

| 改动 | 文件 | 类型 | 复杂度 | 预计 |
|:---|:---|:---|:---|:---|
| PREFIX_MANAGER 去全局化 | `converter.rs` | 修改 | 低 | 30 分钟 |
| | `prefix.rs` | 删除跨请求对比 | 低 | 15 分钟 |
| | `main.rs` | 移除 prefix 初始化 | 一行删 | 5 分钟 |
| GPU 代理切换 | GPU `cc-connect` 配置 | 配置修改 | 低 | 5 分钟 |
| Docker 化 | `Dockerfile` | 新建 | 中 | 30 分钟 |
| | `docker-compose.yml` | 新建 | 低 | 15 分钟 |
| 模型配置化 | `config.rs` | 重写 | 中 | 30 分钟 |
| | `converter.rs` | match→HashMap | 低 | 15 分钟 |
| | `config.toml` | 新建 | 低 | 5 分钟 |

---

## 4. 风险与注意事项

| 风险 | 概率 | 缓解 |
|:---|:---|:---|
| GPU 跨网络代理增加延迟 | 低 | Tailscale 直连 <5ms，实测可接受；保留本地 fallback |
| Docker 网络隔离导致 eswitch 不可达 | 中 | 使用 `--network host` 或 `host.docker.internal` |
| 配置化后缺少默认值导致启动失败 | 低 | 保留硬编码 fallback，无配置文件时按内置默认启动 |
| PREFIX_MANAGER 去全局后丢失稳定性监控 | 低 | fingerprint 仍随日志输出，外部系统可做聚合分析 |
| GPU 的 eswitch 不可用导致回退链路失败 | 低 | GPU 本地 CCR(3456) + codewhale-proxy(11439) 保留不删 |
