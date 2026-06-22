# codewhale-proxy Phase 0+1 改造总结

> **文档日期**: 2026-06-08
> **文档目的**: 供其他 agent 评审本次改造的背景、方案、实现与潜在问题
> **改造范围**: Phase 0 (致命缺陷修复) + Phase 1 (KV cache 优化)
> **参考研究**: `/tmp/kanban-shared/t_a64b8b35/comprehensive-report.md` (605行, 32KB)

## 快速参考：每项改造的开源依据

| 改造 | 参考项目 | 仓库 | 源码文件:行号 |
|:---|:---|:---|:---|
| F1 | **无**（纯 bug fix） | — | `converter.rs:33→35` |
| F2 | **无**（基础设施） | — | `converter.rs:99-108`（tracing 日志） |
| F3 | **CodeWhale** | [Hmbown/CodeWhale](https://github.com/Hmbown/CodeWhale) | `prefix_cache.rs:64-66` |
| F4 | **调研根因分析** | — | `research-codewhale-root-cause.md §3.4` |
| F5 | **DeepSeek-Reasonix** | [esengine/deepseek-reasonix](https://github.com/esengine/deepseek-reasonix) | `openai.go:209-214, 447-448` |
| F6 | **CodeWhale** | [Hmbown/CodeWhale](https://github.com/Hmbown/CodeWhale) | `prefix_cache.rs`（全文 534行） |

**实测验证**: KV cache 命中率 91-94%（eswitch→阿里百炼 DeepSeek V4 Pro），流式 SSE 事件完整（thinking_delta+signature_delta→text_delta）。

---

## 1. 背景

### 1.1 当前架构

```
Claude Code → cc-connect (systemd, ANTHROPIC_BASE_URL=127.0.0.1:11435)
               → codewhale-proxy (11435, Rust) → eswitch (11434, Go)
                 → 阿里百炼 (llmapi.efunds.com.cn) → DeepSeek V4 Pro / V4 Flash
```

codewhale-proxy 是将 Anthropic Messages API 转换为 OpenAI Chat Completions API 的协议代理，专门针对 DeepSeek V4 模型优化。基于 CodeWhale (Hmbown/CodeWhale) 源码翻译而来，约 2500 行 Rust。

### 1.2 核心问题

用户报告：使用第三方 API（阿里百炼）提供的 DeepSeek V4 模型时，**KV cache 完全无法命中，导致费用异常高昂**。需要同时满足：
1. 开启最高推理强度 (reasoning_effort="max")
2. 高 KV cache 命中率（DeepSeek 前缀缓存机制生效）

### 1.3 历史调研结论

| 代理 | 结论 | 原因 |
|------|------|------|
| **CCR** (musistudio/claude-code-router) | ❌ 放弃 | `_reasoningCache` 单例，per-message reasoning 无法保存 |
| **cc-switch** (farion1231/cc-switch) | ❌ 放弃 | `supports_reasoning_effort()` 不包含 DeepSeek |
| **9router** | ❌ 放弃 | DeepSeek + 工具调用 100% 故障 (issue #1382) |
| **codewhale-proxy** (当前) | ⚠️ 修复 | 理论正确但实现有 4 个独立缺陷 |

---

## 2. 根因分析（4 个缺陷）

### 根因 1: `is_reasoning_model` 使用错误的模型名 (Critical)

**文件**: `src/anthropic/converter.rs:33`
**证据**: 调研 A `research-codewhale-root-cause.md:61-133`

```rust
// 修复前
let model = req.model.clone();                    // "claude-opus-4-7"
let is_reasoning_model = requires_reasoning_content(&model);  // → false!
let upstream_model = map_model_to_upstream(&model); // "deepseek-v4-flash"

// 影响链
is_reasoning_model=false → build_chat_messages(include_reasoning=false)
  → thinking 文本被提取但丢弃 → reasoning_content 仅 tool_calls 消息设置 "(reasoning omitted)"
    → DeepSeek 收到的 assistant 消息字节与它自己生成的响应不一致
      → KV cache 从第一个 assistant 消息开始完全失效
```

`requires_reasoning_content("claude-opus-4-7")` 只匹配 `deepseek-v4*` 前缀，对 Anthropic 模型名始终返回 false。

### 根因 2: Tool Schema 顺序不稳定 (Critical, 高概率)

**文件**: `src/anthropic/converter.rs:68-74`

Claude Code 在不同轮次可能以不同顺序发送工具列表。DeepSeek 的 KV cache 基于整个请求体的字节序列进行前缀匹配，工具列表顺序变化 → 字节序列不同 → cache miss。

CodeWhale 在 `prefix_cache.rs:64-66` 中**显式排序**工具列表后再计算 SHA-256 指纹，codewhale-proxy 缺失此防护。

### 根因 3: 零可观测性 (Critical)

在全部 16 个源文件中搜索 `sha256`、`fingerprint`、`cache`、`prefix` 结果均为 0。无法诊断：
- 请求体字节是否跨轮次变化
- 哪部分前缀发生了变化
- 缓存命中率是多少

CodeWhale 的 `PrefixStabilityManager`（`prefix_cache.rs`，534行）和 Reasonix 的 `CompareShape`（`cache_shape.go:67-94`）都提供了完整的诊断体系，codewhale-proxy 未移植任何一项。

### 根因 4: 可变字段透传 (Medium, 高概率)

**文件**: `src/anthropic/converter.rs:58-65`

`temperature`、`top_p`、`stop_sequences` 从 Claude Code 请求直接透传到 OpenAI 请求。`serde_json` 对 `None` 和 `Some(v)` 产生不同的字节序列，跨轮次变化 → KV cache miss。

---

## 3. Phase 0 — 致命缺陷修复（已部署）

### F1: 修复 `is_reasoning_model` 模型名

**参考**: 无（纯 bug fix）
**文件**: `src/anthropic/converter.rs:33-35`，1 行改动

```rust
let upstream_model = map_model_to_upstream(&model);
let is_reasoning_model = requires_reasoning_content(&upstream_model);  // ← 修复
```

### F2: 添加请求体日志

**参考**: 无（可观测性基础建设）
**文件**: `src/anthropic/converter.rs:99-108`，+11 行

```rust
let body_bytes = serde_json::to_vec(&openai_req).unwrap_or_default();
tracing::info!(
    body_len = body_bytes.len(),
    model = %openai_req.model,
    msg_count = openai_req.messages.len(),
    has_tools = openai_req.tools.is_some(),
    reasoning_effort = ?openai_req.reasoning_effort,
    "OpenAI request built"
);
```

### F3: Tool Schema 按名称排序

**参考**: CodeWhale `prefix_cache.rs:64-66`（显式排序消除顺序差异）
**文件**: `src/anthropic/converter.rs:67-72`，+1 行

```rust
let mut openai_tools: Vec<OpenAiTool> = tools.iter().map(|t| convert_tool(t)).collect();
openai_tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));
```

**安全性验证**: `tool_choice` 按名称选择工具，数组索引变化不影响功能。

### F4: 固定 temperature/top_p/stop 为 None

**参考**: 调研报告根因 4 分析
**文件**: `src/anthropic/converter.rs:57-63`，3 行改动

```rust
temperature: None,   // 原: req.temperature
top_p: None,         // 原: req.top_p
stop: None,          // 原: req.stop_sequences.clone()
```

**`stop_sequences` 丢弃验证**: 实测确认 DeepSeek V4 Pro **不支持** `stop` 参数（三组测试均忽略，始终自然结束）。丢弃无功能退化。

---

## 4. Phase 1 — KV Cache 优化（已部署）

### F5: 停止注入 reasoning_content

**参考**: **DeepSeek-Reasonix** (esengine/deepseek-reasonix, Go)

**源码证据 1** — `openai.go:209-214`：
> reasoning_content is deliberately NOT sent back: it's a response-only field. DeepSeek counts re-sent reasoning as billable prompt input (measured ~500 extra tokens per turn on a reasoner chain); MiMo accepts it but does not require it (verified empirically: multi-turn tool-call sessions work fine without it, saving ~18 tokens/turn).

**源码证据 2** — `openai.go:447-448`，`chatMessage` 结构体：
> // reasoning_content is deliberately NOT sent back to the API — it is response-only and re-sending it is counted as prompt tokens.

**文件**: `src/reasoning/build_messages.rs:256-310`

**修改前**（CodeWhale 原始设计）:
```rust
let thinking_text = thinking_parts.join("\n");
let has_reasoning = include_reasoning && !thinking_text.trim().is_empty();
let reasoning_content = if has_reasoning {
    Some(thinking_text)              // 从 thinking blocks 提取并注入
} else if has_tool_calls {
    Some("(reasoning omitted)")
} else {
    None
};
```

**修改后**（参考 Reasonix 不回传策略）:
```rust
// F5: reasoning_content is deliberately NOT sent back.
// Reference: Reasonix openai.go:209-214 — reasoning_content is a response-only field;
// re-sending it is counted as billable prompt input (~500 tokens/turn saved).
// Reference: Reasonix openai.go:447-448 — chatMessage struct comment confirms.
let has_tool_calls = !tool_calls.is_empty();
let reasoning_content = if has_tool_calls {
    Some("(reasoning omitted)".to_string())   // 仅 tool_calls 保留占位符(DeepSeek 400 防护)
} else {
    None                                       // 纯文本消息不注入
};
```

**关键决策理由**:
- 调研报告 section 4.3 指出矛盾信息：CodeWhale 主张回传，Reasonix 主张不回传
- 裁决采用 Reasonix 策略：DeepSeek 官方文档明确 reasoning_content 是响应专属字段，Reasonix 实证验证多轮 tool-call 不带回传完全工作，且节省 ~500 tokens/turn

### F6: 移植 PrefixStabilityManager

**参考**: **CodeWhale** (Hmbown/CodeWhale, Rust)

**源码证据** — `prefix_cache.rs:64-66`（工具排序）+ 完整 534 行实现

**新建文件**: `src/reasoning/prefix.rs` (334行)

**移植的核心组件**:

| 组件 | CodeWhale 原版 | codewhale-proxy 移植 |
|------|---------------|---------------------|
| `compute_prefix_fingerprint()` | SHA-256 双哈希 (system+tools) | 同，+16-char hex truncation |
| `PrefixStabilityManager` 结构体 | 采样计数、稳定性统计 | 同，+`consecutive_stable` 字段 |
| `check_and_update()` | 对比指纹，分类变更 | 同，四分类：None/SystemChanged/ToolsChanged/LogRewrite |
| `stability_ratio()` | 命中率统计 | 同 |
| 工具排序 | `serialized.sort()` | 在 `compute_prefix_fingerprint` 中实现 |

**集成方式**: 在 `src/anthropic/converter.rs` 的日志中输出 prefix_fingerprint，在 `src/main.rs` 中初始化全局 `PrefixStabilityManager`。

---

## 5. 验证结果

### 5.1 单元测试

```
30/30 passed (Phase 0: 23 test + Phase 1: 7 new prefix tests)
  - prefix::test_compute_prefix_fingerprint_deterministic
  - prefix::test_compute_prefix_fingerprint_different_system
  - prefix::test_compute_prefix_fingerprint_with_tools
  - prefix::test_compute_prefix_fingerprint_tool_order_independent
  - prefix::test_stability_manager_initial
  - prefix::test_stability_manager_check_and_update
  - prefix::test_stability_manager_detects_change
```

### 5.2 功能测试 (5/5 PASS)

| 测试 | 结果 | 关键证据 |
|:---|:---:|:---|
| 基础 reasoning 透传 | ✅ | thinking block + text block 正确返回 |
| 多轮 F5 不注入 | ✅ | reasoning_content 不再注入纯文本消息 |
| 工具调用+thinking | ✅ | tool_use + thinking 同时存在 |
| 流式 SSE thinking | ✅ | 59 thinking_delta + signature_delta → content_block_stop(index=0) → text block，完整 9 事件生命周期 |
| F6 prefix 日志 | ✅ | prefix_fingerprint 日志正常输出 (14条) |

### 5.3 KV Cache 命中实测

通过 eswitch → 阿里百炼 → DeepSeek V4 Pro 实测：

| 测试 | 命中率 | 数据 |
|:---|:---:|:---|
| 多轮对话 Round 2 | **91%** | `prompt_cache_hit=1280/1404` |
| 多轮对话 Round 3 | **94%** | `prompt_cache_hit=768/814` |
| 多轮对话 Round 4 | **93%** | `prompt_cache_hit=768/823` |

**依据**: DeepSeek API 文档 `https://api-docs.deepseek.com/guides/kv_cache` — 自动磁盘缓存，`prompt_cache_hit_tokens` 字段报告命中情况。阿里百炼隐式缓存（≥256 tokens 自动触发）兼容此机制。

### 5.4 部署状态

- 二进制: `/home/clawbot/codewhale-proxy-prod/codewhale-proxy` (7.6MB)
- 端口: 11435
- 备份: `codewhale-proxy.bak.20260608-080329`
- cc-connect 已配置 `ANTHROPIC_BASE_URL=http://127.0.0.1:11435`

---

## 6. 已知问题与风险

### 6.1 架构层面的固有限制

**问题**: codewhale-proxy 是无状态代理，不持有会话历史。DeepSeek 的 prefix cache 最佳实践（如 Reasonix 的 append-only 消息循环）要求代理控制消息序列。Claude Code 自行管理消息历史和 compaction，代理无法确保跨轮次 prefix stability。

**影响**: KV cache 命中率的上限由 Claude Code 的消息管理策略决定，代理只能优化自己能控制的部分（请求体格式一致性）。

### 6.2 阿里百炼兼容性

**问题**: 阿里百炼使用非标准 `enable_thinking: true` 参数而非标准的 `thinking: {type: "enabled"}`。调研报告 `research-api-kvcache.md:186-201` 记录此差异。

**当前处理**: eswitch（Go binary，无配置文件）在中间层可能做了转换。如果 eswitch 未处理此差异，thinking 参数可能被阿里百炼忽略。

**验证方式**: 检查 eswitch 发出的实际请求体（eswitch 为黑盒，无法直接验证）。

### 6.3 stop_sequences 丢弃

**问题**: F4 将 `stop_sequences` 设为 `None`。

**验证**: 实测确认 DeepSeek V4 Pro 不支持 `stop` 参数（三组测试均忽略），丢弃无功能退化。但如果后续模型版本（如 V5）支持 `stop`，需要恢复。

### 6.4 reasoning_content 占位符回退与阿里百炼 max thinking 兼容性（✅ 已防护）

**原始问题**: 阿里百炼部署的 DeepSeek API 在开启最高推理强度（`reasoning_effort="max"`）时，多轮对话中若 history 包含 tool_calls 的 assistant 消息，**必须携带 reasoning_content 字段**。缺失会导致：
- DeepSeek API 返回 400 错误（"reasoning_content is required for tool_calls messages"）
- Claude Code 收到 400 → 自动重试 → 再次 400 → **死循环**

**当前防护机制（双重安全网）**:

| 防护层 | 位置 | 机制 | 触发条件 |
|:---|:---|:---|:---|
| **第 1 层** | `build_messages.rs:303-305` | 构建消息时，若 assistant 有 tool_calls，自动注入 `"(reasoning omitted)"` 占位符 | 消息构建阶段 |
| **第 2 层** | `sanitize.rs:6-45` + `converter.rs:104` | 请求发出前最后一刻，遍历所有 assistant 消息，对 tool_calls 且无 reasoning_content 的补充占位符 | 请求序列化后、发送前 |

**实测验证**:
```
Test: tool_calls history WITHOUT reasoning_content + max thinking (budget_tokens=16000)
Result: HTTP 200 ✅, 正常返回 thinking + text blocks
```

**结论**: 阿里百炼的 reasoning_content 必需性问题已在当前代理中通过双重防护完全解决，不会出现 400 错误或死循环。即使第 1 层因代码变更失效，第 2 层（sanitize）作为最后防线仍会兜底。

**风险**: `"(reasoning omitted)"` 是静态占位符，不是模型实际推理内容。对于 max reasoning 场景，使用占位符可能影响模型在多轮对话中的推理连贯性。但这是成本（token 节省）与质量（推理连贯性）之间的权衡，Reasonix 已验证此策略在生产中可用。

---

## 7. 后续改造方向

### Phase 2 — 中期增强（未实施）

| 改造 | 参考 | 内容 | 预计 |
|------|------|------|:---:|
| F7: Schema 递归规范化 | **Reasonix** `schema_canonicalize.go:10-24` | JSON Schema 内部结构递归排序（required 数组等） | 半天 |
| F8: Session 感知 prefix shape | **CodeWhale** `prefix_cache.rs` | 轻量 session 管理，跨对话 prefix 稳定性 | 1-2天 |
| F9: Stream 重连 | — | SSE 流断开后自动重连 | 半天 |

### Phase 3 — 长期演进（未实施）

| 改造 | 借鉴来源 | 内容 |
|------|---------|------|
| F10: SQLite reasoning_content 持久化缓存 | **deepseek-cursor-proxy** (405★) | 响应时保存 reasoning → 请求时从数据库恢复，彻底解决 CC 裁剪 thinking blocks 的问题 |
| F11: LCM DAG 上下文注入评估 | **deeplossless** (4★, Rust) | 架构最接近的 Rust 代理，评估其 LCM 压缩方案是否可借鉴 |
| F12: reasoning_effort 统一处理 | **envoy AI Gateway** | 标准化 reasoning_effort 参数处理逻辑 |

---

## 8. 改造溯源表（含完整开源参考链接）

每一项改造均标注参考的开源项目、代码仓库地址、具体文件路径和行号。**禁止主观推测——所有设计决定均有源码引用**。

| 改造 | 参考项目 | 仓库 | 源码文件 | 行号 | 参考内容 |
|:---|:---|:---|:---|:---|:---|
| **F1** | 无（纯 bug） | — | `codewhale-proxy/src/anthropic/converter.rs` | 33→35 | 将 `requires_reasoning_content(&model)` 改为 `requires_reasoning_content(&upstream_model)` |
| **F2** | 无（基础设施） | — | — | — | tracing 日志框架，无外部参考 |
| **F3** | **CodeWhale** | [Hmbown/CodeWhale](https://github.com/Hmbown/CodeWhale) | `crates/tui/src/client/prefix_cache.rs` | 64-66 | 显式排序工具列表消除顺序差异：`serialized.sort()` |
| **F4** | 调研根因分析 | — | `research-codewhale-root-cause.md` | §3.4 | temperature/top_p/stop 可变字段透传破坏请求体字节一致性 |
| **F5** | **DeepSeek-Reasonix** | [esengine/deepseek-reasonix](https://github.com/esengine/deepseek-reasonix) | `internal/provider/openai/openai.go` | 209-214 | reasoning_content 是 "response-only field"，不回传节省 ~500 tokens/turn |
| | | | `internal/provider/openai/openai.go` | 447-448 | `chatMessage` 结构体注释：不回传 reasoning_content |
| **F6** | **CodeWhale** | [Hmbown/CodeWhale](https://github.com/Hmbown/CodeWhale) | `crates/tui/src/client/prefix_cache.rs` | 全文 534行 | `PrefixStabilityManager`: `compute_prefix_fingerprint` + `check_and_update` + `stability_ratio` |

### 8.1 参考源码原文摘录

#### F3 参考 — CodeWhale `prefix_cache.rs:64-66`（工具排序）

```rust
// CodeWhale 源码 — 显式排序工具列表以消除顺序差异
let mut serialized: Vec<String> =
    tools.iter().filter_map(tool_to_api_json).collect();
serialized.sort();  // ← 关键：消除工具注册顺序对 KV cache 的影响
```

#### F5 参考 — Reasonix `openai.go:209-214`（不发送 reasoning_content）

```go
// Reasonix 源码 — reasoning_content 不回传策略
// reasoning_content is deliberately NOT sent back: it's a response-only
// field. DeepSeek counts re-sent reasoning as billable prompt input
// (measured ~500 extra tokens per turn on a reasoner chain); MiMo accepts
// it but does not require it (verified empirically: multi-turn tool-call
// sessions work fine without it, saving ~18 tokens/turn). The session
// still keeps it (for display/archive); we just don't pay to re-upload it.
```

#### F5 参考 — Reasonix `openai.go:447-448`（chatMessage 结构体）

```go
// Reasonix 源码 — chatMessage 结构体注释
type chatMessage struct {
    Role             string
    Content          string
    // reasoning_content is deliberately NOT sent back to the API —
    // it is response-only and re-sending it is counted as prompt tokens.
    ToolCalls        []chatToolCall
    // ...
}
```

#### F6 参考 — CodeWhale `prefix_cache.rs`（核心 API）

```rust
// CodeWhale 源码 — PrefixStabilityManager 核心接口
impl PrefixStabilityManager {
    pub fn compute_prefix_fingerprint(system: &str, tools: &[Tool]) -> String;
    pub fn check_and_update(&mut self, fingerprint: &str) -> ChangedFields;
    pub fn stability_ratio(&self) -> f64;
}

enum ChangedFields {
    None,
    SystemChanged,
    ToolsChanged,
    LogRewrite,
}
```

---

## 9. 评审检查清单

供其他 agent 评审时逐项检查：

- [ ] **F1**: `converter.rs:35` — `requires_reasoning_content(&upstream_model)` vs 原 `&model`
- [ ] **F3**: `converter.rs:72` — `sort_by(|a,b| a.function.name.cmp(&b.function.name))` 是否破坏 `tool_choice`
- [ ] **F4**: `converter.rs:57-63` — `temperature/top_p/stop→None` 经实测 DeepSeek 不支持 `stop`
- [ ] **F5**: `build_messages.rs` — reasoning_content 不再注入，引用 Reasonix `openai.go:209-214`
- [ ] **F6**: `prefix.rs` — SHA-256 双哈希是否与 CodeWhale `prefix_cache.rs` 算法一致
- [ ] **架构风险**: 无状态代理无法控制消息循环，KV cache 上限由 Claude Code 决定
- [ ] **阿里百炼**: `enable_thinking` 非标准参数，eswitch 黑盒是否处理
- [ ] **测试覆盖**: 30/30 单元测试 + 5/5 功能测试，是否有遗漏的边界条件
- [ ] **回滚方案**: 备份二进制 `*.bak.20260608-080329`，CCR(3456) 保持运行不受影响
