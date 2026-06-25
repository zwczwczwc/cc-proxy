# codewhale-proxy GLM-5.2 接入方案

> 版本: v0.1.3 | 日期: 2026-06-25 | 状态: 评审通过 ✅ | 评审轮次: 5 轮

---

## 一、背景

codewhale-proxy 当前仅支持 DeepSeek 系列模型。eswitch 上有 23 个模型可用，包括 `glm-5.2` 和 `glm-5.1`，但 proxy 未暴露。

### 架构说明

```
Claude Code → codewhale-proxy (Anthropic↔OpenAI 格式转换) → eswitch → 上游模型
```

GLM-5.2 使用与 DeepSeek **完全相同的 OpenAI 兼容格式**，可复用现有转换基础设施。

---

## 二、API 兼容性验证（实机测试）

| 特性 | DeepSeek V4 | GLM-5.2 | 兼容？ |
|------|:--:|:--:|:--:|
| 端点 | `/v1/chat/completions` | `/v1/chat/completions` | ✅ |
| 思考参数 | `thinking.type: "enabled"` | `thinking.type: "enabled"` | ✅ |
| 推理强度 | `reasoning_effort: "max"` | `reasoning_effort: "max"` | ✅ |
| 推理内容 | `reasoning_content` | `reasoning_content` | ✅ |
| 结束原因 | `stop`/`tool_calls`/`length` | `stop`/`tool_calls`/`length` | ✅ |
| 工具调用 | `tool_calls[].function.name/arguments` | 同格式 | ✅ |
| 缓存字段 | `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens` | `prompt_tokens_details.cached_tokens` | ⚠️ 格式不同 |
| 保留式思考 | 无此概念 | `clear_thinking: false`（**必须显式设置**） | ⚠️ GLM 特有 |

---

## 三、Cache Hit 差异

| 维度 | DeepSeek | GLM-5.2 |
|------|------|------|
| 缓存机制 | 严格前缀匹配 | 内容相似度匹配 |
| 缓存字段 | `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`（双字段） | `prompt_tokens_details.cached_tokens`（单字段） |
| 命中率公式 | `hit / (hit + miss)` | `cached_tokens / prompt_tokens` |
| 缓存价格 | 标准价的 ~2% | 标准价的 50% |
| 缓存要求 | 内容字节一致 | 相同或高度相似（完全相同命中率最高） |
| 推理上下文保留 | 自动保留 | **必须设置 `clear_thinking: false`** |

---

## 四、核心发现：`clear_thinking` 参数

### 问题

GLM-5.2 默认 `clear_thinking: true`（每轮清除推理上下文）。必须显式设置 `clear_thinking: false` 才能开启保留式思考（Preserved Thinking）。

### 影响链路

```
proxy 在 build_messages.rs 中回传 reasoning_content
    ↓
GLM 收到 reasoning_content
    ↓
clear_thinking: true（默认）→ GLM 忽略回传的推理内容 ❌
    ↓
推理链断裂 + 缓存命中率大幅下降
    ↓
Claude Code 多轮工具调用场景质量严重下降
```

### 修正

`clear_thinking` 必须在 `thinking` 对象**内部**（非顶层），序列化为：

```json
{"thinking": {"type": "enabled", "clear_thinking": false}}
```

**`types.rs`** — 加到 `DeepSeekThinking` 结构体：

```rust
pub struct DeepSeekThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_thinking: Option<bool>,  // ← 在 thinking 内部，非顶层
}
```

**`converter.rs`** — 在 `convert_request()` 中，`apply_effort_direct` 之后修改 `thinking` 字段：

```rust
// GLM-5.2: 保留式思考需要 clear_thinking=false 在 thinking 对象内
if upstream_model.starts_with("glm-5") {
    if let Some(ref mut thinking) = openai_req.thinking {
        thinking.clear_thinking = Some(false);
    }
}
```

**参考证据**：
- GLM 官方文档：`"thinking": {"type": "enabled", "clear_thinking": false}`
- CodeWhale：`body["thinking"] = json!({"type": "enabled", "clear_thinking": false})`

---

## 五、改动清单（6 处改动 + 1 处测试适配，约 46 行代码）

### 改动 1：`src/reasoning/requires.rs` — 识别 GLM 推理模型

**位置**：`requires_reasoning_content()` 函数，`deepseek-v4` 检查之后

```rust
// 新增
if model_lower.starts_with("glm-5") {
    return true;
}
```

**说明**：eswitch 上有 `glm-5.1` 和 `glm-5.2`（均为推理模型）。`glm-5` 前缀精确匹配，避免未来 `glm-4` 等非推理模型被误判。

---

### 改动 2：`src/anthropic/converter.rs` — GLM 模型名透传

**位置**：`map_model_to_upstream()` 函数，`deepseek` 检查之后

```rust
// 新增
if clean.starts_with("glm") {
    return clean.to_string();
}
```

**说明**：GLM 模型名直接透传到 eswitch，无需映射。`[1m]` 后缀在此之前已被剥离，无冲突。

---

### 改动 3：`src/routes/models.rs` — 暴露 GLM 模型

```rust
{
    "id": "glm-5.2",
    "object": "model",
    "created": 1725148800,
    "owned_by": "zhipuai"
}
```

---

### 改动 4：`src/openai/types.rs` — 新增两个字段

**新增结构体**：
```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u32>,
}
```

**Usage 结构体新增字段**：
```rust
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u32>,
    /// GLM-5.2: cached tokens via prompt_tokens_details
    pub prompt_tokens_details: Option<PromptTokensDetails>,  // 新增
}
```

**ChatCompletionRequest 无需新增字段**（`clear_thinking` 在 `DeepSeekThinking` 内部）。

**DeepSeekThinking 新增字段**：
```rust
pub struct DeepSeekThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// GLM-5.2: preserved thinking mode (clear_thinking=false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_thinking: Option<bool>,  // 新增
}
```

**说明**：
- `Option<T>` 字段在 serde 中自动容错——JSON 缺少该字段时默认 `None`，无需 `#[serde(default)]`
- `#[serde(skip_serializing_if = "Option::is_none")]` 确保 DeepSeek 请求中不包含 `clear_thinking`

---

### 改动 5：`src/sse/stream.rs` — KV Cache 日志双格式兼容

**当前代码**（仅 DeepSeek）：
```rust
let hit = u.prompt_cache_hit_tokens.unwrap_or(0);
let miss = u.prompt_cache_miss_tokens.unwrap_or(0);
```

**替换为**：
```rust
// Try DeepSeek format first, then GLM format
let (hit, miss) = {
    let ds_hit = u.prompt_cache_hit_tokens.unwrap_or(0);
    let ds_miss = u.prompt_cache_miss_tokens.unwrap_or(0);
    if ds_hit > 0 || ds_miss > 0 {
        (ds_hit, ds_miss)
    } else if let Some(ref details) = u.prompt_tokens_details {
        if let Some(cached) = details.cached_tokens {
            let total_prompt = u.prompt_tokens.unwrap_or(0);
            (cached, total_prompt.saturating_sub(cached))
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    }
};
// ... rest unchanged (total > 0 guard, hit_rate calculation)
```

**安全保护**：
- `total > 0` 防止除零
- `saturating_sub` 防止 `cached_tokens > prompt_tokens` 时 panic
- DeepSeek 路径优先，GLM 路径兜底

---

### 改动 6：`src/anthropic/converter.rs` — `clear_thinking: false` 注入

**位置**：`convert_request()` 函数中，`apply_effort_direct` 之后

```rust
// GLM-5.2: 保留式思考需要 clear_thinking=false 在 thinking 对象内
// 注意：此字段必须在 thinking 内部（非顶层），序列化为:
// {"thinking": {"type": "enabled", "clear_thinking": false}}
if upstream_model.starts_with("glm-5") {
    if let Some(ref mut thinking) = openai_req.thinking {
        thinking.clear_thinking = Some(false);
    }
}
```

---

### 改动 7：测试适配

**文件**：`src/anthropic/converter.rs` 测试 + `src/openai/converter.rs` 测试

```rust
// Usage 结构体字面量新增字段
prompt_tokens_details: None,

// DeepSeekThinking 结构体字面量新增字段（4 处）
clear_thinking: None,
```

---

## 六、不影响的部分（10+ 文件无需改动）

| 组件 | 原因 |
|------|------|
| `prefix.rs`（KV 指纹） | 纯内容哈希（system_prompt + tools），与模型名无关 |
| `build_messages.rs`（消息构建） | 通用 `reasoning_content` 处理，模型无关 |
| `relocate.rs`（env 重定位） | Claude Code 专用，模型无关 |
| `sanitize.rs`（内容净化） | 通用字段注入，模型无关 |
| `converter.rs`（`apply_effort_direct`） | GLM 使用相同的 `thinking.type` + `reasoning_effort` |
| `stream.rs`（SSE 状态机） | bool 门控，模型无关 |
| `client.rs`（HTTP 客户端） | 通用请求发送 |
| `messages.rs`（路由） | 通用请求调度 |

---

## 七、已知限制（后续迭代）

| 限制 | 严重度 | 说明 |
|------|:--:|------|
| `join("\n")` 可能影响 GLM 缓存 | P1 | 多 thinking block 边缘情况，Claude Code 通常只有 1 个 |
| `glm-5` 前缀可能过度匹配 | P2 | 如智谱发布 `glm-5-flash` 等非推理模型需调整 |
| `(reasoning omitted)` 占位符 | P2 | 极少触发，后续观察 |
| 默认思考行为差异 | P2 | GLM 默认开启思考，proxy 设 `max` 可能更激进 |

---

## 八、参考实现

| 项目 | 语言 | 特点 |
|------|------|------|
| [sunflower0305/claude-proxy](https://github.com/sunflower0305/claude-proxy) | TypeScript | 透传代理，原生支持 GLM + DeepSeek + Qwen 等 6 个提供商 |
| [xqsit94/glm](https://github.com/xqsit94/glm) | Go | GLM-5 + Claude Code CLI 工具 |
| [Hmbown/CodeWhale](https://github.com/Hmbown/CodeWhale) | Rust | codewhale-proxy 的核心函数来源 |
| [HosheaLi/dsv4-cc-proxy](https://github.com/HosheaLi/dsv4-cc-proxy) | Python | DeepSeek Anthropic 透传代理，600s 超时参考 |

---

## 九、评审记录

| 轮次 | 评审员 | 结论 | 关键发现 |
|:--:|--------|:--:|------|
| 1 | 3 个 subagent | ❌ FAIL | `starts_with("glm")` 太宽泛、缺少 `#[serde(default)]`、`owned_by` 错误 |
| 2 | 反驳验证 | ⚠️ 部分 | `#[serde(default)]` 冗余、`Option<T>` 自动容错 |
| 3 | 2 个 subagent | ✅ PASS | 方案正确，2 个小问题 |
| 4 | 3 个 subagent（外部评审） | ❌ FAIL | 🔴 **P0: `clear_thinking` 参数完全遗漏** |
| 4 修正 | Main Agent 分析确认 | ✅ PASS | 接受 P0 修正，新增改动 6 |
| 5 | 3 个 subagent（外部复审） | ❌ FAIL | 🔴 **P0: `clear_thinking` 位置错误**（应在 `thinking` 内部，非顶层） |
| 5 修正 | Main Agent 分析确认 | ✅ PASS | 移至 `DeepSeekThinking.clear_thinking`，对齐 GLM 官方文档 + CodeWhale |

---

## 十、实施步骤

1. 创建分支 `feature/glm-5.2-support`
2. 应用 7 处改动（6 处代码 + 1 处测试适配）
3. `cargo test` 全部通过
4. `cargo build --release`
5. 提交 + 推送 GitHub
6. 替换运行中二进制 + 重启服务
7. 通过 proxy 发送 GLM-5.2 请求验证
8. 检查 KV cache 日志确认 `prompt_tokens_details.cached_tokens` 正常上报
9. 检查 `clear_thinking: false` 是否生效（多轮推理一致性）