# codewhale-proxy 最终代码评审报告

> 评审日期：2026-06-06
> 评审方式：逐文件对照 CodeWhale 源码（chat.rs + client.rs）
> 评审范围：全部 21 个 .rs 源文件，2246 行代码

---

## 评审结论

**发现 3 个 P0 阻塞项、7 个 P1 重要项、8 个 P2 建议项。建议修复 P0 项后再上线。**

| 级别 | 数量 | 说明 |
|------|------|------|
| P0 | 3 | 阻塞上线 — 运行时安全隐患 |
| P1 | 7 | 重要 — 功能缺失或正确性风险 |
| P2 | 8 | 建议 — 代码质量改进 |

---

## P0（阻塞上线）

### P0-1：SSE 流处理无超时保护 — 可能导致连接永久挂起

- **文件**：`src/sse/stream.rs:62`
- **问题**：`while let Some(chunk_result) = stream.next().await` 无超时包装。如果上游连接断开但 TCP 层面未关闭（如网络分区），此循环将永久阻塞，tokio task 永不退出。
- **CodeWhale 对照**：`chat.rs:19-60` 定义了 `DEFAULT_STREAM_IDLE_TIMEOUT`（300s）和 `DEFAULT_STREAM_OPEN_TIMEOUT`（45s），`L330` 使用 `tokio::time::timeout(idle, byte_stream.next()).await` 包装每次读取。
- **影响**：生产环境长期运行下，tokio task 泄漏会累积，最终耗尽内存。无超时也意味着客户端断开后上游连接不释放。
- **修复建议**：
  ```rust
  // 在 process_stream 的 while 循环外定义
  let idle_timeout = Duration::from_secs(300);
  // 每次读取时包装
  let chunk_result = tokio::time::timeout(idle_timeout, stream.next()).await;
  ```

### P0-2：SSE 行缓冲区无上限 — 可能导致内存耗尽

- **文件**：`src/sse/stream.rs:59,66`
- **问题**：`let mut buffer = String::new()` 无限制增长。恶意或损坏的 SSE 流可以发送永不换行的数据，导致 `buffer.push_str()` 无限增长，内存耗尽。
- **CodeWhale 对照**：`chat.rs:372-375` 有 `MAX_SSE_BUF` 检查，超过限制时 `yield Err(anyhow::anyhow!("SSE buffer exceeded ..."))` 中止流。
- **影响**：DoS 攻击面，恶意上游可耗尽代理内存。
- **修复建议**：添加 `const MAX_SSE_BUF: usize = 4 * 1024 * 1024;` 并在 `push_str` 后检查 `buffer.len() > MAX_SSE_BUF`。

### P0-3：错误响应暴露上游原始错误体 — API Key 泄露风险

- **文件**：`src/client.rs:69-76`、`src/routes/messages.rs:52-61`
- **问题**：`client.rs` 中 `chat_completion_stream` 在 HTTP 非 200 时读取 `response.json()` 并直接返回。如果上游返回包含 API key 的详细错误信息，代理会将其原样透传给客户端。`messages.rs` 中 `INTERNAL_SERVER_ERROR` 响应也直接包含 `format!("Upstream error: {}", e)`。
- **CodeWhale 对照**：`client.rs:1084` 有 `sanitize_http_error_body` 函数对错误体进行脱敏；`chat.rs:165` 有 `bounded_error_text` 限制错误体长度。
- **影响**：安全漏洞 — 如果上游 API key 出现在错误响应中，攻击者可通过触发错误获取。
- **修复建议**：
  1. 限制错误响应体大小（如 1024 字节）
  2. 对错误体进行正则脱敏（移除 `Authorization`、`api_key` 等模式）
  3. 使用通用错误消息而非透传上游错误

---

## P1（重要）

### P1-1：build_chat_messages 缺失 tool_result 去重/压缩逻辑

- **文件**：`src/reasoning/build_messages.rs`
- **CodeWhale 对照**：`chat.rs:1357-1696` 的 `build_chat_messages_with_reasoning` 包含：
  - `pending_tool_calls: HashMap<String, PendingToolCallInfo>` 跟踪工具调用状态
  - `seen_tool_results: HashMap<String, SeenToolResult>` 去重
  - `compact_tool_result_for_wire` 压缩大型工具结果
  - `_tool_result_budget` 元数据传递上下文预算信息
  - `last_full_turn_meta` 和 `turn_meta_budget` 上下文窗口管理
- **代理实现**：简化版只做基本转换，无去重/压缩/预算管理。
- **影响**：多轮工具调用场景下，重复的工具结果会导致上下文窗口快速膨胀，KV cache 命中率降低。
- **严重度**：P1 — 当前功能可用但生产环境效率差。

### P1-2：cleanup_orphan_tool_calls 逻辑过于简化

- **文件**：`src/reasoning/build_messages.rs:284-308`
- **CodeWhale 对照**：`chat.rs:1587-1693` 的实现：
  - 收集 assistant 消息中所有 tool_call ID 到 `expected_ids: HashSet`
  - 扫描后续消息（连续和非连续）收集 `found_ids`
  - 检查 `expected_ids ⊆ found_ids`，不满足时：
    - 移除 assistant 的 tool_calls
    - 如果 assistant 无 content，移除整个 assistant 消息
    - 移除所有关联的 orphan tool result 消息
- **代理实现**：仅检查下一条消息的 role 是否为 "tool"
- **影响**：如果 tool_results 被压缩移除（与 P1-1 相关），或消息顺序因任何原因有间隙，代理会保留 orphan tool_calls 导致 DeepSeek 返回 400 错误。
- **严重度**：P1 — 特定场景下导致请求失败。

### P1-3：sanitize_thinking_mode_messages 缺失 provider 前置检查

- **文件**：`src/reasoning/sanitize.rs:6`
- **CodeWhale 对照**：`chat.rs:1768-1776` 开头有：
  ```rust
  if !should_replay_reasoning_content_for_provider(provider, model, effort) {
      return None;
  }
  ```
  `should_replay_reasoning_content_for_provider` 检查 `provider_accepts_reasoning_content(provider) || requires_reasoning_content(model)`。
- **代理实现**：无条件对所有消息执行 sanitize，无论模型/提供商。
- **影响**：对非 DeepSeek 模型添加 `reasoning_content` 字段可能导致 API 拒绝请求（某些提供商严格校验字段）。
- **严重度**：P1 — 当前仅连接 DeepSeek 时无影响，但扩展性差。

### P1-4：apply_reasoning_effort 缺失多数 provider 支持

- **文件**：`src/reasoning/apply_effort.rs`（定义但未使用）、`src/anthropic/converter.rs:77-104`（内联版本）
- **CodeWhale 对照**：`client.rs:1103-1261` 处理 15+ 个 provider（Deepseek、DeepseekCN、Openrouter、XiaomiMimo、Novita、Siliconflow、Sglang、Volcengine、Fireworks、Vllm、Openai、Atlascloud、WanjieArk、Arcee、Huggingface、Moonshot、Ollama、NvidiaNim），每个有不同的 reasoning_effort 语义。
- **代理实现**：仅处理 "deepseek" 和 "openrouter" 两个字符串 provider。
- **影响**：当前仅连接单一 DeepSeek 后端，无实际影响。但代码标注为"通用代理"，功能不完整。
- **严重度**：P1 — 功能声明与实现不匹配，扩展性受限。

### P1-5：ContentBlock::Image 类型定义与 Anthropic API 不一致

- **文件**：`src/anthropic/types.rs:76-79`
- **CodeWhale 对照**：`models.rs:91-92` 使用 `ContentBlock::ImageUrl { image_url: ImageUrlContent }`，`ImageUrlContent` 有 `url` 字段（直接是 URL 字符串）。
- **代理实现**：`ContentBlock::Image { source: ImageSource }`，`ImageSource` 有 `source_type`、`media_type`、`data` 字段（base64 数据）。
- **影响**：Anthropic 客户端发送的图片格式是 `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`。代理的 `ContentBlock::Image { source: ImageSource }` 可以正确反序列化此格式。但 CodeWhale 使用的是 `image_url` 格式（`{"type": "image_url", "image_url": {"url": "..."}}`），这是 Anthropic 协议到 OpenAI 协议的中间转换格式。代理直接接收 Anthropic 客户端请求，应该使用 `Image { source }` 格式。**经核实，代理的实现是正确的**，它接收的是 Anthropic 原生格式而非 CodeWhale 内部转换后的格式。
- **修正**：此项降级为 P2（信息性注释）。

### P1-6：SSE content_block_start 中 tool_use 的 id/name 字段赋值错误

- **文件**：`src/openai/converter.rs:275-282`
- **问题**：
  ```rust
  events.push(SseEvent::ContentBlockStart {
      index: new_idx,
      content_block: ContentBlockStartData::ToolUse {
          id: tool_name,   // ← 错误：这里应该是 call_id
          name: String::new(),  // ← 错误：这里应该是 tool_name
          input: Value::Object(serde_json::Map::new()),
      },
  });
  ```
  `tool_name` 变量实际上是 `tc.id`（工具调用 ID，如 "call_xxx"），而非工具名称（如 "read_file"）。这导致 Anthropic 客户端收到的 `content_block_start` 事件中 `id` 和 `name` 字段值互换。
- **CodeWhale 对照**：`chat.rs:2378-2403` 正确地从 `tc.get("function").and_then(|f| f.get("name"))` 获取 `name`，从 `tc.get("id")` 获取 `id`。
- **影响**：Anthropic 客户端可能无法正确路由工具调用结果，导致工具调用失败。
- **严重度**：P1 — 影响工具调用功能正确性。

### P1-7：缺失 `reasoning` 字段回退 — 非 DeepSeek 推理字段丢失

- **文件**：`src/sse/stream.rs:108`、`src/openai/converter.rs:152`
- **CodeWhale 对照**：`chat.rs:2000-2005` 有 `reasoning_field()` 函数同时检查 `reasoning_content` 和 `reasoning` 字段：
  ```rust
  fn reasoning_field(value: &Value) -> Option<&str> {
      value.get("reasoning_content")
          .or_else(|| value.get("reasoning"))
          .and_then(Value::as_str)
  }
  ```
- **代理实现**：仅检查 `reasoning_content`，不检查 `reasoning` 字段。
- **影响**：某些 OpenAI-compatible 提供商（如 vLLM 托管的 Qwen3）使用 `reasoning` 而非 `reasoning_content` 字段。代理会丢失这些推理内容，将其错误地渲染为普通文本。
- **严重度**：P1 — 当前仅连接 DeepSeek 时无影响，但扩展性受限。

---

## P2（建议）

### P2-1：两个 reasoning 模块定义但未使用

- **文件**：`src/reasoning/apply_effort.rs`、`src/reasoning/should_replay.rs`
- **问题**：`apply_reasoning_effort` 和 `should_replay_reasoning_content` 有独立模块和测试，但 `anthropic/converter.rs` 中内联了等效逻辑（`apply_effort_direct`），独立模块从未被调用。
- **建议**：统一使用一个实现，删除冗余代码。

### P2-2：缺失 Ping SSE 事件

- **文件**：`src/anthropic/types.rs:200`（SseEvent 枚举）、`src/sse/stream.rs`
- **CodeWhale 对照**：`models.rs:427-428` 有 `StreamEvent::Ping` 变体。
- **影响**：Anthropic 协议要求在长时间无数据时发送 ping 保持连接。代理不发送 ping，可能导致某些客户端超时断开。
- **建议**：在 SSE 流中添加定期 ping（如每 15 秒）。

### P2-3：缺失 `caller` 元数据传递

- **文件**：`src/anthropic/types.rs:63-67`（`ContentBlock::ToolUse`）、`src/reasoning/build_messages.rs:234-241`
- **CodeWhale 对照**：`models.rs:96-102` 的 `ToolUse` 有 `caller: Option<ToolCaller>` 字段，`chat.rs:1425-1429` 在构建 OpenAI 消息时传递 `caller` 元数据。
- **影响**：子代理调用场景下，caller 信息丢失，影响工具调用链追踪。
- **建议**：在 `ContentBlock::ToolUse` 和 `ContentBlockStartData::ToolUse` 中添加 `caller` 字段。

### P2-4：缺失 `ServerToolUse`、`ToolSearchToolResult`、`CodeExecutionToolResult` 类型

- **文件**：`src/anthropic/types.rs`
- **CodeWhale 对照**：`models.rs:112-128` 有 `ServerToolUse`、`ToolSearchToolResult`、`CodeExecutionToolResult` 三个变体。
- **影响**：服务端工具调用（如 code execution）无法被代理正确识别和转发。
- **建议**：添加这些类型，至少在转换时静默跳过而非 panic。

### P2-5：缺失 `cache_control` 支持

- **文件**：`src/anthropic/types.rs`（ContentBlock、Tool）、`src/reasoning/build_messages.rs`
- **CodeWhale 对照**：`models.rs:88-89` 的 `ContentBlock::Text` 有 `cache_control: Option<CacheControl>`，`Tool` 也有 `cache_control`。
- **影响**：KV cache 优化提示无法传递，降低缓存命中率。
- **建议**：添加 `cache_control` 字段并透传。

### P2-6：`config.rs:8` log_level 字段未使用

- **文件**：`src/config.rs:8`
- **问题**：`log_level` 从环境变量读取但 `main.rs:16` 直接从 `EnvFilter::try_from_default_env()` 读取，config 中的字段未被使用。
- **建议**：删除或使用它。

### P2-7：缺失成功请求日志

- **文件**：`src/routes/messages.rs`
- **问题**：只有错误日志（`tracing::error!`），无成功请求的 info/debug 日志。生产环境无法追踪请求量。
- **建议**：添加 `tracing::info!` 或 `tracing::debug!` 记录成功请求（模型、流/非流、耗时）。

### P2-8：`models.rs` 硬编码模型列表

- **文件**：`src/routes/models.rs:7-41`
- **问题**：模型列表硬编码，无法动态反映上游实际可用模型。
- **建议**：从上游 `/v1/models` 端点获取模型列表并缓存，或至少从环境变量/配置文件加载。

---

## 逐文件对照审查详情

### 1. `src/reasoning/build_messages.rs` ↔ `chat.rs:1357-1696`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| system prompt 处理 | `system_to_instructions` + `\n\n---\n\n` 分隔 | `system_prompt_to_text` + `\n` 分隔 | ⚠️ 分隔符不同 |
| turn_meta 处理 | `is_turn_meta_text` + `render_turn_meta_for_wire` | 无 | ❌ 缺失 |
| tool_result 去重/压缩 | `compact_tool_result_for_wire` + `seen_tool_results` | 无 | ❌ 缺失 (P1-1) |
| pending_tool_calls 跟踪 | `HashMap<String, PendingToolCallInfo>` | 无 | ❌ 缺失 |
| tool_result 预算元数据 | `_tool_result_budget` 字段 | 无 | ❌ 缺失 |
| orphan cleanup | 完整 ID 集合匹配 + 多级清理 | 仅检查下一条 role | ❌ 简化 (P1-2) |
| 空 assistant 消息处理 | 跳过无 content 且无 tool_calls 的消息 | 正确跳过 | ✅ |
| reasoning_content 占位符 | `"(reasoning omitted)"` | `"(reasoning omitted)"` | ✅ |
| 图片转换 | `ImageUrl { image_url }` → `image_url` | `Image { source }` → data URL | ✅ 正确（Anthropic 原生格式） |
| 工具调用转换 | `to_api_tool_name` 编码 | 直接透传 | ⚠️ 无编码 |

### 2. `src/reasoning/sanitize.rs` ↔ `chat.rs:1768-1820`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| provider 前置检查 | `should_replay_reasoning_content_for_provider` | 无 | ❌ 缺失 (P1-3) |
| 返回值 | `Option<u32>`（推理 token 估算） | 无返回值 | ⚠️ 缺失诊断信息 |
| null reasoning_content | 检查并替换 | 检查并替换 | ✅ |
| 空字符串 reasoning_content | 检查并替换 | 检查并替换 | ✅ |
| 替换计数和日志 | `substitutions` 计数 + `logging::warn` | 无计数 | ⚠️ 缺失 |

### 3. `src/reasoning/requires.rs` ↔ `chat.rs:1897-1913`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| deepseek-v4 检测 | `lower.contains("deepseek-v4")` | `model_lower.starts_with("deepseek-v4")` | ⚠️ 语义不同 |
| deepseek-chat/reasoner | `lower.starts_with("deepseek-chat")` | `model_lower == "deepseek-chat"` | ⚠️ 使用了 `==` 而非 `starts_with` |
| 通用 markers | `reasoner`/`-reasoning`/`-thinking` | 相同 | ✅ |
| R-series 检测 | `match_indices("deepseek-r")` | `strip_prefix("deepseek")` + 手动解析 | ✅ 逻辑等价 |

**差异说明**：
- CodeWhale 使用 `contains("deepseek-v4")`，代理使用 `starts_with("deepseek-v4")`。如果模型名如 `openrouter/deepseek-v4-pro`，CodeWhale 会匹配，代理不会。**但**代理作为中间层且自己提供模型名，startswith 更精确。
- CodeWhale 使用 `starts_with("deepseek-chat")` 匹配 `deepseek-chat-xxx` 等变体，代理用 `==` 精确匹配，可能遗漏变体。

### 4. `src/reasoning/should_replay.rs` ↔ `chat.rs:1915-1929`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| effort 规范化 | `value.trim().to_ascii_lowercase()` | `eff.to_lowercase()` | ⚠️ 缺少 trim |
| 禁用关键词 | `off`/`disabled`/`none`/`false` | 相同 | ✅ |
| 启用检查 | `requires_reasoning_content(model)` | 相同 | ✅ |

### 5. `src/reasoning/apply_effort.rs` ↔ `client.rs:1103-1261`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| 参数 | `effort: Option<&str>, provider: ApiProvider` | `effort: Option<&str>, provider: &str` | ⚠️ |
| provider 分支 | 15+ providers | 仅 deepseek/openrouter | ❌ 缺失 (P1-4) |
| effort 规范化 | `effort.trim().to_ascii_lowercase()` | `e.to_lowercase()` + `effort.trim()` | ✅ |
| "off" 处理 | 移除 reasoning_effort + 设置 thinking:disabled | 相同 | ✅ |
| "low"/"medium"→"high" | DeepSeek 映射 | 相同 | ✅ |
| "max"/"xhigh" | DeepSeek→max, OpenRouter→xhigh | 相同 | ✅ |
| 未知值处理 | 无操作 (`_ => {}`) | 设 high + enabled | ⚠️ 行为不同 |

### 6. `src/openai/converter.rs` (SseStateMachine) ↔ `chat.rs:2231-2466` (parse_sse_chunk)

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| usage-only chunk | 有 MessageDelta 事件 | 仅更新 input_tokens | ⚠️ 行为不同 |
| reasoning 字段回退 | `reasoning_field()` 检查两种字段 | 仅 `reasoning_content` | ❌ 缺失 (P1-7) |
| text→thinking 切换 | 正确关闭 thinking block | 正确关闭 | ✅ |
| thinking→text 切换 | 正确关闭 thinking block | 正确关闭 | ✅ |
| tool_use start id/name | 正确从 function.name 获取 | **id/name 互换** | ❌ 错误 (P1-6) |
| signature_delta | 不存在（CodeWhale 无此事件） | 代理自行添加 | ✅ 代理增强 |
| 非 reasoning 模型 content 回退 | `effective_content` 逻辑 | 不处理 | ⚠️ 缺失 |
| tool_name_or_fallback | 有 | 硬编码 "unknown" | ⚠️ 简化 |
| caller 传递 | 有 | 无 | ❌ 缺失 (P2-3) |
| 多个 tool_calls 索引 | 每个 call 独立 track | 每个 call 独立 track | ✅ |

### 7. `src/sse/stream.rs` ↔ `chat.rs:287-450`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| 流读取超时 | `tokio_timeout(idle, ...)` | 无 | ❌ 缺失 (P0-1) |
| 缓冲区大小限制 | `MAX_SSE_BUF` 检查 | 无 | ❌ 缺失 (P0-2) |
| 背压管理 | `SSE_BACKPRESSURE` + sleep | 无 | ⚠️ 缺失 |
| 流缓冲区复用 | `acquire/release_stream_buffer` | 每次新建 String | ⚠️ 性能差异 |
| 连接打开超时 | `stream_open_timeout()` | 无 | ⚠️ 缺失 |
| 传输头诊断 | `format_stream_headers` | 无 | ⚠️ 缺失 |
| 字节统计 | `total_bytes` + `stream_start` | 无 | ⚠️ 缺失 |
| Ping 事件 | `StreamEvent::Ping` | 无 | ❌ 缺失 (P2-2) |

### 8. `src/anthropic/types.rs` ↔ `models.rs`

| 类型 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| ContentBlock::Text | 有 `cache_control` | 无 | ⚠️ (P2-5) |
| ContentBlock::Image/ImageUrl | `ImageUrl { image_url }` | `Image { source }` | ✅ 正确（Anthropic 原生格式） |
| ContentBlock::Thinking | 无 `signature` | 有 `signature` | ✅ 代理正确添加 |
| ContentBlock::ToolUse | 有 `caller` | 无 | ⚠️ (P2-3) |
| ContentBlock::ToolResult | `content: String` | `content: ToolResultContent` | ✅ 代理更完整 |
| ServerToolUse | 有 | 无 | ⚠️ (P2-4) |
| ToolSearchToolResult | 有 | 无 | ⚠️ (P2-4) |
| CodeExecutionToolResult | 有 | 无 | ⚠️ (P2-4) |
| ContentBlockStart::ToolUse | 有 `caller` | 无 | ⚠️ (P2-3) |
| SseEvent::Ping | 有 | 无 | ❌ (P2-2) |
| MessageStartData | `message: MessageResponse` | `message: MessageStartData`（独立结构） | ✅ 功能等价 |

### 9. `src/client.rs` ↔ `client.rs` + `chat.rs`

| 功能 | CodeWhale | 代理 | 状态 |
|------|-----------|------|------|
| 错误体脱敏 | `sanitize_http_error_body` | 无 | ❌ (P0-3) |
| 错误体大小限制 | `bounded_error_text` | 无 | ❌ (P0-3) |
| 重试逻辑 | `send_with_retry` | 无 | ⚠️ 缺失 |
| path_suffix 支持 | 有 | 无 | ⚠️ 缺失 |
| 健康检查 | `/v1/models` | `/v1/models` | ✅ |
| 签名生成 | `Sha256` 签名 | 无 | ⚠️ 缺失（已知限制） |

### 10. `src/anthropic/converter.rs`

| 功能 | 正确性 | 说明 |
|------|--------|------|
| 请求转换 | ✅ | 基本转换正确 |
| thinking budget 映射 | ✅ | budget≥4096→max, 否则→high |
| Adaptive thinking | ✅ | 正确识别为 enabled |
| 工具转换 | ✅ | 正确映射 Anthropic→OpenAI 格式 |
| tool_choice 转换 | ✅ | auto/any/tool 正确映射 |
| sanitize 调用 | ✅ | 在转换后正确调用 |

---

## 测试覆盖评估

| 测试文件 | 测试数 | 覆盖场景 |
|----------|--------|----------|
| `build_messages.rs` | 4 | 简单用户消息、assistant+thinking、tool_calls 占位符、system prompt |
| `sanitize.rs` | 4 | 占位符添加、空字符串、有值保持、null handling |
| `apply_effort.rs` | 4 | deepseek max、off、low→high、None |
| `should_replay.rs` | 3 | 回放、off 禁用、非 DS 模型 |
| `requires.rs` | 4 | V4、R-series、非推理模型、通用 markers |
| `converter.rs` (anthropic) | 3 | 基本转换、thinking enabled、adaptive |
| `converter.rs` (openai) | 1 | 非流式 thinking 响应 |

**缺失测试**：
- SSE 流状态机（无测试）
- SSE 流解析（无测试）
- 错误路径（无测试）
- 边界条件（最大消息数、空消息、畸形 JSON）
- tool_use content_block_start id/name 赋值（P1-6 的 bug 无测试覆盖）

---

## 已知待修复项（来自之前评审）

| 项 | 本评审确认 |
|----|-----------|
| 模型白名单缺失 | ✅ 确认 — 硬编码在 models.rs |
| SSE 超时未设置 | ✅ 确认 — P0-1 |
| 签名硬编码 | ✅ 确认 — "sig_proxy_placeholder" |
| 非 release build | ⚠️ 构建配置问题，非代码问题 |

---

## 附录：CodeWhale 源码行号映射

| 代理文件 | 对应 CodeWhale 源码 | 行号 |
|----------|---------------------|------|
| `reasoning/build_messages.rs` | `chat.rs:build_chat_messages_with_reasoning` | L1357-L1696 |
| `reasoning/sanitize.rs` | `chat.rs:sanitize_thinking_mode_messages` | L1768-L1820 |
| `reasoning/requires.rs` | `chat.rs:requires_reasoning_content` | L1897-L1913 |
| `reasoning/requires.rs` | `chat.rs:has_deepseek_r_series_marker` | L1990-L1998 |
| `reasoning/should_replay.rs` | `chat.rs:should_replay_reasoning_content` | L1915-L1929 |
| `reasoning/apply_effort.rs` | `client.rs:apply_reasoning_effort` | L1103-L1261 |
| `openai/converter.rs` (SseStateMachine) | `chat.rs:parse_sse_chunk` | L2231-L2466 |
| `openai/converter.rs` (convert_non_stream) | `chat.rs:parse_chat_message` | L2007+ |
| `sse/stream.rs` | `chat.rs:handle_chat_completion_stream` | L182-L450 |
| `anthropic/types.rs` | `models.rs` | L84-L649 |
| `client.rs` | `client.rs` + `chat.rs` | 多处 |
| `anthropic/converter.rs` | `chat.rs:build_chat_messages_for_request_and_provider` | 多处 |