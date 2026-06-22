# codewhale-proxy 深度分析：thinking & KV-cache

**日期**: 2026-06-06
**分析对象**: 当前运行中的 codewhale-proxy (Rust, 11435端口)
**对比参考**: CodeWhale (Hmbown/CodeWhale) + DeepSeek-Reasonix (esengine/deepseek-reasonix)
**源分析报告**: /home/Projecrt/deepseek-v4-pro-adapter-analysis.md

---

## 0. 首先：分析报告与当前代理的关系

**分析报告分析的是旧代理，不是当前运行的代理。**

| 维度 | 分析报告描述 | 当前实际 |
|------|-------------|---------|
| 代理 | `anthropic-proxy.js` (Node.js, 581行) | `codewhale-proxy` (Rust, 2246行) |
| 端口 | 11435 (anthropic-proxy.js) | 11435 (codewhale-proxy) |
| 架构 | 三层链: CCR→anthropic-proxy.js→CodeWhale | 直接: cc-connect→codewhale-proxy→eswitch→DeepSeek |
| thinking 参数 | ❌ 报告说丢失 | ✅ 实际正确传递 |
| reasoning_content | ❌ 报告说消失 | ✅ 实际正确转换 |

报告分析的 `anthropic-proxy.js` 是旧架构的中间层，已于 6/6 凌晨被替换为当前的 Rust `codewhale-proxy`。

---

## 1. thinking/reasoning_effort 分析

### 1.1 当前代理实现 (converter.rs:82-92)

```rust
if let Some(thinking) = &req.thinking {
    if thinking.is_enabled() {
        let budget = thinking.budget_tokens().unwrap_or(0);
        let effort = if budget >= 4096 { "max" } else { "high" };
        apply_effort_direct(&mut openai_req, effort);
    } else {
        apply_effort_direct(&mut openai_req, "off");
    }
} else if is_reasoning_model {
    apply_effort_direct(&mut openai_req, "high");
}
```

`apply_effort_direct` (L102-129):
```rust
"max" | "xhigh" => {
    req.reasoning_effort = Some("max");
    req.thinking = Some(DeepSeekThinking { thinking_type: "enabled" });
}
"low" | "medium" | "high" => {
    req.reasoning_effort = Some("high");
    req.thinking = Some(DeepSeekThinking { thinking_type: "enabled" });
}
"off" => {
    req.thinking = Some(DeepSeekThinking { thinking_type: "disabled" });
    req.reasoning_effort = None;
}
```

### 1.2 CodeWhale 参考实现 (client.rs:1103-1261)

```rust
// DeepSeek 分支:
"max" → body["reasoning_effort"] = "max"; body["thinking"] = {"type": "enabled"}
"high" → body["reasoning_effort"] = "high"; body["thinking"] = {"type": "enabled"}
"low"/"medium" → 兼容映射为 "high"
"off" → body["thinking"] = {"type": "disabled"}
```

### 1.3 Reasonix 参考实现 (openai.go:buildRequest)

```go
if c.deepseek {
    out.Thinking = &thinkingMode{Type: "enabled"}
    out.ReasoningEffort = &effort  // "high" 或 "max"
}
```

### 1.4 结论：✅ 当前代理的 thinking 处理正确

- budget_tokens ≥ 4096 → `reasoning_effort: "max"` + `thinking: {type: "enabled"}` — 与 CodeWhale 和 Reasonix 一致
- budget_tokens < 4096 → `reasoning_effort: "high"` + `thinking: {type: "enabled"}` — 正确
- thinking disabled → `thinking: {type: "disabled"}` — 正确
- 无 thinking 配置但推理模型 → 默认 `reasoning_effort: "high"` — 合理

**唯一值得讨论的差异**：budget 阈值。当前代理用 4096 作为 max/high 的分界线，而分析报告建议 16000。CodeWhale 没有固定的 budget→effort 映射（它是客户端，用户自己控制 effort）。DeepSeek 官方文档中 `"max"` 和 `"high"` 的区别在于推理深度，没有明确的 token 阈值。当前 4096 阈值偏保守，但功能正确。

---

## 2. reasoning_content 响应分析

### 2.1 非流式 (converter.rs:20-29)

```rust
// DeepSeek reasoning_content → Anthropic Thinking block
if let Some(ref rc) = msg.reasoning_content {
    content.push(ResponseContentBlock::Thinking {
        thinking: rc.clone(),
        signature: "sig_proxy_placeholder".to_string(),
    });
}
```

**结果：Claude Code 可以看到 DeepSeek 的思考过程。**

### 2.2 流式 (converter.rs:152-181)

```rust
// 首次收到 reasoning_content → 开始 Thinking 块
ContentBlockStart { content_block: Thinking { ... } }
// 后续 delta → 流式输出
ContentBlockDelta { delta: ThinkingDelta { thinking: rc } }
// 思考结束 → 关闭块，开启 Text 块
ContentBlockStop → ContentBlockStart(Text)
```

**代码证据**：`converter.rs` L152-181 正确实现了 `Thinking → Text` 的块切换，与 CodeWhale `chat.rs:2231-2466` 一致。

### 2.3 结论：✅ 当前代理正确输出 reasoning_content

**分析报告说"reasoning_content 在响应中消失"——这是针对旧 `anthropic-proxy.js` 的，不适用于当前代理。** 当前 Rust 代理在流式和非流式两种模式下都正确地将 DeepSeek 的 `reasoning_content` 转换为 Anthropic 的 `Thinking` 块。

---

## 3. 多轮对话 reasoning_content 回放 & KV-cache 分析

### 3.1 核心机制

这是最关键的问题。当 DeepSeek 返回 reasoning_content 后，下一轮请求中如何处理？

**当前代理 (build_messages.rs:248-300)**：
```rust
// 从 Claude Code 的 Thinking 块提取 reasoning
ContentBlock::Thinking { thinking, signature: _ } => {
    thinking_parts.push(thinking.clone());
}

// 构建 OpenAI 消息
let reasoning_content = if has_reasoning {
    Some(thinking_text)          // 有实际推理内容 → 原样回放
} else if has_tool_calls {
    Some("(reasoning omitted)")  // 有 tool_calls 无推理 → 占位符
} else {
    None                          // 纯文本无推理 → 省略
};
```

### 3.2 CodeWhale 的设计原则 (chat.rs:1459-1468)

**这是关键注释，直接回答了你的问题：**

```rust
// Reasoning replay must be a function of the stored message ONLY,
// never of later history. DeepSeek's prefix cache hashes the raw
// bytes of every message; flipping `reasoning_content` on/off
// depending on whether a follow-up user turn exists rewrites a
// historical message between turns and busts the cache from that
// point onwards. Always emit `reasoning_content` when the model
// requires replay AND the stored message carries thinking text.
// Tool-call messages with empty thinking still need a placeholder
// (DeepSeek 400s without it), but text-only assistant messages
// simply omit the field when there's nothing to replay.
```

**翻译**：
1. `reasoning_content` 必须是消息本身的函数，**绝不能依赖后续历史**
2. 如果同一消息在两轮之间 `reasoning_content` 有/无变化 → 字节序列不同 → **KV cache 从该点完全失效**
3. 有推理内容 → 始终回放
4. 有 tool_calls 无推理 → 必须占位符（否则 DeepSeek 400）
5. 纯文本无推理 → 省略字段

### 3.3 当前代理与 CodeWhale 的一致性

| 场景 | CodeWhale | 当前代理 | 一致？ |
|------|-----------|---------|--------|
| 有 thinking 块 | 原样回放 | 原样回放 | ✅ |
| tool_calls 无 thinking | `"(reasoning omitted)"` | `"(reasoning omitted)"` | ✅ |
| 纯文本无 thinking | 省略字段 | 省略字段 | ✅ |
| 占位符确定性 | 编译时常量 | 编译时常量 | ✅ |

### 3.4 Reasonix 的对比

Reasonix 采用 **"reasoning_content 零回传"** 策略：
```
从 API 接收的推理内容 → 仅用于本地展示/归档 → 绝不上传回 API
```

节省 ~500 tokens/turn。但这是**成本优化**，不是 KV cache 优化。原因：
- 不回传 → 消息字节更短 → 但**历史消息在不同轮次之间字节不同** → 缓存失效
- 回传 → 消息字节更长 → 但**历史消息字节完全一致** → 缓存命中

**Counter-intuitive 但正确：回传 reasoning_content 对 KV cache 更有利。**

### 3.5 结论：✅ 当前代理的 KV cache 策略正确

**回答你的三个问题：**

1. **"当前代理不再使用 SSE 输出思考过程"** — **不属实**。这是旧 `anthropic-proxy.js` 的问题。当前 Rust 代理在流式 SSE 中正确输出 `Thinking → ThinkingDelta → ContentBlockStop` 序列。

2. **"DeepSeek 最高强度思考需要将思考内容传入下一轮"** — **正确**。但不是因为"最高强度思考"需要，而是因为：
   - DeepSeek 要求有 tool_calls 的 assistant 消息必须有 `reasoning_content`（否则 400 错误）
   - 即使没有 tool_calls，回放 `reasoning_content` 可以**保持 KV cache 字节一致性**
   - 当前代理正确处理了这两种情况

3. **"没有思考内容作为输入是否会影响 KV cache"** — **会，但当前代理处理正确**。关键不是"有没有"思考内容，而是**同一消息的 reasoning_content 在前後两轮是否一致**。当前代理：
   - 有 thinking 块 → 始终回放 → 一致
   - 无 thinking 有 tool_calls → 始终用 `"(reasoning omitted)"` 占位符 → 一致
   - 无 thinking 无 tool_calls → 始终省略 → 一致

---

## 4. 当前代理的 KV-cache 缺失项（对照 CodeWhale/Reasonix）

虽然核心策略正确，但与两个参考项目相比，当前代理缺少以下 KV-cache 优化：

| 缺失项 | CodeWhale | Reasonix | 当前代理 | 影响 |
|--------|-----------|----------|---------|------|
| Prefix 稳定性监控 | SHA-256 指纹 | CacheDiagnostics | ❌ 无 | 无法检测 cache bust |
| Tool schema 排序 | 按 name 排序后 SHA-256 | 按 name 排序后哈希 | ❌ 无 | 注册顺序变化→缓存失效 |
| JSON 属性顺序 | 固定顺序序列化 | 确定性序列化 | ❌ 无保证 | 属性顺序不同→缓存失效 |
| System prompt 冻结 | 整个 session 不变 | 整个 session 不变 | ❌ 无检查 | 可能跨轮变化 |
| cache_control 支持 | 透传 | 四层不变性 | ❌ 无 | 无法标记缓存边界 |

**这些是优化项，不是阻塞项。** 当前代理的核心 KV cache 策略（确定性 reasoning_content 回放）是正确的。

---

## 5. 总体结论

### 当前代理状态：✅ 功能正确，策略正确

| 维度 | 状态 | 证据 |
|------|------|------|
| thinking 参数传递 | ✅ 正确 | converter.rs:82-92，与 CodeWhale/Reasonix 一致 |
| reasoning_content 响应 | ✅ 正确 | converter.rs:20-29 + 152-181，流式+非流式 |
| 多轮 reasoning_content 回放 | ✅ 正确 | build_messages.rs:248-300，与 CodeWhale 原则一致 |
| reasoning_content 占位符 | ✅ 正确 | `"(reasoning omitted)"` 与 CodeWhale 一致 |
| KV cache 字节确定性 | ✅ 正确 | 三种场景确定性处理 |
| Prefix 稳定性监控 | ⚠️ 缺失 | 建议从 CodeWhale 复用 prefix_cache.rs |
| Tool schema 排序 | ⚠️ 缺失 | 建议添加 |
| JSON 属性顺序 | ⚠️ 缺失 | 建议固定序列化顺序 |

### 新旧代理对比

```
旧 (anthropic-proxy.js):         新 (codewhale-proxy):
  thinking 参数 → ❌ 丢失           ✅ 正确映射
  reasoning_content → ❌ 消失       ✅ 正确转换
  KV cache → ❌ 完全不可用          ✅ 策略正确（缺监控）
  架构 → 三层链，维护困难           ✅ 单层，Rust 编译型
```

### 建议的改进优先级

1. **P1**: 从 CodeWhale 复用 `prefix_cache.rs`，添加 SHA-256 前缀稳定性监控
2. **P2**: 添加 JSON 属性顺序固定 + tool schema 按名称排序
3. **P3**: 添加 `cache_control` 支持

这些都是锦上添花，当前代理的核心功能已经正确。