# Multi-stage build for codewhale-proxy
# Stage 1: Build
FROM rust:1.86-slim AS builder

WORKDIR /app

# Copy dependency manifests first for caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source
COPY src/ src/

# Build the release binary (touch main.rs to invalidate the dummy cache)
RUN touch src/main.rs && cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/codewhale-proxy /usr/local/bin/

EXPOSE 11435

ENV LISTEN_ADDR="0.0.0.0:11435"
ENV ESWITCH_URL="http://127.0.0.1:11434"
ENV DEEPSEEK_API_KEY="not-needed"
ENV RUST_LOG="info"

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11435/health || exit 1

ENTRYPOINT ["/usr/local/bin/codewhale-proxy"]
