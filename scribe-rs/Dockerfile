# Multi-stage build for scribe
FROM rust:1.70-slim as builder

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY deny.toml ./

# Copy workspace crates
COPY scribe-core/ ./scribe-core/
COPY scribe-analysis/ ./scribe-analysis/
COPY scribe-graph/ ./scribe-graph/
COPY scribe-scanner/ ./scribe-scanner/
COPY scribe-patterns/ ./scribe-patterns/
COPY scribe-selection/ ./scribe-selection/
COPY scribe-scaling/ ./scribe-scaling/
COPY scribe-webservice/ ./scribe-webservice/

# Copy source files
COPY src/ ./src/
COPY test_budget_utilization.rs ./

# Build for release
RUN cargo build --release --bin scribe

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false scribe

# Copy the binary from builder stage
COPY --from=builder /app/target/release/scribe /usr/local/bin/scribe

# Make sure the binary is executable
RUN chmod +x /usr/local/bin/scribe

# Switch to app user
USER scribe

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/scribe"]
CMD ["--help"]