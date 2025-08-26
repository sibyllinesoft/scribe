# PackRepo Hermetic Build Container
# Multi-stage build with security best practices and reproducible builds

ARG PYTHON_VERSION=3.12.7
ARG UV_VERSION=0.5.6

# ================================
# Base stage with common dependencies
# ================================
FROM python:${PYTHON_VERSION}-slim-bullseye AS base

# Install system security updates and essential packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gcc \
        g++ \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 packrepo \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash packrepo

# Install uv for fast Python dependency management
ARG UV_VERSION
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || cp /root/.cargo/bin/uv /usr/local/bin/uv

# ================================
# Development stage with all tools
# ================================
FROM base AS development

# Install development tools for static analysis and security scanning
RUN apt-get update && apt-get install -y --no-install-recommends \
        tree \
        jq \
        sqlite3 \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Switch to non-root user
USER packrepo
WORKDIR /app

# Copy lock files first for optimal Docker layer caching
COPY --chown=packrepo:packrepo uv.lock pyproject.toml ./

# Install Python dependencies with pinned versions
RUN uv sync --frozen --all-extras

# Copy source code
COPY --chown=packrepo:packrepo . .

# Record build environment information
RUN python -c "import sys,platform,hashlib,subprocess; print(f'Platform: {platform.platform()}\\nPython: {sys.version}\\nBuild timestamp: {__import__(\"datetime\").datetime.now().isoformat()}')" > /tmp/build_info.txt

# Health check for development container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import packrepo; print('OK')" || exit 1

# Default development command
CMD ["/bin/bash"]

# ================================
# Production stage - minimal runtime
# ================================
FROM base AS production

# Switch to non-root user
USER packrepo
WORKDIR /app

# Copy only production dependencies
COPY --chown=packrepo:packrepo --from=development /app/.venv /app/.venv
COPY --chown=packrepo:packrepo pyproject.toml ./

# Copy only necessary source files for production
COPY --chown=packrepo:packrepo packrepo ./packrepo/
COPY --chown=packrepo:packrepo rendergit.py ./
COPY --chown=packrepo:packrepo README.md ./

# Create artifacts directory
RUN mkdir -p artifacts logs

# Health check for production container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD /app/.venv/bin/python -c "import packrepo.library; print('Healthy')" || exit 1

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default production command
ENTRYPOINT ["/app/.venv/bin/python", "rendergit.py"]

# ================================
# CI stage for testing and validation
# ================================
FROM development AS ci

# Install additional CI/testing tools
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        docker.io \
        docker-compose \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER packrepo

# Install Python testing and security tools
RUN uv add --dev \
        pytest \
        pytest-cov \
        pytest-xdist \
        mypy \
        ruff \
        bandit[toml] \
        safety \
        semgrep

# Copy test suites and scripts
COPY --chown=packrepo:packrepo tests ./tests/
COPY --chown=packrepo:packrepo scripts ./scripts/
COPY --chown=packrepo:packrepo spec ./spec/

# Make scripts executable
RUN find scripts -name "*.sh" -exec chmod +x {} \;

# Health check includes tool availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/.venv/bin/python -c "import pytest,mypy,ruff; print('CI Tools Ready')" || exit 1

# Default CI command runs full test suite
CMD ["scripts/ci_full_test.sh"]

# ================================
# Metadata and labels
# ================================
LABEL org.opencontainers.image.title="PackRepo"
LABEL org.opencontainers.image.description="Sophisticated Repository Packing System with Hermetic Builds"
LABEL org.opencontainers.image.source="https://github.com/nathanrice/rendergit"
LABEL org.opencontainers.image.documentation="https://github.com/nathanrice/rendergit/blob/main/PACKREPO_README.md"
LABEL org.opencontainers.image.licenses="0BSD"