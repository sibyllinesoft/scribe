# PackRepo Infrastructure Documentation

This document describes the hermetic build and CI infrastructure established for PackRepo according to the TODO.md requirements.

## 🏗️ Infrastructure Overview

The PackRepo infrastructure provides:

- **Hermetic Builds**: Reproducible, containerized build environment
- **Multi-Stage Containers**: Development, CI, and production optimized images  
- **Local Development**: Docker Compose stack with health-gated dependencies
- **Quality Gates**: Static analysis, security scanning, and contract validation
- **Boot Verification**: Cryptographic verification of hermetic build integrity

## 📁 Directory Structure

```
PackRepo/
├── infra/
│   ├── Dockerfile              # Multi-stage container build
│   ├── compose.yaml           # Main development stack
│   └── compose.override.yaml  # Development enhancements
├── scripts/
│   ├── spinup_smoke.sh        # Hermetic boot verification
│   ├── sign_transcript.py     # Boot transcript signing
│   ├── pack_verify.py         # Pack validation oracles
│   ├── ci_full_test.sh       # Complete CI test suite
│   └── scan_secrets.py       # Secret detection
├── spec/
│   └── index.schema.json     # Pack format JSON schema
├── tests/
│   ├── properties/           # Property-based tests
│   ├── metamorphic/          # Metamorphic tests
│   ├── mutation/             # Mutation testing
│   ├── fuzzing/              # Fuzz testing
│   └── e2e/                  # End-to-end tests
├── artifacts/                # Build artifacts and reports
├── locks/                    # Dependency lockfiles
├── requirements.txt          # Pinned Python dependencies
└── Makefile                  # Convenient operations
```

## 🚀 Quick Start

### 1. Bootstrap Development Environment

```bash
# Initialize the environment
make bootstrap

# Or manually:
mkdir -p artifacts locks spec tests/{properties,metamorphic,mutation,fuzzing,e2e}
uv sync  # or pip install -r requirements.txt
```

### 2. Start Development Environment

```bash
# Start interactive development container
make dev

# Or with Docker Compose directly:
docker-compose -f infra/compose.yaml -f infra/compose.override.yaml up packrepo-dev
```

### 3. Run Hermetic Boot Verification

```bash
# Full hermetic boot test
make smoke

# Or with specific parameters:
./scripts/spinup_smoke.sh --repo https://github.com/karpathy/nanoGPT --budget 50000 --tokenizer cl100k --no-llm
```

### 4. Run Complete CI Pipeline

```bash
# Full CI pipeline
make ci

# Quick development tests
make quick-test
```

## 🐳 Container Architecture

### Development Container (`development` stage)
- Full development tools (ruff, mypy, pytest, etc.)
- Source code mounted for live editing
- Non-root user for security
- Health checks for service readiness

### CI Container (`ci` stage)
- Additional testing and security tools
- Semgrep, bandit, safety for security scanning
- Docker-in-Docker capability for integration tests
- Comprehensive tool availability validation

### Production Container (`production` stage)
- Minimal runtime dependencies
- Only necessary source files
- Optimized for deployment size and security
- Runtime health monitoring

## 🔐 Security Features

### Container Security
- Non-root user execution
- Minimal base images
- Security updates in base layer
- No hardcoded secrets

### Secret Scanning
- Automated secret detection in codebase
- Pattern-based detection for API keys, passwords, tokens
- Filename-based suspicious file detection
- CI integration for continuous monitoring

### Static Analysis
- **Ruff**: Fast Python linting and formatting
- **MyPy**: Static type checking  
- **Bandit**: Security vulnerability detection
- **Semgrep**: Advanced static analysis
- **Safety**: Dependency vulnerability scanning

## 📋 Quality Gates

### Schema Validation
- JSON Schema for pack format validation
- Runtime contract enforcement
- Budget constraint verification
- Chunk overlap detection

### Hermetic Verification
- Environment reproducibility verification
- Dependency lock validation
- Container digest verification
- Boot transcript cryptographic signing

### Testing Framework
- Unit tests with pytest
- Property-based testing (planned)
- Metamorphic testing (planned)
- Mutation testing (planned)
- Fuzz testing (planned)

## 🔄 CI/CD Workflows

### Hermetic Boot Process
1. **Environment Validation**: Check required tools and Git state
2. **Container Build**: Multi-stage Docker build with caching
3. **Dependency Verification**: Validate and lock dependencies
4. **Core Functionality**: Test imports and basic operations
5. **Golden Smoke Tests**: Run actual PackRepo operations
6. **Transcript Signing**: Cryptographic verification of boot integrity

### Quality Pipeline
1. **Static Analysis**: Linting, type checking, complexity analysis
2. **Security Scanning**: Secret detection, vulnerability scanning
3. **Unit Testing**: Core functionality validation
4. **Integration Testing**: Service interaction validation
5. **Pack Verification**: Schema and contract validation

## 📊 Monitoring and Artifacts

### Boot Transcripts
- Comprehensive build environment recording
- Cryptographic signing for tamper detection
- Git state and container digest recording
- Phase timing and error tracking

### Artifacts Generated
- `boot_transcript.json`: Signed boot verification record
- `boot_env.txt`: Environment snapshot
- `container_metadata.json`: Container build information
- Security scan reports (JSON format)
- Test results and coverage reports

## 🛠️ Development Workflow

### Day-to-Day Development
```bash
# Start development environment
make dev

# In container:
python rendergit.py --help
scripts/pack_verify.py --help
scripts/ci_full_test.sh

# Run specific tests
make test-unit
make lint
make typecheck
```

### Pre-Commit Workflow
```bash
# Run quality checks
make lint
make typecheck
make security

# Run tests
make test

# Verify pack format
make verify
```

### Release Workflow
```bash
# Complete validation
make ci

# Generate schemas
make schema

# Run hermetic verification
make smoke
```

## 🎯 Success Criteria Achievement

✅ **Hermetic Build**: Clean checkout → container build → install/lock → readiness → golden smokes → signed transcript

✅ **Container Setup**: Multi-stage Dockerfile with security best practices and health checks

✅ **Docker Compose**: Local development stack with health-gated service dependencies

✅ **Environment Pinning**: Dependency locking with hash recording and container digest tracking

✅ **Quality Gates**: Static analysis (mypy, ruff, bandit, semgrep) and license scanning capability

✅ **Boot Transcript**: Cryptographic signing of successful hermetic boot verification

## 🔧 Configuration

### Environment Variables
- `PACKREPO_ENV`: Environment mode (development/testing/production)
- `DEBUG`: Enable debug logging
- `PYTHONPATH`: Python module path
- `DATASET_REPOS`: Test repository list
- `CONTAINER_TAG`: Docker container tag

### Docker Compose Profiles
- `dev`: Development services
- `ci`: CI/testing services  
- `static`: Static analysis only
- `security`: Security scanning only
- `integration`: Integration testing services
- `prod`: Production simulation

## 📚 References

- [TODO.md](TODO.md) - Original requirements and specifications
- [PACKREPO_README.md](PACKREPO_README.md) - PackRepo functionality documentation
- [Dockerfile](infra/Dockerfile) - Container build specifications
- [Docker Compose](infra/compose.yaml) - Service orchestration
- [Pack Schema](spec/index.schema.json) - Pack format validation schema

## 🆘 Troubleshooting

### Common Issues

**Container Build Failures**:
- Check Docker daemon status
- Verify base image availability
- Check disk space and memory

**Hermetic Boot Failures**:
- Review boot transcript logs in `artifacts/`
- Check dependency lock integrity
- Verify Git repository state

**Test Failures**:
- Check artifacts directory for detailed reports
- Verify all required tools are installed
- Review container health status

### Debug Commands
```bash
# Check container status
make status

# View development logs
make dev-logs

# Clean and rebuild
make clean-all
make build

# Run individual components
docker-compose -f infra/compose.yaml up static-analysis
./scripts/pack_verify.py --help
```

This infrastructure establishes the foundation for sophisticated, reproducible PackRepo development with comprehensive quality assurance and hermetic build verification.