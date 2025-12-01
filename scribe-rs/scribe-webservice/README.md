# scribe-webservice

Web service interface for Scribe repository analysis and interactive bundle editing.

## Overview

`scribe-webservice` provides a RESTful API and interactive web UI for Scribe's repository analysis capabilities. It enables browser-based repository exploration, real-time bundle customization, and integration with web-based development tools. The service is built on Axum, a high-performance async web framework.

## Key Features

### REST API
- **Repository analysis**: Analyze repositories via HTTP POST
- **Covering set queries**: Surgical file selection for specific entities
- **Bundle export**: Generate bundles in multiple formats (Markdown, JSON, HTML, XML)
- **Configuration endpoints**: Customize analysis parameters via API
- **Health checks**: Service status and metrics endpoints

### Interactive Web UI
- **File tree visualization**: Hierarchical view of repository structure
- **Real-time filtering**: Search and filter files by name, language, or score
- **Selection customization**: Toggle file inclusion interactively
- **Metadata display**: View file scores, centrality, and inclusion reasons
- **Export options**: Download bundles in preferred format
- **Responsive design**: Mobile-friendly interface

### Bundle Editor
- **Self-contained HTML**: Single-file bundle with embedded editor
- **Syntax highlighting**: Language-aware code display
- **Navigation**: Jump to files, search content
- **Statistics**: Token counts, file counts, selection summaries
- **No backend required**: Fully client-side after generation

### Performance
- **Async processing**: Non-blocking analysis using tokio
- **Streaming responses**: Progressive results for large repos
- **Resource limits**: Configurable timeouts and memory bounds
- **Caching**: HTTP caching headers for static assets
- **CORS support**: Cross-origin requests for integration

## Architecture

```
HTTP Request → Axum Router → Service Layer → Scribe Core → Response
      ↓             ↓              ↓              ↓            ↓
  JSON Body    Path Matching   Analysis      Scanner      JSON/HTML
   + Query     + Middleware    Orchestration  + Graph     + Headers
    Params                                     + Selection
      ↓                                            ↓
  CORS                                       Async Tokio
  Logging                                    Processing
```

### Core Components

#### `ScribeService`
Main service orchestrator:
- Manages repository analysis lifecycle
- Coordinates scanner, graph, and selection components
- Handles configuration and state
- Provides progress reporting

#### `ApiHandlers`
HTTP request handlers:
- `POST /api/analyze`: Full repository analysis
- `POST /api/covering-set`: Surgical entity selection
- `GET /api/health`: Service health and metrics
- `GET /api/config`: Current configuration
- `POST /api/config`: Update configuration

#### `BundleRenderer`
Generates output in multiple formats:
- **Markdown**: Human-readable text format
- **JSON**: Structured data with full metadata
- **HTML**: Interactive self-contained editor
- **XML**: For tool integration
- Handlebars templates for HTML rendering

#### `StaticAssetServer`
Serves web UI resources:
- JavaScript bundles (tree visualization, syntax highlighting)
- CSS stylesheets
- Fonts and icons
- Embedded in binary or served from disk

## Usage

### Running the Service

```rust
use scribe_webservice::{ScribeService, ServiceConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = ServiceConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        token_budget: 100_000,
        max_file_size: 1_000_000,
        ..Default::default()
    };

    let service = ScribeService::new(config);
    service.start().await?;

    println!("Scribe service running on http://127.0.0.1:8080");
    Ok(())
}
```

### CLI Integration

The service is exposed via the Scribe CLI:

```bash
# Start service and open browser
scribe --port 8080 --open-browser

# Specify repository
scribe /path/to/repo --port 9000

# Custom configuration
scribe --port 8080 --token-budget 150000 --max-file-size 2000000

# Generate HTML bundle with editor
scribe --style html --editor --output bundle.html
```

### API Examples

#### Analyze Repository

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/repo",
    "token_budget": 100000,
    "exclude_tests": true,
    "algorithm": "simple_router"
  }'
```

Response:
```json
{
  "status": "success",
  "files_selected": 156,
  "total_tokens": 98534,
  "analysis_time_ms": 1234,
  "bundle_url": "/api/bundle/abc123"
}
```

#### Covering Set Query

```bash
curl -X POST http://localhost:8080/api/covering-set \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/repo",
    "entity_name": "authenticate_user",
    "entity_type": "function",
    "max_files": 20,
    "max_depth": 3
  }'
```

Response:
```json
{
  "status": "success",
  "files": [
    {
      "path": "src/auth.rs",
      "reason": "Target",
      "depth": 0
    },
    {
      "path": "src/db.rs",
      "reason": "DirectDependency",
      "depth": 1
    }
  ]
}
```

#### Health Check

```bash
curl http://localhost:8080/api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.5.1",
  "uptime_seconds": 12345,
  "requests_processed": 42,
  "cache_hit_rate": 0.75
}
```

### Programmatic Usage

```rust
use scribe_webservice::{ApiClient, AnalyzeRequest};

let client = ApiClient::new("http://localhost:8080")?;

let request = AnalyzeRequest {
    repo_path: PathBuf::from("."),
    token_budget: 100_000,
    exclude_tests: true,
    algorithm: Algorithm::SimpleRouter,
};

let response = client.analyze(request).await?;

println!("Selected {} files", response.files_selected);
println!("Bundle available at: {}", response.bundle_url);
```

## Interactive Bundle Editor

### Features

The generated HTML bundle includes a full-featured editor:

- **File tree**: Collapsible directory structure
- **Search**: Find files and content
- **Filtering**: By language, size, score
- **Metadata panel**: File details and metrics
- **Export**: Download as Markdown or JSON
- **Theme toggle**: Light/dark mode
- **Responsive**: Works on mobile devices

### Generation

```bash
# Generate interactive HTML bundle
scribe --style html --editor --output bundle.html

# Opens in browser automatically
scribe --style html --editor --output bundle.html --open-browser
```

### Self-Contained Design

The HTML bundle is a single file containing:
- Full repository file tree
- All file contents with syntax highlighting
- JavaScript for interactivity (React tree component)
- CSS for styling
- No external dependencies (works offline)

**File size**: ~280KB smaller with CDN-linked assets (vs embedded).

### Customization

Modify templates in `templates/` directory:
- `bundle_editor.hbs`: Main HTML structure
- `file_tree.hbs`: Tree component template
- CSS in `assets/styles.css`
- JavaScript in `assets/scribe-tree-bundle.js`

Rebuild frontend:
```bash
cd scribe-rs/frontend
bun run build
cp dist/scribe-tree-bundle.js ../assets/
```

## Configuration

### `ServiceConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `String` | `"127.0.0.1"` | Bind address |
| `port` | `u16` | `8080` | HTTP port |
| `token_budget` | `usize` | `100_000` | Default token budget |
| `max_file_size` | `usize` | `1_000_000` | Max file size to process |
| `max_requests` | `usize` | `100` | Rate limit |
| `request_timeout_secs` | `u64` | `300` | Request timeout (5 min) |
| `enable_cors` | `bool` | `true` | CORS support |
| `log_level` | `String` | `"info"` | Logging verbosity |

### Environment Variables

Configure via environment:

```bash
SCRIBE_HOST=0.0.0.0
SCRIBE_PORT=9000
SCRIBE_TOKEN_BUDGET=150000
SCRIBE_LOG_LEVEL=debug
RUST_LOG=scribe_webservice=debug
```

## Endpoints

### Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Full repository analysis |
| `POST` | `/api/covering-set` | Entity-targeted selection |
| `GET` | `/api/bundle/:id` | Retrieve generated bundle |

### Configuration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/config` | Get current config |
| `POST` | `/api/config` | Update config |

### Health & Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/metrics` | Performance metrics |

### Static Assets

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/assets/*` | JavaScript, CSS, fonts |
| `GET` | `/` | Web UI homepage |

## Deployment

### Local Development

```bash
# Run with hot reload
cd scribe-rs
cargo watch -x 'run --bin scribe -- --port 8080'
```

### Docker

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p scribe-webservice

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/scribe /usr/local/bin/
EXPOSE 8080
CMD ["scribe", "--port", "8080", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t scribe-webservice .
docker run -p 8080:8080 -v $(pwd):/repo scribe-webservice scribe /repo
```

### Production

For production deployment:

1. **Reverse proxy**: Use nginx/Caddy for HTTPS termination
2. **Process manager**: systemd, supervisord, or Docker
3. **Resource limits**: Set memory/CPU limits
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Rate limiting**: Configure `max_requests` appropriately
6. **CORS**: Restrict allowed origins

Example nginx config:
```nginx
server {
    listen 80;
    server_name scribe.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Integration

`scribe-webservice` integrates with:

- **CLI**: Exposes web UI for local repositories
- **CI/CD**: Generate bundle reports in pipelines
- **IDEs**: Potential for editor plugins
- **Cloud platforms**: Deploy as microservice
- **Documentation sites**: Embed interactive bundles

## See Also

- `scribe-scaling`: Handles large repository processing
- `scribe-selection`: Implements selection algorithms exposed via API
- `templates/`: Handlebars templates for HTML rendering
- `frontend/`: React-based interactive UI components
- `../../BUNDLE_EDITOR.md`: Bundle editor documentation
