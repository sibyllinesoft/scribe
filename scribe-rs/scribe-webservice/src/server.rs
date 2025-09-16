//! Web server implementation for Scribe web service

use crate::{handlers, AppState, WebServiceConfig, WebServiceError, Result};
use axum::{
    routing::{get, post},
    Router,
};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::RwLock;
use tower_http::{
    cors::CorsLayer,
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::{info, warn};

/// Main web service struct
pub struct WebService {
    config: WebServiceConfig,
    app_state: AppState,
}

impl WebService {
    /// Create a new web service with the given configuration
    pub fn new(config: WebServiceConfig) -> Result<Self> {
        // Validate repository path
        if !config.repo_path.exists() {
            return Err(WebServiceError::RepositoryNotFound {
                path: config.repo_path.clone(),
            });
        }

        let app_state = AppState {
            config: config.clone(),
            bundle_state: Arc::new(RwLock::new(Default::default())),
            last_ping: Arc::new(tokio::sync::RwLock::new(tokio::time::Instant::now())),
            shutdown_sender: Arc::new(tokio::sync::RwLock::new(None)),
        };

        Ok(Self { config, app_state })
    }

    /// Start the web service and optionally open browser
    pub async fn start(self) -> Result<()> {
        let addr = SocketAddr::from(([127, 0, 0, 1], self.config.port));
        let auto_open_browser = self.config.auto_open_browser;
        let auto_shutdown = self.config.auto_shutdown;
        let shutdown_timeout = self.config.auto_shutdown_timeout;
        
        info!("🚀 Starting Scribe web service on http://{}", addr);
        info!("📁 Repository: {}", self.config.repo_path.display());
        info!("🎯 Token budget: {}", self.config.token_budget);
        
        if auto_shutdown {
            info!("⏰ Auto-shutdown enabled: {}s timeout", shutdown_timeout);
        }

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        
        // Store shutdown sender in app state
        {
            let mut sender_lock = self.app_state.shutdown_sender.write().await;
            *sender_lock = Some(shutdown_tx);
        }

        // Clone necessary values before moving self
        let last_ping = self.app_state.last_ping.clone();

        let app = self.create_router();

        // Open browser if requested  
        if auto_open_browser {
            let url = format!("http://{}/editor", addr); // Go directly to editor
            info!("🌐 Opening browser to {}", url);
            
            if let Err(e) = open::that(&url) {
                warn!("Failed to open browser: {}. Please navigate to {} manually", e, url);
            }
        }

        // Start auto-shutdown monitoring if enabled
        if auto_shutdown {
            let timeout_duration = tokio::time::Duration::from_secs(shutdown_timeout);
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    
                    let last_ping_time = *last_ping.read().await;
                    let elapsed = last_ping_time.elapsed();
                    
                    if elapsed > timeout_duration {
                        info!("⏰ Auto-shutdown triggered after {}s of inactivity", elapsed.as_secs());
                        std::process::exit(0);
                    }
                }
            });
        }

        // Start the server
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        info!("✅ Web service ready at http://{}", addr);
        
        // Run server until shutdown signal
        tokio::select! {
            result = axum::serve(listener, app) => {
                result.map_err(|e| WebServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            }
            _ = shutdown_rx => {
                info!("🛑 Shutdown signal received, stopping server");
            }
        }

        Ok(())
    }

    /// Create the Axum router with all routes
    pub fn create_router(self) -> Router {
        // Static file serving for web assets
        let static_service = ServeDir::new(self.get_static_dir());

        Router::new()
            // API routes
            .route("/api/status", get(handlers::status))
            .route("/api/ping", post(handlers::ping))
            .route("/api/shutdown", post(handlers::shutdown))
            .route("/api/scan", post(handlers::scan_repository))
            .route("/api/files", get(handlers::list_files))
            .route("/api/files/toggle", post(handlers::toggle_file))
            .route("/api/files/toggle-directory", post(handlers::toggle_directory))
            .route("/api/bundle/generate", post(handlers::generate_bundle))
            .route("/api/bundle/save", post(handlers::save_bundle))
            .route("/api/bundle/export", post(handlers::export_bundle))
            .route("/api/config", get(handlers::get_config))
            .route("/api/config", post(handlers::update_config))
            
            // Main web interface
            .route("/", get(handlers::index))
            .route("/editor", get(handlers::bundle_editor))
            
            // Static files (CSS, JS, images)
            .nest_service("/static", static_service)
            
            // Add middleware
            .layer(CorsLayer::permissive())
            .layer(TraceLayer::new_for_http())
            
            // Share state across all handlers
            .with_state(self.app_state)
    }

    /// Get the directory containing static web assets
    fn get_static_dir(&self) -> PathBuf {
        // In development, look for static files relative to cargo project
        let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
        let static_dir = PathBuf::from(cargo_manifest_dir).join("static");
        
        if static_dir.exists() {
            static_dir
        } else {
            // Fallback to creating a temporary directory with embedded assets
            self.create_embedded_static_dir()
        }
    }

    /// Create static directory with embedded web assets
    fn create_embedded_static_dir(&self) -> PathBuf {
        // For now, return a basic directory
        // TODO: Embed static assets using include_str! or similar
        std::env::temp_dir().join("scribe-webservice-static")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use axum_test::TestServer;
    use std::time::Duration;

    #[tokio::test]
    async fn test_webservice_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 0, // Use random port for testing
            ..Default::default()
        };

        let service = WebService::new(config);
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_repo_path() {
        let config = WebServiceConfig {
            repo_path: PathBuf::from("/nonexistent/path"),
            ..Default::default()
        };

        let service = WebService::new(config);
        assert!(service.is_err());
        
        if let Err(WebServiceError::RepositoryNotFound { path }) = service {
            assert_eq!(path, PathBuf::from("/nonexistent/path"));
        } else {
            panic!("Expected RepositoryNotFound error");
        }
    }

    #[tokio::test]
    async fn test_webservice_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test valid configuration
        let valid_config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 8080,
            host: "127.0.0.1".to_string(),
            token_budget: 50000,
            auto_open_browser: false,
            max_file_size: 1024 * 1024,
            auto_exclude_tests: true,
        };
        
        let service = WebService::new(valid_config);
        assert!(service.is_ok());
        
        let service = service.unwrap();
        assert_eq!(service.config.port, 8080);
        assert_eq!(service.config.host, "127.0.0.1");
        assert_eq!(service.config.token_budget, 50000);
    }

    #[tokio::test]
    async fn test_create_router() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 0,
            auto_open_browser: false,
            ..Default::default()
        };

        let service = WebService::new(config).unwrap();
        let router = service.create_router();
        
        // Test that the router was created (it's hard to test routes directly without starting the server)
        // But we can at least verify it doesn't panic
        let _router = router;
    }

    #[tokio::test]
    async fn test_get_static_dir() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 0,
            auto_open_browser: false,
            ..Default::default()
        };

        let service = WebService::new(config).unwrap();
        let static_dir = service.get_static_dir();
        
        // Should return some path
        assert!(static_dir.is_absolute());
    }

    #[tokio::test]
    async fn test_create_embedded_static_dir() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 0,
            auto_open_browser: false,
            ..Default::default()
        };

        let service = WebService::new(config).unwrap();
        let embedded_dir = service.create_embedded_static_dir();
        
        // Should return a path in temp directory
        assert!(embedded_dir.to_string_lossy().contains("scribe-webservice-static"));
    }

    #[tokio::test]
    async fn test_app_state_structure() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 8080,
            host: "0.0.0.0".to_string(),
            token_budget: 25000,
            auto_open_browser: true,
            max_file_size: 512 * 1024,
            auto_exclude_tests: false,
        };

        let service = WebService::new(config.clone()).unwrap();
        
        // Test that app_state has the correct config
        assert_eq!(service.app_state.config.port, config.port);
        assert_eq!(service.app_state.config.host, config.host);
        assert_eq!(service.app_state.config.token_budget, config.token_budget);
        assert_eq!(service.app_state.config.auto_open_browser, config.auto_open_browser);
        assert_eq!(service.app_state.config.max_file_size, config.max_file_size);
        assert_eq!(service.app_state.config.auto_exclude_tests, config.auto_exclude_tests);
        
        // Test that bundle_state is initialized
        let bundle_state = service.app_state.bundle_state.try_read().unwrap();
        assert_eq!(bundle_state.included_files.len(), 0);
        assert_eq!(bundle_state.excluded_files.len(), 0);
        assert_eq!(bundle_state.token_estimate, 0);
        assert_eq!(bundle_state.total_size, 0);
    }

    #[tokio::test]
    async fn test_webservice_with_different_hosts() {
        let temp_dir = TempDir::new().unwrap();
        
        let hosts = vec!["127.0.0.1", "localhost", "0.0.0.0"];
        
        for host in hosts {
            let config = WebServiceConfig {
                repo_path: temp_dir.path().to_path_buf(),
                port: 0,
                host: host.to_string(),
                auto_open_browser: false,
                ..Default::default()
            };
            
            let service = WebService::new(config);
            assert!(service.is_ok(), "Failed to create service with host: {}", host);
            
            let service = service.unwrap();
            assert_eq!(service.config.host, host);
        }
    }

    #[tokio::test]
    async fn test_webservice_port_configuration() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test various port configurations
        let ports = vec![0, 8080, 8081, 3000, 9000];
        
        for port in ports {
            let config = WebServiceConfig {
                repo_path: temp_dir.path().to_path_buf(),
                port,
                auto_open_browser: false,
                ..Default::default()
            };
            
            let service = WebService::new(config);
            assert!(service.is_ok(), "Failed to create service with port: {}", port);
            
            let service = service.unwrap();
            assert_eq!(service.config.port, port);
        }
    }

    #[tokio::test]
    async fn test_webservice_token_budget_configuration() {
        let temp_dir = TempDir::new().unwrap();
        
        let budgets = vec![1000, 10000, 50000, 100000, 500000];
        
        for budget in budgets {
            let config = WebServiceConfig {
                repo_path: temp_dir.path().to_path_buf(),
                port: 0,
                token_budget: budget,
                auto_open_browser: false,
                ..Default::default()
            };
            
            let service = WebService::new(config);
            assert!(service.is_ok(), "Failed to create service with budget: {}", budget);
            
            let service = service.unwrap();
            assert_eq!(service.config.token_budget, budget);
        }
    }

    #[tokio::test]
    async fn test_webservice_file_size_limits() {
        let temp_dir = TempDir::new().unwrap();
        
        let file_sizes = vec![1024, 1024 * 1024, 5 * 1024 * 1024, 10 * 1024 * 1024];
        
        for max_file_size in file_sizes {
            let config = WebServiceConfig {
                repo_path: temp_dir.path().to_path_buf(),
                port: 0,
                max_file_size,
                auto_open_browser: false,
                ..Default::default()
            };
            
            let service = WebService::new(config);
            assert!(service.is_ok(), "Failed to create service with max_file_size: {}", max_file_size);
            
            let service = service.unwrap();
            assert_eq!(service.config.max_file_size, max_file_size);
        }
    }

    #[tokio::test]
    async fn test_bundle_state_default() {
        let bundle_state = crate::BundleState::default();
        
        assert_eq!(bundle_state.included_files.len(), 0);
        assert_eq!(bundle_state.excluded_files.len(), 0);
        assert_eq!(bundle_state.token_estimate, 0);
        assert_eq!(bundle_state.total_size, 0);
        // last_updated should be recent
        let now = chrono::Utc::now();
        assert!(now.signed_duration_since(bundle_state.last_updated).num_seconds() < 5);
    }
}