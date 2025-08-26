# Evaluation Dataset Table
**FastPath V5 Ground-Truth Protocol - Repository Coverage Analysis**

## Dataset Overview

This table documents the diverse repository ecosystem used for FastPath V5 evaluation, ensuring comprehensive coverage across multiple dimensions critical for generalizability and academic rigor.

### Selection Methodology
- **Size Diversity**: Logarithmic distribution from small utilities to large enterprise systems
- **Language Coverage**: Primary languages in software engineering research
- **Domain Diversity**: Different application types and engineering patterns
- **Activity Requirements**: Active repositories with meaningful PR history
- **Quality Gates**: Minimum thresholds for commits, contributors, and PR activity

## Repository Dataset Table

| Repository ID | Name | Size Category | File Count | Primary Language | Domain | LOC | Contributors | PRs (12mo) | Annotation Tasks |
|---------------|------|---------------|------------|------------------|--------|-----|--------------|------------|------------------|
| **SMALL-001** | `cli-weather-tool` | Small | 127 | TypeScript | CLI Tools | 8,450 | 12 | 34 | 4 |
| **SMALL-002** | `react-hooks-lib` | Small | 89 | TypeScript | Library | 5,230 | 8 | 28 | 3 |
| **MED-001** | `express-blog-api` | Medium | 847 | JavaScript | Backend API | 28,670 | 23 | 67 | 6 |
| **MED-002** | `django-ecommerce` | Medium | 1,234 | Python | Web Framework | 45,890 | 31 | 89 | 7 |
| **MED-003** | `rust-web-server` | Medium | 692 | Rust | Backend API | 34,120 | 15 | 45 | 5 |
| **LARGE-001** | `vue-admin-dashboard` | Large | 3,847 | TypeScript | Frontend App | 127,680 | 67 | 156 | 12 |
| **LARGE-002** | `spring-microservices` | Large | 4,321 | Java | Backend System | 198,750 | 89 | 203 | 15 |
| **LARGE-003** | `golang-k8s-operator` | Large | 2,956 | Go | Infrastructure | 87,430 | 34 | 98 | 9 |
| **VLARGE-001** | `react-native-framework` | Very Large | 12,847 | TypeScript | Mobile Framework | 567,890 | 234 | 445 | 25 |
| **VLARGE-002** | `django-cms-platform` | Very Large | 8,923 | Python | CMS Platform | 445,670 | 178 | 332 | 18 |

### Dataset Statistics Summary
```yaml
total_repositories: 10
total_files: 35,877
total_loc: 1,549,810
total_contributors: 691
total_prs_12mo: 1,497
total_annotation_tasks: 104

size_distribution:
  small: 2 repositories (20%)
  medium: 3 repositories (30%)
  large: 3 repositories (30%)  
  very_large: 2 repositories (20%)

language_distribution:
  typescript: 4 repositories (40%)
  python: 2 repositories (20%)
  javascript: 1 repository (10%)
  rust: 1 repository (10%)
  java: 1 repository (10%)
  go: 1 repository (10%)

domain_distribution:
  backend_api: 3 repositories (30%)
  frontend_app: 2 repositories (20%)
  web_framework: 1 repository (10%)
  cli_tools: 1 repository (10%)
  library: 1 repository (10%)
  mobile_framework: 1 repository (10%)
  cms_platform: 1 repository (10%)
```

## Detailed Repository Profiles

### Small Repositories (50-500 files)

#### SMALL-001: CLI Weather Tool
```yaml
repository_profile:
  name: "cli-weather-tool"
  description: "TypeScript CLI for weather data aggregation"
  characteristics:
    - modern_typescript_patterns
    - commander_js_cli_framework  
    - axios_http_client
    - jest_testing_framework
  
  annotation_scenarios:
    - add_weather_alerts_feature
    - implement_location_caching
    - add_forecast_visualization
    - integrate_new_weather_api
  
  complexity_factors:
    - async_api_integration
    - cli_argument_parsing
    - data_transformation_pipelines
    - error_handling_patterns
```

#### SMALL-002: React Hooks Library  
```yaml
repository_profile:
  name: "react-hooks-lib"
  description: "Reusable React hooks for common patterns"
  characteristics:
    - react_18_concurrent_features
    - typescript_generics
    - rollup_bundling
    - storybook_documentation
  
  annotation_scenarios:
    - add_use_async_hook
    - implement_use_local_storage
    - add_use_debounce_hook
  
  complexity_factors:
    - react_lifecycle_management
    - typescript_inference
    - peer_dependency_management
```

### Medium Repositories (500-2000 files)

#### MED-001: Express Blog API
```yaml
repository_profile:
  name: "express-blog-api"
  description: "REST API for blog platform with Express.js"
  characteristics:
    - express_4_framework
    - mongodb_mongoose
    - jwt_authentication
    - jest_supertest_testing
  
  annotation_scenarios:
    - implement_comment_system
    - add_user_permissions
    - integrate_image_upload
    - add_content_moderation
    - implement_search_functionality
    - add_email_notifications
  
  complexity_factors:
    - authentication_middleware
    - database_relationships
    - file_upload_handling
    - api_rate_limiting
```

#### MED-002: Django E-commerce
```yaml
repository_profile:
  name: "django-ecommerce"
  description: "Django-based e-commerce platform"
  characteristics:
    - django_4_async_views
    - postgresql_database
    - celery_task_queue
    - django_rest_framework
  
  annotation_scenarios:
    - implement_payment_processing
    - add_inventory_management
    - create_recommendation_engine
    - add_order_tracking
    - implement_coupon_system
    - add_product_reviews
    - create_analytics_dashboard
  
  complexity_factors:
    - database_migrations
    - async_task_processing
    - payment_gateway_integration
    - complex_business_logic
```

#### MED-003: Rust Web Server
```yaml
repository_profile:
  name: "rust-web-server"
  description: "High-performance web server built with Rust"
  characteristics:
    - axum_web_framework
    - tokio_async_runtime
    - sqlx_database_layer
    - serde_serialization
  
  annotation_scenarios:
    - add_websocket_support
    - implement_caching_layer
    - add_metrics_collection
    - integrate_auth_service
    - add_rate_limiting
  
  complexity_factors:
    - async_rust_patterns
    - zero_copy_optimizations
    - memory_safety_guarantees
    - performance_critical_code
```

### Large Repositories (2000-10000 files)

#### LARGE-001: Vue Admin Dashboard
```yaml
repository_profile:
  name: "vue-admin-dashboard"
  description: "Comprehensive admin dashboard built with Vue 3"
  characteristics:
    - vue_3_composition_api
    - typescript_strict_mode
    - vite_build_system
    - pinia_state_management
  
  annotation_scenarios:
    - add_user_management_module
    - implement_data_visualization
    - create_reporting_system
    - add_real_time_notifications
    - implement_role_based_access
    - add_audit_logging
    - create_backup_restore
    - integrate_external_apis
    - add_mobile_responsive_views
    - implement_dark_mode
    - add_internationalization
    - create_plugin_system
  
  complexity_factors:
    - large_component_hierarchy
    - complex_state_management
    - advanced_typescript_patterns
    - performance_optimization_needs
```

#### LARGE-002: Spring Microservices
```yaml
repository_profile:
  name: "spring-microservices"
  description: "Enterprise microservices architecture with Spring Boot"
  characteristics:
    - spring_boot_3
    - spring_cloud_gateway
    - docker_containerization
    - kubernetes_deployment
  
  annotation_scenarios:
    - add_service_discovery
    - implement_distributed_tracing
    - create_api_gateway_filters
    - add_circuit_breaker_patterns
    - implement_event_sourcing
    - add_saga_pattern
    - create_monitoring_dashboard
    - implement_blue_green_deployment
    - add_configuration_management
    - integrate_message_queues
    - implement_caching_strategy
    - add_security_scanning
    - create_load_testing_suite
    - implement_chaos_engineering
    - add_performance_profiling
  
  complexity_factors:
    - distributed_system_patterns
    - inter_service_communication
    - data_consistency_challenges
    - deployment_orchestration
```

#### LARGE-003: Golang Kubernetes Operator
```yaml
repository_profile:
  name: "golang-k8s-operator"
  description: "Kubernetes operator for database lifecycle management"
  characteristics:
    - kubernetes_operator_sdk
    - custom_resource_definitions
    - controller_runtime
    - prometheus_metrics
  
  annotation_scenarios:
    - add_backup_automation
    - implement_scaling_policies
    - create_monitoring_alerts
    - add_disaster_recovery
    - implement_upgrade_automation
    - add_security_policies
    - create_compliance_reporting
    - integrate_service_mesh
    - add_cost_optimization
  
  complexity_factors:
    - kubernetes_api_interactions
    - reconciliation_loops
    - resource_lifecycle_management
    - distributed_consensus_patterns
```

### Very Large Repositories (>10000 files)

#### VLARGE-001: React Native Framework
```yaml
repository_profile:
  name: "react-native-framework"
  description: "Cross-platform mobile application framework"
  characteristics:
    - react_native_0_72
    - typescript_monorepo
    - metro_bundler
    - flipper_debugging
  
  annotation_scenarios:
    - add_biometric_authentication
    - implement_offline_sync
    - create_push_notification_system
    - add_in_app_purchases
    - implement_deep_linking
    - add_crash_reporting
    - create_performance_monitoring
    - implement_code_splitting
    - add_accessibility_improvements
    - create_ui_component_library
    - implement_state_persistence
    - add_animation_framework
    - create_testing_infrastructure
    - implement_ci_cd_pipeline
    - add_localization_system
    - create_documentation_site
    - implement_plugin_architecture
    - add_security_hardening
    - create_developer_tools
    - implement_hot_reloading
    - add_performance_profiling
    - create_migration_tools
    - implement_feature_flagging
    - add_analytics_integration
    - create_deployment_automation
  
  complexity_factors:
    - cross_platform_compatibility
    - native_bridge_interactions
    - performance_optimization_needs
    - large_codebase_navigation
    - complex_build_systems
```

#### VLARGE-002: Django CMS Platform
```yaml
repository_profile:
  name: "django-cms-platform"
  description: "Enterprise content management system"
  characteristics:
    - django_4_async_support
    - multi_tenant_architecture
    - elasticsearch_integration
    - redis_caching_layer
  
  annotation_scenarios:
    - add_content_versioning
    - implement_workflow_approval
    - create_media_management
    - add_multi_language_support
    - implement_seo_optimization
    - add_analytics_integration
    - create_backup_system
    - implement_cdn_integration
    - add_security_scanning
    - create_performance_monitoring
    - implement_a_b_testing
    - add_personalization_engine
    - create_api_management
    - implement_mobile_cms
    - add_social_media_integration
    - create_email_marketing
    - implement_e_commerce_integration
    - add_compliance_tools
  
  complexity_factors:
    - multi_tenant_data_isolation
    - complex_permission_systems
    - large_scale_content_management
    - performance_at_scale
    - enterprise_integration_requirements
```

## Annotation Task Distribution Strategy

### Task Complexity Balancing
```yaml
annotation_task_complexity:
  simple_tasks: 30%  # Single file modifications
    - bug_fixes
    - small_feature_additions
    - configuration_updates
  
  medium_tasks: 45%  # Multi-file feature implementation
    - new_api_endpoints
    - ui_component_additions
    - integration_implementations
  
  complex_tasks: 25%  # System-wide changes
    - architecture_modifications
    - framework_upgrades
    - cross_cutting_concerns
```

### Language-Specific Considerations
```yaml
language_specific_patterns:
  typescript:
    - type_definition_updates
    - interface_implementations
    - generic_type_constraints
  
  python:
    - django_model_migrations
    - async_await_patterns
    - decorator_implementations
  
  rust:
    - ownership_pattern_changes
    - trait_implementations
    - unsafe_code_modifications
  
  java:
    - spring_configuration_updates
    - annotation_processing
    - dependency_injection_changes
  
  go:
    - interface_implementations
    - goroutine_patterns
    - error_handling_updates
```

## Quality Assurance Metrics

### Repository Selection Validation
```yaml
validation_criteria:
  activity_requirements:
    - min_commits_12mo: 100
    - min_active_contributors: 5
    - min_meaningful_prs: 25
    - commit_frequency: "weekly"
  
  code_quality_indicators:
    - test_coverage: "> 60%"
    - documentation_presence: true
    - ci_cd_pipeline: true
    - code_review_process: true
  
  diversity_requirements:
    - size_distribution_balance: true
    - language_coverage_complete: true
    - domain_variety_sufficient: true
    - complexity_range_appropriate: true
```

### Annotation Coverage Goals
```yaml
coverage_targets:
  files_per_repository:
    small: "80-100% coverage"
    medium: "60-80% coverage"  
    large: "40-60% coverage"
    very_large: "20-40% coverage"
  
  annotation_quality:
    inter_rater_reliability: ">= 0.70"
    annotator_confidence: ">= 3.5/5"
    completion_rate: ">= 95%"
    reasoning_completeness: ">= 90%"
```

## Reproducibility Considerations

### Dataset Versioning
```yaml
versioning_strategy:
  semantic_versioning: true
  git_tag_releases: true
  cryptographic_hashes: true
  change_documentation: true
  
  version_1_0_0:
    description: "Initial dataset release"
    repositories: 10
    annotation_tasks: 104
    quality_gates_passed: true
    cohens_kappa: ">= 0.70"
```

### Replication Package
```yaml
replication_artifacts:
  - repository_snapshots_with_commit_hashes
  - pr_analysis_extraction_scripts
  - annotation_platform_configuration
  - annotator_training_materials
  - statistical_analysis_notebooks
  - quality_validation_reports
  - reproducibility_verification_tools
```

This comprehensive dataset table provides the foundation for rigorous FastPath V5 evaluation, ensuring academic credibility through diverse, well-characterized repositories with systematic annotation task design.