//! Tests for the CLI binary functionality
//! 
//! These tests verify the command-line interface and argument parsing

use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Test that the binary can be invoked with --help
#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "scribe-web", "--", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    assert!(stdout.contains("Scribe Web Service"));
    assert!(stdout.contains("repository analysis"));
    assert!(stdout.contains("--port"));
    assert!(stdout.contains("--host"));
    assert!(stdout.contains("--token-budget"));
    assert!(stdout.contains("--no-browser"));
    assert!(stdout.contains("--max-file-size"));
    assert!(stdout.contains("--no-exclude-tests"));
}

/// Test that the binary can be invoked with --version
#[test]
fn test_cli_version() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "scribe-web", "--", "--version"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Should contain version information
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")));
}

/// Test that the binary fails with a non-existent repository path
#[test]
fn test_cli_invalid_repo_path() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "scribe-web", "--", "/nonexistent/path"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Debug output to see what we actually get
    if !stderr.contains("Repository path does not exist") && !stdout.contains("Repository path does not exist") {
        println!("STDERR: {}", stderr);
        println!("STDOUT: {}", stdout);
    }
    
    // The error might be in stdout or stderr depending on log configuration
    assert!(stderr.contains("Repository path does not exist") || stdout.contains("Repository path does not exist"));
}

/// Test that the binary accepts valid arguments
#[test]
fn test_cli_valid_arguments() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test with various valid arguments - we'll kill it quickly since we just want to test parsing
    let mut child = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--port", "8081",
            "--host", "0.0.0.0",
            "--token-budget", "25000",
            "--no-browser",
            "--max-file-size", "2048000",
            "--no-exclude-tests"
        ])
        .spawn()
        .expect("Failed to start process");

    // Give it a moment to start and parse arguments
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    // Kill the process - we just wanted to test argument parsing
    let _ = child.kill();
    let _ = child.wait();
}

/// Test CLI argument parsing edge cases
#[test]
fn test_cli_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test minimum port
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--port", "0"  // Should be valid (OS will assign random port)
        ])
        .env("RUST_LOG", "off")  // Reduce log output
        .spawn();
    
    if let Ok(mut child) = output {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Test invalid port number
#[test]
fn test_cli_invalid_port() {
    let temp_dir = TempDir::new().unwrap();
    
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--port", "999999"  // Invalid port number
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    
    assert!(stderr.contains("Invalid port number"));
}

/// Test invalid token budget
#[test]
fn test_cli_invalid_token_budget() {
    let temp_dir = TempDir::new().unwrap();
    
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--token-budget", "not_a_number"
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    
    assert!(stderr.contains("Invalid token budget"));
}

/// Test invalid max file size
#[test]
fn test_cli_invalid_max_file_size() {
    let temp_dir = TempDir::new().unwrap();
    
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--max-file-size", "invalid"
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    
    assert!(stderr.contains("Invalid max file size"));
}

/// Test directory vs file validation
#[test]
fn test_cli_file_instead_of_directory() {
    let temp_dir = TempDir::new().unwrap();
    let temp_file = temp_dir.path().join("test_file.txt");
    std::fs::write(&temp_file, "test content").unwrap();
    
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Debug output to see what we actually get
    if !stderr.contains("Repository path is not a directory") && !stdout.contains("Repository path is not a directory") {
        println!("STDERR: {}", stderr);
        println!("STDOUT: {}", stdout);
    }
    
    // The error might be in stdout or stderr depending on log configuration
    assert!(stderr.contains("Repository path is not a directory") || stdout.contains("Repository path is not a directory"));
}

/// Test all boolean flags
#[test]
fn test_cli_boolean_flags() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test --no-browser flag
    let mut child = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--no-browser"
        ])
        .spawn();
    
    if let Ok(mut proc) = child {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = proc.kill();
        let _ = proc.wait();
    }
    
    // Test --no-exclude-tests flag
    let mut child = Command::new("cargo")
        .args(&[
            "run", "--bin", "scribe-web", "--",
            temp_dir.path().to_str().unwrap(),
            "--no-exclude-tests"
        ])
        .spawn();
    
    if let Ok(mut proc) = child {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = proc.kill();
        let _ = proc.wait();
    }
}