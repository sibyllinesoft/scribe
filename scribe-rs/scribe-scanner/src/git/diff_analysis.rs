//! Diff analysis methods for GitIntegrator

use scribe_core::{Result, ScribeError};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command as AsyncCommand;

use super::diff::{DiffAnalysisConfig, DiffAnalysisResult, DiffChangeType, DiffSource, GitDiffEntry};
use super::git_integration::GitIntegrator;

impl GitIntegrator {
    /// Perform comprehensive diff-based analysis
    pub async fn analyze_diffs(&self, config: &DiffAnalysisConfig) -> Result<DiffAnalysisResult> {
        if !self.is_git_available() {
            return Err(ScribeError::git("Git not available for diff analysis".to_string()));
        }

        let mut all_diffs = Vec::new();

        if config.include_staged {
            let staged_diffs = self.extract_staged_diffs(config).await?;
            all_diffs.extend(staged_diffs);
        }

        if config.include_unstaged {
            let unstaged_diffs = self.extract_unstaged_diffs(config).await?;
            all_diffs.extend(unstaged_diffs);
        }

        if let Some(ref commits) = config.include_commits {
            for commit_hash in commits {
                let commit_diffs = self.extract_commit_diffs(commit_hash, config).await?;
                all_diffs.extend(commit_diffs);
            }
        }

        if let Some(ref range) = config.commit_range {
            let range_diffs = self.extract_range_diffs(range, config).await?;
            all_diffs.extend(range_diffs);
        }

        if let Some(ref branch_comp) = config.branch_comparison {
            let branch_diffs = self.extract_branch_comparison_diffs(branch_comp, config).await?;
            all_diffs.extend(branch_diffs);
        }

        all_diffs = self.filter_diffs(all_diffs, config).await?;

        let total_files_changed = all_diffs.len();
        let total_additions = all_diffs.iter().map(|d| d.line_additions).sum();
        let total_deletions = all_diffs.iter().map(|d| d.line_deletions).sum();

        let analysis_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(DiffAnalysisResult {
            diffs: all_diffs,
            total_files_changed,
            total_additions,
            total_deletions,
            commit_range_analyzed: config.commit_range.clone(),
            analysis_timestamp,
        })
    }

    async fn extract_staged_diffs(&self, _config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        let output = AsyncCommand::new("git")
            .arg("diff")
            .arg("--cached")
            .arg("--numstat")
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get staged diffs: {}", e)))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_numstat_output(&stdout, DiffSource::Staged).await
    }

    async fn extract_unstaged_diffs(&self, _config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        let output = AsyncCommand::new("git")
            .arg("diff")
            .arg("--numstat")
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get unstaged diffs: {}", e)))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_numstat_output(&stdout, DiffSource::Unstaged).await
    }

    async fn extract_commit_diffs(&self, commit_hash: &str, _config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        let output = AsyncCommand::new("git")
            .arg("show")
            .arg("--numstat")
            .arg("--name-status")
            .arg("--pretty=format:%H|%an|%at|%s")
            .arg(commit_hash)
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get commit diffs: {}", e)))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_commit_diff_output(&stdout, commit_hash).await
    }

    async fn extract_range_diffs(&self, range: &str, config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        let output = AsyncCommand::new("git")
            .arg("log")
            .arg("--numstat")
            .arg("--pretty=format:%H|%an|%at|%s")
            .arg(format!("--max-count={}", config.max_commits))
            .arg(range)
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get range diffs: {}", e)))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_log_diff_output(&stdout).await
    }

    async fn extract_branch_comparison_diffs(&self, branch_comp: &str, _config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        let output = AsyncCommand::new("git")
            .arg("diff")
            .arg("--numstat")
            .arg(branch_comp)
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get branch comparison diffs: {}", e)))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_numstat_output(&stdout, DiffSource::BranchComparison).await
    }

    async fn parse_numstat_output(&self, output: &str, source: DiffSource) -> Result<Vec<GitDiffEntry>> {
        let mut diffs = Vec::new();

        for line in output.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let additions = if parts[0] == "-" { 0 } else { parts[0].parse::<usize>().unwrap_or(0) };
                let deletions = if parts[1] == "-" { 0 } else { parts[1].parse::<usize>().unwrap_or(0) };
                let file_path = PathBuf::from(parts[2]);

                let diff_content = self.get_file_diff_content(&file_path, &source).await?;
                let change_type = self.determine_change_type(&file_path, &source).await?;

                diffs.push(GitDiffEntry {
                    file_path,
                    change_type,
                    diff_content,
                    line_additions: additions,
                    line_deletions: deletions,
                    commit_hash: None,
                    commit_message: None,
                    author: None,
                    timestamp: None,
                    old_file_path: None,
                });
            }
        }

        Ok(diffs)
    }

    async fn parse_commit_diff_output(&self, output: &str, commit_hash: &str) -> Result<Vec<GitDiffEntry>> {
        let lines: Vec<&str> = output.lines().collect();
        let mut diffs = Vec::new();

        if lines.is_empty() {
            return Ok(diffs);
        }

        let (commit_info, author, timestamp, message) = if let Some(first_line) = lines.first() {
            if first_line.contains('|') && first_line.split('|').count() >= 4 {
                let parts: Vec<&str> = first_line.split('|').collect();
                (Some(parts[0].to_string()), Some(parts[1].to_string()), parts[2].parse::<u64>().ok(), Some(parts[3].to_string()))
            } else {
                (Some(commit_hash.to_string()), None, None, None)
            }
        } else {
            (Some(commit_hash.to_string()), None, None, None)
        };

        for line in lines.iter().skip(1) {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let additions = if parts[0] == "-" { 0 } else { parts[0].parse::<usize>().unwrap_or(0) };
                let deletions = if parts[1] == "-" { 0 } else { parts[1].parse::<usize>().unwrap_or(0) };
                let file_path = PathBuf::from(parts[2]);
                let diff_content = self.get_commit_file_diff_content(&file_path, commit_hash).await?;

                diffs.push(GitDiffEntry {
                    file_path,
                    change_type: DiffChangeType::Modified,
                    diff_content,
                    line_additions: additions,
                    line_deletions: deletions,
                    commit_hash: commit_info.clone(),
                    commit_message: message.clone(),
                    author: author.clone(),
                    timestamp,
                    old_file_path: None,
                });
            }
        }

        Ok(diffs)
    }

    async fn parse_log_diff_output(&self, output: &str) -> Result<Vec<GitDiffEntry>> {
        let mut diffs = Vec::new();
        let lines: Vec<&str> = output.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];

            if line.contains('|') && line.split('|').count() >= 4 {
                let parts: Vec<&str> = line.split('|').collect();
                let commit_hash = parts[0].to_string();
                let author = parts[1].to_string();
                let timestamp = parts[2].parse::<u64>().ok();
                let message = parts[3].to_string();

                i += 1;

                while i < lines.len() && !lines[i].contains('|') {
                    let file_line = lines[i];
                    if file_line.trim().is_empty() {
                        i += 1;
                        continue;
                    }

                    let parts: Vec<&str> = file_line.split('\t').collect();
                    if parts.len() >= 3 {
                        let additions = if parts[0] == "-" { 0 } else { parts[0].parse::<usize>().unwrap_or(0) };
                        let deletions = if parts[1] == "-" { 0 } else { parts[1].parse::<usize>().unwrap_or(0) };
                        let file_path = PathBuf::from(parts[2]);
                        let diff_content = self.get_commit_file_diff_content(&file_path, &commit_hash).await?;

                        diffs.push(GitDiffEntry {
                            file_path,
                            change_type: DiffChangeType::Modified,
                            diff_content,
                            line_additions: additions,
                            line_deletions: deletions,
                            commit_hash: Some(commit_hash.clone()),
                            commit_message: Some(message.clone()),
                            author: Some(author.clone()),
                            timestamp,
                            old_file_path: None,
                        });
                    }
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        Ok(diffs)
    }

    async fn get_file_diff_content(&self, file_path: &Path, source: &DiffSource) -> Result<String> {
        let mut cmd = AsyncCommand::new("git");
        cmd.arg("diff");

        match source {
            DiffSource::Staged => { cmd.arg("--cached"); }
            DiffSource::Unstaged | DiffSource::BranchComparison => {}
        }

        let output = cmd
            .arg("--")
            .arg(file_path)
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get file diff: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Ok(String::new())
        }
    }

    async fn get_commit_file_diff_content(&self, file_path: &Path, commit_hash: &str) -> Result<String> {
        let output = AsyncCommand::new("git")
            .arg("show")
            .arg(format!("{}:{}", commit_hash, file_path.display()))
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get commit file diff: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Ok(String::new())
        }
    }

    async fn determine_change_type(&self, file_path: &Path, _source: &DiffSource) -> Result<DiffChangeType> {
        let output = AsyncCommand::new("git")
            .arg("status")
            .arg("--porcelain")
            .arg("--")
            .arg(file_path)
            .current_dir(self.repo_path())
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to determine change type: {}", e)))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = stdout.lines().next() {
                let status_code = first_line.chars().take(2).collect::<String>();
                return Ok(match status_code.as_str() {
                    "A " | " A" => DiffChangeType::Added,
                    "D " | " D" => DiffChangeType::Deleted,
                    "R " | " R" => DiffChangeType::Renamed,
                    "C " | " C" => DiffChangeType::Copied,
                    _ => DiffChangeType::Modified,
                });
            }
        }

        Ok(DiffChangeType::Modified)
    }

    async fn filter_diffs(&self, mut diffs: Vec<GitDiffEntry>, config: &DiffAnalysisConfig) -> Result<Vec<GitDiffEntry>> {
        diffs.retain(|diff| {
            !config.ignore_patterns.iter().any(|pattern| {
                if pattern.ends_with("/*") {
                    let prefix = &pattern[..pattern.len() - 2];
                    diff.file_path.to_string_lossy().starts_with(prefix)
                } else if pattern.starts_with("*.") {
                    let suffix = &pattern[1..];
                    diff.file_path.to_string_lossy().ends_with(suffix)
                } else {
                    diff.file_path.to_string_lossy().contains(pattern)
                }
            })
        });

        diffs.retain(|diff| {
            let diff_size_kb = diff.diff_content.len() / 1024;
            diff_size_kb <= config.max_diff_size_kb
        });

        diffs.retain(|diff| {
            let line_count = diff.line_additions + diff.line_deletions;
            line_count <= config.max_lines_per_diff
        });

        if !config.include_binary_diffs {
            diffs.retain(|diff| !is_likely_binary_file(&diff.file_path));
        }

        if !config.include_generated_files {
            diffs.retain(|diff| !is_likely_generated_file(&diff.file_path));
        }

        Ok(diffs)
    }
}

fn is_likely_binary_file(file_path: &Path) -> bool {
    if let Some(extension) = file_path.extension().and_then(|ext| ext.to_str()) {
        matches!(
            extension.to_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "svg" | "pdf" |
            "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" |
            "zip" | "tar" | "gz" | "7z" | "rar" |
            "exe" | "dll" | "so" | "dylib" |
            "mp3" | "mp4" | "avi" | "mov" | "wav"
        )
    } else {
        false
    }
}

fn is_likely_generated_file(file_path: &Path) -> bool {
    let path_str = file_path.to_string_lossy().to_lowercase();
    path_str.contains("generated") ||
    path_str.contains(".generated.") ||
    path_str.contains("node_modules") ||
    path_str.contains("__pycache__") ||
    path_str.contains(".pyc") ||
    path_str.contains("target/") ||
    path_str.contains("build/") ||
    path_str.contains("dist/") ||
    path_str.ends_with(".min.js") ||
    path_str.ends_with(".min.css") ||
    path_str.contains("package-lock.json") ||
    path_str.contains("yarn.lock") ||
    path_str.contains("Cargo.lock")
}
