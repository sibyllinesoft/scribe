//! Agent installation module
//!
//! Handles configuring AI coding agents (Claude Code, OpenCode) to use scribe.

use std::fs;
use std::path::PathBuf;

const SCRIBE_MD: &str = include_str!("../../../npm/scribe/scripts/SCRIBE.md");
const HOOK_ENFORCE: &str = include_str!("../../../npm/scribe/scripts/hooks/scribe_enforce.sh");
const HOOK_REMIND: &str = include_str!("../../../npm/scribe/scripts/hooks/scribe_remind.sh");

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HookMode {
    Block,
    Warn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Agent {
    Claude,
    OpenCode,
    All,
}

impl std::str::FromStr for Agent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "claude" | "claude-code" => Ok(Agent::Claude),
            "opencode" => Ok(Agent::OpenCode),
            "all" => Ok(Agent::All),
            _ => Err(format!("Unknown agent: {}. Use 'claude', 'opencode', or 'all'", s)),
        }
    }
}

fn home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

fn detect_agents() -> Vec<(Agent, PathBuf)> {
    let mut found = Vec::new();

    if let Some(home) = home_dir() {
        // Claude Code: ~/.claude/
        let claude_dir = home.join(".claude");
        if claude_dir.exists() {
            found.push((Agent::Claude, claude_dir));
        }

        // OpenCode: ~/.config/opencode/
        let opencode_dir = home.join(".config").join("opencode");
        if opencode_dir.exists() {
            found.push((Agent::OpenCode, opencode_dir));
        }
    }

    found
}

fn install_claude(config_dir: &PathBuf, mode: HookMode) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Configuring Claude Code...");

    // 1. Copy SCRIBE.md
    let scribe_md_path = config_dir.join("SCRIBE.md");
    fs::write(&scribe_md_path, SCRIBE_MD)?;
    println!("    ✓ Installed SCRIBE.md");

    // 2. Create hooks directory and install hook
    let hooks_dir = config_dir.join("hooks");
    fs::create_dir_all(&hooks_dir)?;

    let hook_content = match mode {
        HookMode::Block => HOOK_ENFORCE,
        HookMode::Warn => HOOK_REMIND,
    };
    let hook_path = hooks_dir.join("scribe_hook.sh");
    fs::write(&hook_path, hook_content)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&hook_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&hook_path, perms)?;
    }
    println!("    ✓ Installed hook script ({} mode)", if mode == HookMode::Block { "block" } else { "warn" });

    // 3. Update settings.json
    let settings_path = config_dir.join("settings.json");
    let mut settings: serde_json::Value = if settings_path.exists() {
        let content = fs::read_to_string(&settings_path)?;
        serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    // Ensure hooks structure
    if settings.get("hooks").is_none() {
        settings["hooks"] = serde_json::json!({});
    }
    if settings["hooks"].get("PreToolUse").is_none() {
        settings["hooks"]["PreToolUse"] = serde_json::json!([]);
    }

    // Remove existing scribe hook
    if let Some(arr) = settings["hooks"]["PreToolUse"].as_array_mut() {
        arr.retain(|h| {
            if let Some(hooks) = h.get("hooks").and_then(|h| h.as_array()) {
                !hooks.iter().any(|hh| {
                    hh.get("command")
                        .and_then(|c| c.as_str())
                        .map(|s| s.contains("scribe_hook.sh"))
                        .unwrap_or(false)
                })
            } else {
                true
            }
        });

        // Add new scribe hook
        arr.push(serde_json::json!({
            "matcher": "Read|Grep",
            "hooks": [{
                "type": "command",
                "command": format!("bash \"{}\"", hook_path.display())
            }]
        }));
    }

    fs::write(&settings_path, serde_json::to_string_pretty(&settings)? + "\n")?;
    println!("    ✓ Updated settings.json");

    // 4. Update CLAUDE.md with include
    let claude_md_path = config_dir.join("CLAUDE.md");
    let include_directive = "@~/.claude/SCRIBE.md";

    if claude_md_path.exists() {
        let content = fs::read_to_string(&claude_md_path)?;
        if !content.contains(include_directive) && !content.contains("SCRIBE.md") {
            let new_content = format!("{}\n\n{}", include_directive, content);
            fs::write(&claude_md_path, new_content)?;
            println!("    ✓ Added SCRIBE.md include to CLAUDE.md");
        } else {
            println!("    ✓ CLAUDE.md already includes SCRIBE.md");
        }
    } else {
        fs::write(&claude_md_path, format!("{}\n", include_directive))?;
        println!("    ✓ Created CLAUDE.md with SCRIBE.md include");
    }

    Ok(())
}

fn install_opencode(config_dir: &PathBuf, _mode: HookMode) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Configuring OpenCode...");

    // Ensure directory exists
    fs::create_dir_all(config_dir)?;

    // 1. Copy SCRIBE.md
    let scribe_md_path = config_dir.join("SCRIBE.md");
    fs::write(&scribe_md_path, SCRIBE_MD)?;
    println!("    ✓ Installed SCRIBE.md");

    // 2. Update AGENTS.md with include
    let agents_md_path = config_dir.join("AGENTS.md");
    let include_directive = "@~/.config/opencode/SCRIBE.md";

    if agents_md_path.exists() {
        let content = fs::read_to_string(&agents_md_path)?;
        if !content.contains(include_directive) && !content.contains("SCRIBE.md") {
            let new_content = format!("{}\n\n{}", include_directive, content);
            fs::write(&agents_md_path, new_content)?;
            println!("    ✓ Added SCRIBE.md include to AGENTS.md");
        } else {
            println!("    ✓ AGENTS.md already includes SCRIBE.md");
        }
    } else {
        fs::write(&agents_md_path, format!("{}\n", include_directive))?;
        println!("    ✓ Created AGENTS.md with SCRIBE.md include");
    }

    println!("    Note: OpenCode hooks require plugin setup");

    Ok(())
}

pub fn run_install(agent: Agent, mode: HookMode) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Scribe - AI Coding Agent Integration");
    println!("══════════════════════════════════════════════════════════════\n");

    let detected = detect_agents();

    if detected.is_empty() && agent == Agent::All {
        println!("No AI coding agents detected.");
        println!("Supported agents: Claude Code (~/.claude), OpenCode (~/.config/opencode)");
        println!("\nTo install for a specific agent anyway, use:");
        println!("  scribe --install claude");
        println!("  scribe --install opencode");
        return Ok(());
    }

    let agents_to_install: Vec<(Agent, PathBuf)> = match agent {
        Agent::All => detected,
        Agent::Claude => {
            let home = home_dir().ok_or("Could not determine home directory")?;
            let dir = home.join(".claude");
            fs::create_dir_all(&dir)?;
            vec![(Agent::Claude, dir)]
        }
        Agent::OpenCode => {
            let home = home_dir().ok_or("Could not determine home directory")?;
            let dir = home.join(".config").join("opencode");
            fs::create_dir_all(&dir)?;
            vec![(Agent::OpenCode, dir)]
        }
    };

    if agents_to_install.is_empty() {
        println!("No agents to install.");
        return Ok(());
    }

    println!("Installing for {} agent(s) in {} mode:\n",
             agents_to_install.len(),
             if mode == HookMode::Block { "block" } else { "warn" });

    for (agent_type, config_dir) in &agents_to_install {
        match agent_type {
            Agent::Claude => install_claude(config_dir, mode)?,
            Agent::OpenCode => install_opencode(config_dir, mode)?,
            Agent::All => unreachable!(),
        }
        println!();
    }

    println!("══════════════════════════════════════════════════════════════");
    println!("  Installation complete!");
    println!("══════════════════════════════════════════════════════════════\n");

    Ok(())
}
