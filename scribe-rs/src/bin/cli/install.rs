//! Agent installation module
//!
//! Handles configuring AI coding agents (Claude Code, OpenCode, Pi) to use scribe.

use std::fs;
use std::path::PathBuf;

const SCRIBE_MD: &str = include_str!("../../../npm/scribe/scripts/SCRIBE.md");
const HOOK_ENFORCE: &str = include_str!("../../../npm/scribe/scripts/hooks/scribe_enforce.sh");
const HOOK_REMIND: &str = include_str!("../../../npm/scribe/scripts/hooks/scribe_remind.sh");
const PI_EXTENSION: &str = include_str!("../../../npm/scribe/scripts/extensions/scribe.ts");
const OPENCODE_PLUGIN: &str = include_str!("../../../npm/scribe/scripts/plugins/scribe.ts");

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HookMode {
    Block,
    Warn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Agent {
    Claude,
    OpenCode,
    Pi,
    All,
}

impl std::str::FromStr for Agent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "claude" | "claude-code" => Ok(Agent::Claude),
            "opencode" => Ok(Agent::OpenCode),
            "pi" | "pi-coding-agent" => Ok(Agent::Pi),
            "all" => Ok(Agent::All),
            _ => Err(format!("Unknown agent: {}. Use 'claude', 'opencode', 'pi', or 'all'", s)),
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

        // Pi: ~/.pi/ or project-local .pi/
        let pi_dir = home.join(".pi");
        if pi_dir.exists() {
            found.push((Agent::Pi, pi_dir));
        }
    }

    // Also check current directory for .pi/
    if let Ok(cwd) = std::env::current_dir() {
        let local_pi = cwd.join(".pi");
        if local_pi.exists() {
            // Only add if not already in the list
            let already_found = found.iter().any(|(a, p)| *a == Agent::Pi && *p == local_pi);
            if !already_found {
                found.push((Agent::Pi, local_pi));
            }
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

    // 1. Create plugins directory and install plugin
    let plugins_dir = config_dir.join("plugins");
    fs::create_dir_all(&plugins_dir)?;

    let plugin_path = plugins_dir.join("scribe.ts");
    fs::write(&plugin_path, OPENCODE_PLUGIN)?;
    println!("    ✓ Installed scribe.ts plugin");

    // 2. Copy SCRIBE.md
    let scribe_md_path = config_dir.join("SCRIBE.md");
    fs::write(&scribe_md_path, SCRIBE_MD)?;
    println!("    ✓ Installed SCRIBE.md");

    // 3. Update AGENTS.md with include
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

    println!("    Note: Plugin will be loaded automatically by OpenCode");

    Ok(())
}

fn install_pi(config_dir: &PathBuf, _mode: HookMode) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Configuring Pi coding agent...");
    println!("    Config dir: {}", config_dir.display());

    // Ensure directory exists
    fs::create_dir_all(config_dir)?;

    // 1. Create extensions directory and install extension
    let extensions_dir = config_dir.join("extensions");
    fs::create_dir_all(&extensions_dir)?;

    let extension_path = extensions_dir.join("scribe.ts");
    fs::write(&extension_path, PI_EXTENSION)?;
    println!("    ✓ Installed scribe.ts extension");

    // 2. Copy SCRIBE.md for reference
    let scribe_md_path = config_dir.join("SCRIBE.md");
    fs::write(&scribe_md_path, SCRIBE_MD)?;
    println!("    ✓ Installed SCRIBE.md");

    // 3. Update or create AGENTS.md with include
    let agents_md_path = config_dir.join("AGENTS.md");
    let include_content = format!("@{}/SCRIBE.md", config_dir.display());

    if agents_md_path.exists() {
        let content = fs::read_to_string(&agents_md_path)?;
        if !content.contains("SCRIBE.md") {
            let new_content = format!("{}\n\n{}", include_content, content);
            fs::write(&agents_md_path, new_content)?;
            println!("    ✓ Added SCRIBE.md include to AGENTS.md");
        } else {
            println!("    ✓ AGENTS.md already includes SCRIBE.md");
        }
    } else {
        fs::write(&agents_md_path, format!("{}\n", include_content))?;
        println!("    ✓ Created AGENTS.md with SCRIBE.md include");
    }

    println!("    Note: Extension will be loaded automatically by Pi agent");

    Ok(())
}

pub fn run_install(agent: Agent, mode: HookMode) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Scribe - AI Coding Agent Integration");
    println!("══════════════════════════════════════════════════════════════\n");

    let detected = detect_agents();

    if detected.is_empty() && agent == Agent::All {
        println!("No AI coding agents detected.");
        println!("Supported agents:");
        println!("  - Claude Code (~/.claude)");
        println!("  - OpenCode (~/.config/opencode)");
        println!("  - Pi coding agent (~/.pi or .pi/)");
        println!("\nTo install for a specific agent anyway, use:");
        println!("  scribe --install claude");
        println!("  scribe --install opencode");
        println!("  scribe --install pi");
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
        Agent::Pi => {
            // Prefer local .pi/ if it exists, otherwise use ~/.pi/
            let local_pi = std::env::current_dir()?.join(".pi");
            let dir = if local_pi.exists() {
                local_pi
            } else {
                let home = home_dir().ok_or("Could not determine home directory")?;
                home.join(".pi")
            };
            fs::create_dir_all(&dir)?;
            vec![(Agent::Pi, dir)]
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
            Agent::Pi => install_pi(config_dir, mode)?,
            Agent::All => unreachable!(),
        }
        println!();
    }

    println!("══════════════════════════════════════════════════════════════");
    println!("  Installation complete!");
    println!("══════════════════════════════════════════════════════════════\n");

    println!("Scribe is now configured with surgical context guidance:");
    println!("  - Use --max-depth 1 for tight focus");
    println!("  - Use --token-target 800 for small slices");
    println!("  - Multiple small calls > one large call");
    println!();

    Ok(())
}
