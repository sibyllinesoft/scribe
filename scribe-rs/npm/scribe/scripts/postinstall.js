#!/usr/bin/env node
/**
 * Scribe post-install script
 *
 * Detects installed AI coding agents (Claude Code, OpenCode) and configures
 * them to use scribe for code context retrieval.
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const os = require("os");

const HOME = os.homedir();
const SCRIPTS_DIR = __dirname;

// ============================================================================
// Agent Detection
// ============================================================================

function detectAgents() {
  const agents = [];

  // Claude Code: check for ~/.claude/
  const claudeDir = path.join(HOME, ".claude");
  if (fs.existsSync(claudeDir)) {
    agents.push({
      name: "Claude Code",
      type: "claude",
      configDir: claudeDir,
    });
  }

  // OpenCode: check for ~/.config/opencode/ or ~/.opencode/
  const opencodeConfigDir = path.join(HOME, ".config", "opencode");
  const opencodeDir = path.join(HOME, ".opencode");
  if (fs.existsSync(opencodeConfigDir)) {
    agents.push({
      name: "OpenCode",
      type: "opencode",
      configDir: opencodeConfigDir,
    });
  } else if (fs.existsSync(opencodeDir)) {
    agents.push({
      name: "OpenCode",
      type: "opencode",
      configDir: opencodeConfigDir, // Use standard config location
    });
  }

  return agents;
}

// ============================================================================
// Interactive Prompting
// ============================================================================

function isInteractive() {
  return process.stdin.isTTY && process.stdout.isTTY;
}

async function promptHookMode() {
  // Non-interactive: default to block mode
  if (!isInteractive()) {
    return "block";
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    console.log("\nHow should scribe hooks handle Read/Grep on code files?\n");
    console.log("  [1] Block and redirect to scribe (Recommended)");
    console.log("      Blocks Read/Grep on code, tells agent to use scribe.");
    console.log("      More effective - agents learn faster, use fewer tokens.\n");
    console.log("  [2] Warn but allow");
    console.log("      Shows reminder to use scribe but allows the operation.\n");

    rl.question("Enter choice [1]: ", (answer) => {
      rl.close();
      const choice = answer.trim();
      if (choice === "2") {
        resolve("warn");
      } else {
        resolve("block");
      }
    });
  });
}

// ============================================================================
// Claude Code Configuration
// ============================================================================

function configureClaudeCode(configDir, mode) {
  console.log("Configuring Claude Code...");

  // 1. Copy SCRIBE.md
  const scribeMdSrc = path.join(SCRIPTS_DIR, "SCRIBE.md");
  const scribeMdDest = path.join(configDir, "SCRIBE.md");
  fs.copyFileSync(scribeMdSrc, scribeMdDest);

  // 2. Create hooks directory and copy hook script
  const hooksDir = path.join(configDir, "hooks");
  if (!fs.existsSync(hooksDir)) {
    fs.mkdirSync(hooksDir, { recursive: true });
  }

  const hookScriptName = mode === "block" ? "scribe_enforce.sh" : "scribe_remind.sh";
  const hookSrc = path.join(SCRIPTS_DIR, "hooks", hookScriptName);
  const hookDest = path.join(hooksDir, "scribe_hook.sh");
  fs.copyFileSync(hookSrc, hookDest);
  fs.chmodSync(hookDest, 0o755);

  // 3. Update settings.json (merge with existing)
  const settingsPath = path.join(configDir, "settings.json");
  let settings = {};

  if (fs.existsSync(settingsPath)) {
    try {
      settings = JSON.parse(fs.readFileSync(settingsPath, "utf8"));
    } catch (e) {
      console.warn("  Warning: Could not parse existing settings.json, creating new one");
    }
  }

  // Ensure hooks structure exists
  if (!settings.hooks) {
    settings.hooks = {};
  }
  if (!settings.hooks.PreToolUse) {
    settings.hooks.PreToolUse = [];
  }

  // Remove any existing scribe hook
  settings.hooks.PreToolUse = settings.hooks.PreToolUse.filter(
    (h) => !h.hooks?.some((hh) => hh.command?.includes("scribe_hook.sh"))
  );

  // Add scribe hook
  settings.hooks.PreToolUse.push({
    matcher: "Read|Grep",
    hooks: [
      {
        type: "command",
        command: `bash "${hookDest}"`,
      },
    ],
  });

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");

  // 4. Update CLAUDE.md with include (if not already present)
  const claudeMdPath = path.join(configDir, "CLAUDE.md");
  const includeDirective = "@~/.claude/SCRIBE.md";

  if (fs.existsSync(claudeMdPath)) {
    let content = fs.readFileSync(claudeMdPath, "utf8");
    if (!content.includes(includeDirective) && !content.includes("SCRIBE.md")) {
      // Add include at the top
      content = includeDirective + "\n\n" + content;
      fs.writeFileSync(claudeMdPath, content);
    }
  } else {
    // Create new CLAUDE.md with just the include
    fs.writeFileSync(claudeMdPath, includeDirective + "\n");
  }

  console.log("  Done");
}

// ============================================================================
// OpenCode Configuration
// ============================================================================

function configureOpenCode(configDir, mode) {
  console.log("Configuring OpenCode...");

  // Ensure config directory exists
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }

  // 1. Copy SCRIBE.md
  const scribeMdSrc = path.join(SCRIPTS_DIR, "SCRIBE.md");
  const scribeMdDest = path.join(configDir, "SCRIBE.md");
  fs.copyFileSync(scribeMdSrc, scribeMdDest);

  // 2. Update AGENTS.md with include (if not already present)
  const agentsMdPath = path.join(configDir, "AGENTS.md");
  const includeDirective = "@~/.config/opencode/SCRIBE.md";

  if (fs.existsSync(agentsMdPath)) {
    let content = fs.readFileSync(agentsMdPath, "utf8");
    if (!content.includes(includeDirective) && !content.includes("SCRIBE.md")) {
      // Add include at the top
      content = includeDirective + "\n\n" + content;
      fs.writeFileSync(agentsMdPath, content);
    }
  } else {
    // Create new AGENTS.md with just the include
    fs.writeFileSync(agentsMdPath, includeDirective + "\n");
  }

  // Note: OpenCode uses a plugin system for hooks, which is more complex.
  // For now, we rely on the AGENTS.md instructions. Full hook support would
  // require creating a JS/TS plugin.
  console.log("  Done (instructions added; OpenCode hooks require plugin setup)");
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const agents = detectAgents();

  if (agents.length === 0) {
    // No agents detected - silent exit (common case for CI/servers)
    return;
  }

  console.log("\n" + "=".repeat(60));
  console.log("Scribe - AI Coding Agent Integration");
  console.log("=".repeat(60));
  console.log("\nDetected AI coding agents:");
  agents.forEach((a) => console.log(`  - ${a.name} (${a.configDir})`));

  const mode = await promptHookMode();
  console.log(`\nUsing ${mode === "block" ? "block" : "warn"} mode for hooks.\n`);

  for (const agent of agents) {
    try {
      if (agent.type === "claude") {
        configureClaudeCode(agent.configDir, mode);
      } else if (agent.type === "opencode") {
        configureOpenCode(agent.configDir, mode);
      }
    } catch (err) {
      console.error(`  Error configuring ${agent.name}: ${err.message}`);
    }
  }

  console.log("\nScribe is now integrated with your AI coding agents!");
  console.log("See ~/.claude/SCRIBE.md for usage instructions.\n");
}

main().catch((err) => {
  // Don't fail npm install on postinstall errors
  console.error("Scribe postinstall warning:", err.message);
});
