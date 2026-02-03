/**
 * Scribe Guidance Extension for Pi Coding Agent
 *
 * Guides agent toward targeted, surgical scribe usage.
 * Philosophy: Multiple small focused slices > few large dumps
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// Code file extensions
const CODE_EXTS = new Set([
	".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
	".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc",
	".rb", ".php", ".swift", ".scala", ".cs", ".lua", ".ex", ".exs",
	".hs", ".ml", ".clj", ".vue", ".svelte", ".sol"
]);

// Config/doc extensions that don't need scribe
const CONFIG_EXTS = new Set([
	".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
	".cfg", ".xml", ".html", ".css", ".env", ".gitignore", ".lock"
]);

function isCodeFile(path: string): boolean {
	const ext = path.substring(path.lastIndexOf(".")).toLowerCase();
	return CODE_EXTS.has(ext);
}

function isConfigFile(path: string): boolean {
	const ext = path.substring(path.lastIndexOf(".")).toLowerCase();
	return CONFIG_EXTS.has(ext);
}

// Surgical scribe guidance
const SCRIBE_GUIDANCE = `Use scribe for surgical code slices with dependencies.

PATTERN: Start small, expand if needed
  1. First call: Get just the target function (--max-depth 1 --token-target 800)
  2. If you need more context: Get a specific dependency (--max-depth 1)
  3. Repeat for each piece you need

COMMAND FORMAT:
  scribe --covering-set "FILE:FUNCTION" --max-depth 1 --token-target 800 --stdout

EXAMPLES:
  # Get just the handler function
  scribe --covering-set "api/handler.go:HandleRequest" --max-depth 1 --token-target 800 --stdout

  # Then get a helper it calls
  scribe --covering-set "api/validate.go:ValidateInput" --max-depth 1 --token-target 800 --stdout

KEY PRINCIPLES:
- Use --max-depth 1 for tight focus (only direct dependencies)
- Use --token-target 800 for small slices
- Multiple small calls > one large call
- Target specific functions, never whole files`;

export default function (pi: ExtensionAPI) {
	// Track metrics
	let readBlocks = 0;
	let grepBlocks = 0;
	let scribeCalls = 0;
	let totalReads = 0;
	let totalGreps = 0;

	// Intercept Read tool calls
	pi.on("tool_call", (event, ctx) => {
		if (event.toolName === "read") {
			totalReads++;
			const filePath = event.input.path;

			// Allow config files
			if (isConfigFile(filePath)) {
				return {};
			}

			// Block code files, suggest scribe
			if (isCodeFile(filePath)) {
				readBlocks++;
				return {
					block: true,
					reason: SCRIBE_GUIDANCE
				};
			}

			return {};
		}

		if (event.toolName === "grep") {
			totalGreps++;
			const grepPath = event.input.path;

			// If grepping a specific code file, suggest scribe
			if (grepPath && isCodeFile(grepPath)) {
				grepBlocks++;
				return {
					block: true,
					reason: SCRIBE_GUIDANCE
				};
			}

			return {};
		}

		// Track scribe calls via bash
		if (event.toolName === "bash") {
			const cmd = event.input.command || "";
			if (/^\s*scribe\s/i.test(cmd) || /\|\s*scribe\s/i.test(cmd)) {
				scribeCalls++;
			}
			return {};
		}

		return {};
	});

	// Register command to show metrics
	pi.registerCommand("scribe-stats", {
		description: "Show scribe enforcement statistics",
		handler: async (_args, ctx) => {
			const stats = `Scribe Stats:
  Read calls: ${totalReads} (blocked: ${readBlocks})
  Grep calls: ${totalGreps} (blocked: ${grepBlocks})
  Scribe calls: ${scribeCalls}`;

			ctx.ui.notify(stats, "info");
		}
	});

	// Log stats on session end
	pi.on("agent_end", () => {
		console.error(`[scribe-ext] Stats: reads=${totalReads}, readBlocks=${readBlocks}, greps=${totalGreps}, grepBlocks=${grepBlocks}, scribeCalls=${scribeCalls}`);
	});
}
