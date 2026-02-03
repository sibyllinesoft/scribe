/**
 * Scribe Guidance Plugin for OpenCode
 *
 * Guides agent toward targeted, surgical scribe usage.
 * Philosophy: Multiple small focused slices > few large dumps
 */

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

// Track metrics
let readBlocks = 0;
let grepBlocks = 0;
let scribeCalls = 0;
let totalReads = 0;
let totalGreps = 0;

export const ScribeGuidance = async ({ project, client, $, directory, worktree }: any) => {
	return {
		"tool.execute.before": async (input: any, output: any) => {
			const toolName = input.tool?.toLowerCase() || "";

			// Handle read tool
			if (toolName === "read") {
				totalReads++;
				const filePath = output.args?.filePath || output.args?.file_path || "";

				// Allow config files
				if (isConfigFile(filePath)) {
					return;
				}

				// Block code files, suggest scribe
				if (isCodeFile(filePath)) {
					readBlocks++;
					throw new Error(`BLOCKED: ${SCRIBE_GUIDANCE}`);
				}
			}

			// Handle grep tool
			if (toolName === "grep") {
				totalGreps++;
				const grepPath = output.args?.path || "";

				// If grepping a specific code file, suggest scribe
				if (grepPath && isCodeFile(grepPath)) {
					grepBlocks++;
					throw new Error(`BLOCKED: ${SCRIBE_GUIDANCE}`);
				}
			}

			// Track scribe calls via bash
			if (toolName === "bash") {
				const cmd = output.args?.command || "";
				if (/^\s*scribe\s/i.test(cmd) || /\|\s*scribe\s/i.test(cmd)) {
					scribeCalls++;
				}
			}
		},

		"session.error": async () => {
			console.error(`[scribe-plugin] Stats: reads=${totalReads}, readBlocks=${readBlocks}, greps=${totalGreps}, grepBlocks=${grepBlocks}, scribeCalls=${scribeCalls}`);
		}
	};
};

export default ScribeGuidance;
