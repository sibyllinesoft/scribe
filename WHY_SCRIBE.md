# Why Scribe?

## TL;DR

**Scribe is the intelligent repository bundler for developers who need precision, not just packaging.** While other tools simply concatenate files, Scribe uses research-grade graph algorithms, surgical entity selection, and transformer-aware context positioning to build bundles that maximize LLM reasoning quality. Built in Rust for production-grade performance, Scribe analyzes 100k+ file repositories in under 30 seconds while intelligently prioritizing what matters—no babysitting required.

**Use Scribe when you need:**
- Surgical precision to extract only the files needed to understand a specific function or class
- Intelligent file prioritization using PageRank centrality analysis (not naive heuristics)
- Context optimization that exploits transformer attention patterns for better LLM reasoning
- Production-grade performance: sub-second analysis on small repos, <30s on enterprise codebases
- Transparent, explainable decisions about what was included and why

## Quickstart

```bash
# Install from source
cargo install --path scribe-rs --locked

# Generate a Markdown bundle for your repository
scribe --style markdown --output bundle.md

# Create an interactive HTML editor to review and customize your bundle
scribe --style html --editor --output bundle.html

# Surgical selection: Get only files needed to understand a specific function
scribe --covering-set "authenticate_user" --entity-type function --max-files 20

# Use with custom token budget
scribe --token-budget 100000 --style markdown
```

Run `scribe --help` to see available algorithms, token budgeting controls, Git integration flags, and output formats.

## Why I Created Scribe

When feeding codebases to LLMs like Claude or ChatGPT, I kept hitting the same frustrations with existing tools:

### 1. **They Weren't Actually Intelligent**

Most repository bundlers are glorified file concatenators. They either:
- Dump entire repositories without prioritization, wasting precious context on boilerplate
- Use simplistic heuristics (file size, alphabetical order) that miss critical dependencies
- Require manual file selection, forcing you to babysit the process

**What I wanted:** A tool that truly understands code structure through AST parsing and dependency analysis, then uses graph algorithms (like PageRank) to identify what's genuinely important—just like Google ranks web pages.

### 2. **They Didn't Optimize for LLM Recall**

Transformers don't attend equally to all tokens in their context window. Research shows they focus heavily on the beginning and end, with the middle getting less attention. Yet existing tools treat all positions equally, placing important files wherever they happen to fall.

**What I wanted:** Intelligent context positioning that exploits transformer architecture—high-priority, query-relevant files at the HEAD (where attention is strongest), supporting context in the MIDDLE, and core functionality at the TAIL (where attention picks up again). This 3-tier positioning maximizes LLM reasoning quality.

### 3. **They Were Too Slow**

When you're iterating on a problem, waiting 30+ seconds for a tool to analyze a medium-sized repository kills flow. Existing tools written in Python or JavaScript can't handle enterprise-scale codebases efficiently.

**What I wanted:** Rust-level performance with parallel processing—sub-second analysis on small repos, ~5 seconds on medium projects, and under 30 seconds even on 100k+ file enterprises. Fast enough to integrate into your actual workflow.

### 4. **They Lacked Surgical Precision**

Sometimes you don't need the whole repository—you need to understand one function, or analyze the impact of changing a specific class. Existing tools force you to either include everything or manually hunt down dependencies.

**What I wanted:** "Covering set selection" that can target a specific entity (function, class, module), automatically compute its transitive dependencies and dependents, and return the minimal set of files needed for understanding or impact analysis. Get exactly what you need, nothing more.

### 5. **They Didn't Degrade Gracefully**

Hit your token budget and most tools either fail or arbitrarily truncate files, losing critical context. There's no intelligence about what to preserve.

**What I wanted:** Progressive demotion that intelligently reduces content when approaching limits: FULL → CHUNK (keep important sections via AST) → SIGNATURE (preserve type definitions and interfaces). Maintain maximum information density within any budget.

## The Competitive Landscape

The repository bundler ecosystem has exploded with 36+ tools trying to solve this problem, indicating both its importance and the shortcomings of existing solutions. Here's how Scribe compares to the major players:

### Repomix (20k+ stars) – The Popular Choice

**What it does well:**
- Excellent adoption and community support
- Multiple output formats (XML, Markdown, JSON)
- Good security with Secretlint integration
- Tree-sitter compression support
- Easy to use with sensible defaults
- Browser extensions for GitHub

**Where Scribe differs:**
- **Intelligence:** Repomix concatenates files with optional compression. Scribe uses PageRank centrality, dependency analysis, and multi-tier scoring to understand what's actually important.
- **Precision:** Repomix is all-or-nothing. Scribe offers surgical covering set selection to target specific entities and compute minimal dependency closures.
- **Positioning:** Repomix doesn't optimize file order for LLM attention patterns. Scribe's context positioning exploits transformer architecture for better reasoning.
- **Performance:** Repomix is Node.js-based. Scribe's Rust core with parallel processing handles 100k+ file repos in <30s.
- **Progressive degradation:** Repomix truncates or fails at budget limits. Scribe intelligently demotes content (full → chunks → signatures) to maximize information density.

**When to choose Repomix:** You need a battle-tested tool with broad ecosystem support, browser integration, and remote repository support. You're okay with simpler selection logic.

**When to choose Scribe:** You need intelligent prioritization, surgical precision for specific entities, transformer-aware positioning, or enterprise-scale performance.

### Code2Prompt (6.7k stars) – The Rust Alternative

**What it does well:**
- Fast Rust implementation
- Source tree visualization
- Prompt templating system
- Token counting
- Good documentation

**Where Scribe differs:**
- **Graph algorithms:** Code2Prompt doesn't compute centrality or dependency importance. Scribe uses research-grade PageRank and transitive closure computation.
- **Semantic understanding:** Code2Prompt focuses on file-level operations. Scribe parses AST for entity-level targeting and semantic chunking.
- **Context optimization:** Code2Prompt doesn't position files strategically. Scribe's 3-tier positioning maximizes LLM attention utilization.
- **Selection sophistication:** Code2Prompt offers template-based filtering. Scribe provides multiple algorithms (simple, complex, covering-set) with explainable inclusion reasons.

**When to choose Code2Prompt:** You want Rust performance with template-based customization and primarily need file concatenation with good organization.

**When to choose Scribe:** You need semantic understanding, graph-based importance ranking, or entity-targeted selection with dependency analysis.

### bundle-repo and Others (Various) – The Niche Players

The ecosystem includes dozens of smaller tools, each with specific focuses:
- **GPT-Repository-Loader** (3k stars): Python-based, early solution, simpler feature set
- **bundle-repo**: Rust-based XML bundler with remote repo support
- **LLM Context**: Focuses on flexible rule systems and MCP integration
- Dozens more in TypeScript, Python, PowerShell, etc.

**Common pattern:** Most tools focus on packaging and formatting. Few address the core intelligence problems: understanding code structure, prioritizing intelligently, or optimizing for LLM cognition.

**Scribe's distinction:** Treats repository bundling as a graph algorithms and information retrieval problem, not just a file concatenation task.

## What Makes Scribe Unique

### 1. Research-Grade Graph Algorithms
- **PageRank centrality** adapted specifically for code dependency graphs
- Transitive dependency/dependent computation for surgical selection
- Strongly connected component detection (Kosaraju's algorithm)
- Configurable damping, convergence detection, parallel computation
- Performance: 10ms for small repos, ~100ms for large ones

### 2. Surgical Covering Set Selection
- Target specific functions, classes, or modules by name
- Automatic transitive closure computation with configurable depth
- Minimal file sets for understanding or impact analysis
- Explainable inclusion reasons (target, direct dependency, transitive, etc.)
- Multi-language AST support (Python, JS/TS, Rust, Go, Java)

### 3. Transformer-Aware Context Positioning
- **HEAD (20%):** Query-relevant high-centrality files where transformers attend best
- **MIDDLE (60%):** Supporting context with lower attention
- **TAIL (20%):** Core functionality where attention strengthens again
- Query-aware relevance scoring and file grouping
- Configurable weights and tier sizes

### 4. Progressive Content Demotion
When approaching token budgets:
- **FULL:** Complete file content
- **CHUNK:** AST-based semantic sections (preserve important functions/classes)
- **SIGNATURE:** Type signatures and interfaces only
- Achieves 3-10x compression while preserving intent
- Quality scoring tracks information preservation

### 5. Multi-Dimensional File Scoring
Sophisticated heuristic combining:
- Documentation coverage
- PageRank centrality in import graph
- Test-to-source linkage detection
- Git churn (change frequency and recency)
- Path depth and entrypoint detection
- Template/boilerplate demotion
- Configurable weight presets

### 6. Production-Grade Performance
- **Time:** Small repos <1s, medium ~5s, large ~15s, 100k+ files <30s
- **Memory:** Scales from 50MB to ~2GB based on repo size
- **Parallelism:** Rayon-based multi-core processing
- **Caching:** Persistent caching with signature-based invalidation
- **Streaming:** Progressive loading avoids memory overload

### 7. Transparent and Explainable
- Detailed inclusion reasons for every selected file
- Scoring breakdown showing why files were prioritized
- Quality metrics (compression ratio, information preservation)
- Multiple output formats with full metadata

## When Should You Use Scribe?

**Choose Scribe if you:**
- Need to understand specific functions/classes without analyzing entire repositories
- Want intelligent prioritization based on code structure, not guesswork
- Are working with large codebases where performance matters
- Need to maximize LLM reasoning quality within context budgets
- Value transparency about what was included and why
- Require production-grade reliability and performance

**Choose alternatives if you:**
- Need browser extensions and GitHub integration (Repomix)
- Want MCP protocol support and flexible rule systems (LLM Context)
- Primarily need simple file concatenation with templates (Code2Prompt)
- Require remote repository support without local cloning (various tools)
- Want battle-tested solutions with massive community support (Repomix)

## The Bottom Line

**Scribe exists because repository bundling is an information retrieval problem, not just a file concatenation task.** The difference between dumping code into an LLM and intelligently curating context is the difference between mediocre and exceptional results.

If you're serious about leveraging LLMs for code understanding, and you need tools that respect both your time and your token budget, Scribe was built for you.

---

*Built with Rust. Powered by graph algorithms. Optimized for LLM reasoning.*
