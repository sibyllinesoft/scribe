### Summary of Recommendations

1.  **Cofounder Dispatch:** Modify `AGENTS.md` and `RULES.md` to create explicit, mutually exclusive conditions for invoking the `cofounder` vs. the `parallel-worker`. The `cofounder` should only be triggered by ambiguity and strategic needs, while the `parallel-worker` handles pre-defined plans.
2.  **Parallel Execution:** Update the core instructions for the `parallel-worker` agent to mandate the use of a non-blocking `Task` tool for spawning sub-agents, and explicitly forbid it from implementing any logic itself.
3.  **Agent Description Tuning:** Apply a set of principles to your agent descriptions in `AGENTS.md` to make them more robust for the dispatcher. This involves focusing on roles and outcomes, using strong action verbs, and defining clear exclusions to prevent incorrect routing based on random keywords.

---

### 1. Tuning the `cofounder` vs. `parallel-worker` Dispatch Logic

The issue is that the trigger for `cofounder` is too broad. We need to make the dispatcher's choice deterministic based on the presence or absence of a clear plan.

#### Recommended Changes for `AGENTS.md`

In the `<orchestrationLogic>` section, the conditions are not mutually exclusive. A user could provide a detailed plan that still contains a "vague goal" within it. Let's make the logic stricter.

```diff
--- AGENTS.md
+++ AGENTS.md
<orchestrationLogic>
  <decision>
-    <condition>User provides a vague, high-level, or ambiguous goal.</condition>
+    <condition>User provides a vague, high-level, or ambiguous goal AND no detailed execution plan (e.g., genesis.xml, epic.md) is provided.</condition>
     <action>Invoke **`cofounder`** to perform strategic analysis and create a detailed plan.</action>
     <example>"Build a social media app."</example>
   </decision>
   <decision>
-    <condition>User provides a detailed plan, a `TODO.md`, or a well-defined epic.</condition>
+    <condition>A detailed execution plan (e.g., genesis.xml, epic.md, TODO.md) already exists or is provided by the user.</condition>
     <action>Invoke **`parallel-worker`** to execute the existing plan.</action>
     <example>"Implement the features outlined in `epics/feature-x/epic.md`."</example>
   </decision>
   <decision>
     <condition>Project is complete and needs comprehensive documentation and closure.</condition>
     <action>Invoke **`project-shipper`** to generate post-flight recap using genesis.xml data.</action>
     <example>"Create a comprehensive recap of the Living Blueprint implementation."</example>
   </decision>
 </orchestrationLogic>
```

#### Recommended Changes for `RULES.md`

Let's refine the trigger keywords to be more specific and less likely to overlap. We will make the `cofounder` triggers exclusively about strategy and ambiguity.

```diff
--- RULES.md
+++ RULES.md
<coordination_agents>
-    <trigger keywords="ambiguous, unclear, strategic analysis">cofounder</trigger>
+    <trigger keywords="strategic plan, business goal, vague idea, high-level concept, requirements analysis">cofounder</trigger>
-    <trigger keywords="complex, multi-step, coordinate">studio-producer</trigger>
+    <trigger keywords="coordinate, execute plan, manage tasks, run epic">studio-producer</trigger>
     <trigger keywords="analyze, investigate, research">domain-specific + sequential-thinking</trigger>
   </coordination_agents>
 </trigger_matrix>
```

**Result:** These changes create a clear decision boundary. If a plan exists, the `parallel-worker` is used. If not, and the goal is ambiguous, the `cofounder` is called. This prevents the `cofounder` from re-analyzing work that has already been planned.

---

### 2. Enforcing Parallelism and Delegation for `parallel-worker`

The `parallel-worker` must be a pure delegator. It should never perform implementation tasks, and it must use non-blocking tools to achieve parallelism. This requires modifying its core instructions and description.

#### Recommended Changes for `AGENTS.md`

Update the `parallel-worker`'s description in the "Studio Management" department to be extremely clear about its role.

```diff
--- AGENTS.md
+++ AGENTS.md
 <department name="Studio Management">
   <agent id="cofounder" role="Strategic Head - Socratic questioning, requirement clarification, strategic brief creation. For ambiguous goals, outputs strategic-brief.md for 'hydra plan' processing."/>
   <agent id="plan-generator" role="Living Blueprint Architect - Transforms strategic briefs into detailed genesis.xml files with execution DAGs. Core component of 'hydra plan' workflow."/>
   <agent id="studio-producer" role="Tactical Orchestrator - Execution management, resource allocation, timeline coordination. Primary orchestrator for 'hydra run' operations."/>
   <agent id="project-shipper" role="Delivery Manager - Living Blueprint recap specialist. Reads completed genesis.xml files to generate comprehensive project documentation via 'hydra recap'."/>
-  <agent id="parallel-worker" role="Technical Execution Engine - Genesis.xml-driven execution coordinator. Reads DAGs from genesis.xml and orchestrates specialist agents for parallel task completion."/>
+  <agent id="parallel-worker" role="Technical Execution Engine &amp; Delegator - Reads execution DAGs from genesis.xml and orchestrates specialist agents. MUST use the non-blocking 'Task' tool to spawn agents in parallel. MUST NOT write any implementation code, business logic, or tests itself. Its only role is to delegate and monitor."/>
 </department>
```

#### Add a "Core Protocol" Instruction for the Agent

You should ensure the system prompt for the `parallel-worker` agent contains the following non-negotiable protocol. You could formalize this in a new XML tag within `AGENTS.md` or embed it directly in the agent's definition file.

```xml
<agent id="parallel-worker">
  <core_protocol>
    <rule>Your SOLE responsibility is delegation and orchestration.</rule>
    <rule>You MUST use the 'Task' tool to spawn sub-agents for all tasks defined in the execution plan. This ensures non-blocking, parallel execution.</rule>
    <rule>You MUST NOT use the 'bash' tool to call other agents, as this is a blocking operation.</rule>
    <rule>You MUST NOT implement any features, write code, create UI, or perform any direct development work. Delegate 100% of implementation to specialist agents.</rule>
    <rule>Your output should be a summary of the tasks you have dispatched and the agents now working on them.</rule>
  </core_protocol>
</agent>
```

**Result:** The `parallel-worker` now has an unambiguous mandate. It understands it is a delegator, not a doer, and it knows the correct, non-blocking tool to use, which is the key to achieving true parallel execution.

---

### 3. Tuning Agent Descriptions for Better Dispatching

The main Claude dispatcher uses the agent's description to make a routing decision. Vague descriptions or simple keyword lists lead to errors. The key is to describe the agent's **role, function, and outputs** while also defining its **boundaries**.

Here are four principles to apply:

1.  **Lead with a "Role Title":** Start the description with a clear job title. This immediately frames the agent's purpose.
2.  **Use Action Verbs for Primary Tasks:** Describe what the agent *does* (e.g., "Designs," "Implements," "Optimizes," "Validates").
3.  **Specify Key Outputs:** Mention what the agent *produces* (e.g., "API contracts," "React components," "Test coverage reports"). This helps the dispatcher match the user's desired outcome.
4.  **Define Exclusions (Anti-Triggers):** Explicitly state what the agent *does not* do. This is the most powerful technique for preventing incorrect dispatching.

#### Example "Before" and "After" Agent Descriptions

Let's apply these principles to a few of your agents in `AGENTS.md`.

**Example 1: `backend-architect`**

*   **Before:** `role="API/system design" coords="devops-automator,api-tester"`
*   **Problem:** Too brief. "design" is an overloaded term. A user asking to "design a logo" might incorrectly be routed here.

*   **After (Applying Principles):**
    ```xml
    <agent id="backend-architect"
           role="System Architecture &amp; API Design Specialist">
      <tasks>Designs scalable backend systems, defines data models, and creates API contracts (e.g., OpenAPI specs). Establishes architectural patterns and technology stacks.</tasks>
      <outputs>Architecture diagrams, API documentation, technology recommendations, sequence diagrams.</outputs>
      <exclusions>Does not write frontend UI code, CSS, or configure CI/CD pipelines.</exclusions>
    </agent>
    ```

**Example 2: `refactoring-specialist`**

*   **Before:** `specialization="AI-assisted code transformation and technical debt reduction"`
*   **Problem:** A user asking to "refactor this variable name" could be sent here, which is overkill. Its scope needs to be clarified.

*   **After (Applying Principles):**
    ```xml
    <agent id="refactoring-specialist"
           role="Structural Code Modernization Expert">
      <tasks>Performs large-scale, structural refactoring of codebases to improve maintainability, reduce technical debt, and migrate legacy systems. Applies automated transformation patterns safely.</tasks>
      <outputs>Modernized code with improved quality metrics, Architectural Decision Records (ADRs) for major changes, updated test suites.</outputs>
      <exclusions>Does not perform minor refactoring like variable renaming or simple function extraction (use language-specific developers for that). Does not fix bugs unless they are a direct result of the modernization effort.</exclusions>
    </agent>
    ```

By making these descriptions more robust, you provide the main Claude dispatcher with a much clearer signal, dramatically reducing the chances of it selecting an agent based on a single, random keyword.