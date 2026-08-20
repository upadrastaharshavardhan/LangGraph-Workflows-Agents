# 🚀 LangGraph Workflows & Agents
## A Practical, Detailed Guide to Workflow Patterns, Multi-Agent Systems, Tool Calling, and Production Design

> **A deep, ready-to-use reference for understanding how LangGraph workflows and agents are designed, composed, executed, and scaled.**

---

## 📚 Documentation Index

> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt  
> Use this file to discover all available pages before exploring further.

---

# 🧭 Table of Contents

- [1. Workflows and Agents](#workflows-and-agents)
- [2. How to Think About Workflow Design](#how-to-think-about-workflow-design)
- [3. Setup](#setup)
- [4. LLMs and Augmentations](#llms-and-augmentations)
- [5. Prompt Chaining](#prompt-chaining)
- [6. Parallelization](#parallelization)
- [7. Routing](#routing)
- [8. Orchestrator-Worker](#orchestrator-worker)
- [9. Evaluator-Optimizer](#evaluator-optimizer)
- [10. Agents](#agents)
- [11. ToolNode](#toolnode)
- [12. Pattern Selection Guide](#pattern-selection-guide)
- [13. Production Design Checklist](#production-design-checklist)
- [14. Common Mistakes and Anti-Patterns](#common-mistakes-and-anti-patterns)

---

# Workflows and agents

This guide reviews common workflow and agent patterns.

## 🎯 The Core Difference

A useful starting point is to separate **control flow** from **decision making**:

| Characteristic | Workflow | Agent |
|---|---|---|
| Execution path | Mostly predetermined | Dynamic |
| Next step | Defined by application logic | Often selected by the model |
| Tool usage | Explicitly designed | Model can decide among available tools |
| Predictability | High | Lower |
| Flexibility | Lower | Higher |
| Debugging | Usually simpler | Can require tracing and observability |
| Best for | Stable, repeatable processes | Open-ended or unpredictable tasks |

### Workflows

A workflow has a structure that the application designer largely knows in advance. The system may contain LLM calls, tools, validation, retries, and conditional branches, but the possible execution architecture is intentionally controlled.

### Agents

An agent gives the LLM more autonomy. The model can inspect the current situation, choose an action or tool, observe the result, and decide what to do next. This makes agents useful when the exact sequence of steps cannot be fully predicted before execution.

## 🧠 A Simple Mental Model

```text
WORKFLOW
Input
  ↓
Known Step A
  ↓
Known Step B
  ↓
Conditional Check
  ├── Pass → Finish
  └── Fail → Known Recovery Step
```

```text
AGENT
Input
  ↓
LLM decides next action
  ↓
Use Tool / Think / Respond
  ↓
Observe result
  ↓
LLM decides next action
  ↓
Repeat until completion
```

## ⚠️ Important Design Principle

**Do not use an agent simply because a workflow contains an LLM.**

An LLM-powered system can still be a deterministic workflow. If you already know the sequence of operations, explicit workflow control is often easier to test, debug, secure, and optimize.

Use more autonomy only when the problem genuinely benefits from dynamic decision making.

<img src="https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/agent_workflow.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=c217c9ef517ee556cae3fc928a21dc55" alt="Agent Workflow" width="4572" height="2047" data-path="oss/images/agent_workflow.png" />

## 🔍 What the Architecture Image Represents

The workflow/agent architecture can be understood as a sequence of **state transitions**.

Each node receives the information available at that point, performs one responsibility, and returns an update. Edges determine which node receives control next. Conditional edges introduce branching, while loops allow a system to retry, evaluate, call tools, or continue reasoning.

A strong graph design therefore starts with three questions:

1. **What information must exist in state?**
2. **What responsibility belongs to each node?**
3. **What conditions determine the next transition?**

LangGraph supports workflow and agent systems with capabilities such as persistence, streaming, debugging, and deployment.

---

# 🧠 How to Think About Workflow Design

Before choosing a pattern, define the problem in terms of **inputs, state, decisions, actions, and completion conditions**.

## 1. Define the Input

Examples:

- A user question
- A document
- A support request
- A test requirement
- A failure log
- A structured event

## 2. Define the Shared State

State is the information that must survive between steps.

```text
State
├── input
├── intermediate_results
├── routing_decision
├── tool_results
├── validation_feedback
├── retry_count
└── final_output
```

Avoid putting every possible value into state. Store information that downstream nodes genuinely need.

## 3. Define Node Responsibilities

A node should ideally have one clear responsibility:

- Generate
- Validate
- Route
- Retrieve
- Call a tool
- Aggregate
- Evaluate
- Synthesize

Avoid nodes that silently perform many unrelated responsibilities.

## 4. Define Transitions

Ask:

- What always happens next?
- What depends on a condition?
- What can run in parallel?
- What may repeat?
- What ends execution?

## 5. Define Failure Behaviour

Every production graph should consider:

- Invalid model output
- Tool failures
- Timeouts
- Empty results
- Retry limits
- Infinite loops
- Partial failures
- Human intervention

---


---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Workflows and agents

This guide reviews common workflow and agent patterns.

* Workflows have predetermined code paths and are designed to operate in a certain order.
* Agents are dynamic and define their own processes and tool usage.

<img src="https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/agent_workflow.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=c217c9ef517ee556cae3fc928a21dc55" alt="Agent Workflow" width="4572" height="2047" data-path="oss/images/agent_workflow.png" />

LangGraph offers several benefits when building agents and workflows, including [persistence](/oss/python/langgraph/persistence), [streaming](/oss/python/langgraph/streaming), and support for debugging as well as [deployment](/oss/python/langgraph/deploy).

<Tip>
  Trace and compare these workflow patterns with [LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-langgraph-workflows-agents). Follow the [tracing quickstart](/langsmith/trace-with-langgraph) to see how data flows through each step. We recommend you also set up [LangSmith Engine](/langsmith/engine) which monitors your traces, detects issues, and proposes fixes.
</Tip>

## Setup

### 🛠️ What You Need Before Building

A workflow or agent requires an LLM integration appropriate for the capabilities your graph needs. The examples below use a chat model that supports capabilities such as structured outputs and tool calling.

Before running examples, ensure you understand:

- **Model configuration** — which provider and model are being used.
- **Authentication** — how credentials are supplied.
- **Structured output support** — useful when the application needs predictable fields.
- **Tool calling support** — required when the model must request application actions.
- **Environment management** — secrets should not be hard-coded into source files.

### Recommended Execution Flow

```text
Install dependencies
        ↓
Configure credentials
        ↓
Initialize model
        ↓
Define state/schema
        ↓
Define nodes or tasks
        ↓
Connect execution paths
        ↓
Compile
        ↓
Invoke / Stream / Debug
```



To build a workflow or agent, you can use [any chat model](/oss/python/integrations/chat) that supports structured outputs and tool calling. The following example uses Anthropic:

1. Install dependencies:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install langchain_core langchain-anthropic langgraph
```

2. Initialize the LLM:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import os
import getpass

from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-6")
```

## LLMs and augmentations

### 🧩 Why an LLM Alone Is Usually Not Enough

A base LLM receives input and generates output. Agentic applications become more capable when the model is augmented with application-level capabilities.

The main augmentations in this reference are:

| Augmentation | Purpose |
|---|---|
| Structured output | Return predictable data that application code can consume |
| Tool calling | Allow the model to request external actions |
| Memory/state | Preserve relevant information across steps |
| Graph control flow | Coordinate multi-step execution |

### A Useful Architecture

```text
User Input
    ↓
LLM
    ├── Structured Output
    ├── Tool Calls
    ├── State / Memory
    └── Graph-Controlled Next Steps
```

The key design idea is that the graph remains responsible for application orchestration while the LLM contributes reasoning or generation where appropriate.



Workflows and agentic systems are based on LLMs and the various augmentations you add to them. [Tool calling](/oss/python/langchain/tools), [structured outputs](/oss/python/langchain/structured-output), and [short term memory](/oss/python/langchain/short-term-memory) are a few options for tailoring LLMs to your needs.

<img src="https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/augmented_llm.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=7ea9656f46649b3ebac19e8309ae9006" alt="LLM augmentations" width="1152" height="778" data-path="oss/images/augmented_llm.png" />

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
# Schema for structured output
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

# Get the tool call
msg.tool_calls
```

## Prompt chaining

### 🔗 What Problem Does Prompt Chaining Solve?

Prompt chaining is appropriate when a complex task can be decomposed into a sequence of smaller transformations.

Instead of asking one model call to perform everything at once:

```text
Input → One Large Prompt → Final Result
```

you use controlled stages:

```text
Input
  ↓
Generate
  ↓
Check
  ↓
Improve if needed
  ↓
Polish
  ↓
Final Result
```

### When to Use It

Use prompt chaining when:

- Steps have a natural order.
- Later stages depend on earlier outputs.
- Intermediate outputs can be checked.
- You want clear observability for each transformation.

### Design Considerations

Each stage should have:

1. A clear input.
2. A clear output.
3. A narrowly defined responsibility.
4. An explicit success or failure condition where possible.

The examples below demonstrate both Graph API and Functional API styles.



Prompt chaining is when each LLM call processes the output of the previous call. It's often used for performing well-defined tasks that can be broken down into smaller, verifiable steps. Some examples include:

* Translating documents into different languages
* Verifying generated content for consistency

<img src="https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/prompt_chain.png?fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=762dec147c31b8dc6ebb0857e236fc1f" alt="Prompt chaining" width="1412" height="444" data-path="oss/images/prompt_chain.png" />

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing_extensions import TypedDict
  from langgraph.graph import StateGraph, START, END
  from IPython.display import Image, display


  # Graph state
  class State(TypedDict):
      topic: str
      joke: str
      improved_joke: str
      final_joke: str


  # Nodes
  def generate_joke(state: State):
      """First LLM call to generate initial joke"""

      msg = llm.invoke(f"Write a short joke about {state['topic']}")
      return {"joke": msg.content}


  def check_punchline(state: State):
      """Gate function to check if the joke has a punchline"""

      # Simple check - does the joke contain "?" or "!"
      if "?" in state["joke"] or "!" in state["joke"]:
          return "Pass"
      return "Fail"


  def improve_joke(state: State):
      """Second LLM call to improve the joke"""

      msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
      return {"improved_joke": msg.content}


  def polish_joke(state: State):
      """Third LLM call for final polish"""
      msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
      return {"final_joke": msg.content}


  # Build workflow
  workflow = StateGraph(State)

  # Add nodes
  workflow.add_node("generate_joke", generate_joke)
  workflow.add_node("improve_joke", improve_joke)
  workflow.add_node("polish_joke", polish_joke)

  # Add edges to connect nodes
  workflow.add_edge(START, "generate_joke")
  workflow.add_conditional_edges(
      "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
  )
  workflow.add_edge("improve_joke", "polish_joke")
  workflow.add_edge("polish_joke", END)

  # Compile
  chain = workflow.compile()

  # Show workflow
  display(Image(chain.get_graph().draw_mermaid_png()))

  # Invoke
  state = chain.invoke({"topic": "cats"})
  print("Initial joke:")
  print(state["joke"])
  print("\n--- --- ---\n")
  if "improved_joke" in state:
      print("Improved joke:")
      print(state["improved_joke"])
      print("\n--- --- ---\n")

      print("Final joke:")
      print(state["final_joke"])
  else:
      print("Final joke:")
      print(state["joke"])
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langgraph.func import entrypoint, task


  # Tasks
  @task
  def generate_joke(topic: str):
      """First LLM call to generate initial joke"""
      msg = llm.invoke(f"Write a short joke about {topic}")
      return msg.content


  def check_punchline(joke: str):
      """Gate function to check if the joke has a punchline"""
      # Simple check - does the joke contain "?" or "!"
      if "?" in joke or "!" in joke:
          return "Fail"

      return "Pass"


  @task
  def improve_joke(joke: str):
      """Second LLM call to improve the joke"""
      msg = llm.invoke(f"Make this joke funnier by adding wordplay: {joke}")
      return msg.content


  @task
  def polish_joke(joke: str):
      """Third LLM call for final polish"""
      msg = llm.invoke(f"Add a surprising twist to this joke: {joke}")
      return msg.content


  @entrypoint()
  def prompt_chaining_workflow(topic: str):
      original_joke = generate_joke(topic).result()
      if check_punchline(original_joke) == "Pass":
          return original_joke

      improved_joke = improve_joke(original_joke).result()
      return polish_joke(improved_joke).result()

  # Invoke
  stream = prompt_chaining_workflow.stream_events("cats", version="v3")
  for snapshot in stream.values:
      print(snapshot)
      print("\n")
  ```
</CodeGroup>

## Parallelization

### ⚡ What Problem Does Parallelization Solve?

Parallelization reduces end-to-end latency when multiple tasks are independent.

Instead of:

```text
Task A → Task B → Task C
```

you can use a fan-out/fan-in design:

```text
             ┌→ Task A ─┐
Input ───────┼→ Task B ─┼→ Aggregate → Output
             └→ Task C ─┘
```

### Two Common Forms

#### 1. Independent Subtasks

Different workers process different parts of the problem simultaneously.

#### 2. Multiple Evaluations

The same output is evaluated from different perspectives simultaneously.

### Important Constraint

Parallel execution is only useful when branches do not depend on each other's unfinished results. If Task B requires Task A's output, the operations are sequential.



With parallelization, LLMs work simultaneously on a task. This is either done by running multiple independent subtasks at the same time, or running the same task multiple times to check for different outputs. Parallelization is commonly used to:

* Split up subtasks and run them in parallel, which increases speed
* Run tasks multiple times to check for different outputs, which increases confidence

Some examples include:

* Running one subtask that processes a document for keywords, and a second subtask to check for formatting errors
* Running a task multiple times that scores a document for accuracy based on different criteria, like the number of citations, the number of sources used, and the quality of the sources

<img src="https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/parallelization.png?fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=8afe3c427d8cede6fed1e4b2a5107b71" alt="parallelization.png" width="1020" height="684" data-path="oss/images/parallelization.png" />

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # Graph state
  class State(TypedDict):
      topic: str
      joke: str
      story: str
      poem: str
      combined_output: str


  # Nodes
  def call_llm_1(state: State):
      """First LLM call to generate initial joke"""

      msg = llm.invoke(f"Write a joke about {state['topic']}")
      return {"joke": msg.content}


  def call_llm_2(state: State):
      """Second LLM call to generate story"""

      msg = llm.invoke(f"Write a story about {state['topic']}")
      return {"story": msg.content}


  def call_llm_3(state: State):
      """Third LLM call to generate poem"""

      msg = llm.invoke(f"Write a poem about {state['topic']}")
      return {"poem": msg.content}


  def aggregator(state: State):
      """Combine the joke, story and poem into a single output"""

      combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
      combined += f"STORY:\n{state['story']}\n\n"
      combined += f"JOKE:\n{state['joke']}\n\n"
      combined += f"POEM:\n{state['poem']}"
      return {"combined_output": combined}


  # Build workflow
  parallel_builder = StateGraph(State)

  # Add nodes
  parallel_builder.add_node("call_llm_1", call_llm_1)
  parallel_builder.add_node("call_llm_2", call_llm_2)
  parallel_builder.add_node("call_llm_3", call_llm_3)
  parallel_builder.add_node("aggregator", aggregator)

  # Add edges to connect nodes
  parallel_builder.add_edge(START, "call_llm_1")
  parallel_builder.add_edge(START, "call_llm_2")
  parallel_builder.add_edge(START, "call_llm_3")
  parallel_builder.add_edge("call_llm_1", "aggregator")
  parallel_builder.add_edge("call_llm_2", "aggregator")
  parallel_builder.add_edge("call_llm_3", "aggregator")
  parallel_builder.add_edge("aggregator", END)
  parallel_workflow = parallel_builder.compile()

  # Show workflow
  display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

  # Invoke
  state = parallel_workflow.invoke({"topic": "cats"})
  print(state["combined_output"])
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  @task
  def call_llm_1(topic: str):
      """First LLM call to generate initial joke"""
      msg = llm.invoke(f"Write a joke about {topic}")
      return msg.content


  @task
  def call_llm_2(topic: str):
      """Second LLM call to generate story"""
      msg = llm.invoke(f"Write a story about {topic}")
      return msg.content


  @task
  def call_llm_3(topic):
      """Third LLM call to generate poem"""
      msg = llm.invoke(f"Write a poem about {topic}")
      return msg.content


  @task
  def aggregator(topic, joke, story, poem):
      """Combine the joke and story into a single output"""

      combined = f"Here's a story, joke, and poem about {topic}!\n\n"
      combined += f"STORY:\n{story}\n\n"
      combined += f"JOKE:\n{joke}\n\n"
      combined += f"POEM:\n{poem}"
      return combined


  # Build workflow
  @entrypoint()
  def parallel_workflow(topic: str):
      joke_fut = call_llm_1(topic)
      story_fut = call_llm_2(topic)
      poem_fut = call_llm_3(topic)
      return aggregator(
          topic, joke_fut.result(), story_fut.result(), poem_fut.result()
      ).result()

  # Invoke
  stream = parallel_workflow.stream_events("cats", version="v3")
  for snapshot in stream.values:
      print(snapshot)
      print("\n")
  ```
</CodeGroup>

## Routing

### 🔀 What Problem Does Routing Solve?

Routing chooses a specialized execution path based on the input or current state.

```text
Input
  ↓
Classify / Decide
  ├── Route A → Specialist A
  ├── Route B → Specialist B
  └── Route C → Specialist C
```

### Good Routing Design

A router should return a constrained decision rather than free-form prose whenever possible. Structured output is especially useful because the graph can safely map a known route value to a known node.

### Production Considerations

Define behaviour for:

- Unknown categories
- Low-confidence decisions
- Invalid structured output
- Missing routes
- Fallback handling



Routing workflows process inputs and then directs them to context-specific tasks. This allows you to define specialized flows for complex tasks. For example, a workflow built to answer product related questions might process the type of question first, and then route the request to specific processes for pricing, refunds, returns, etc.

<img src="https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/routing.png?fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=272e0e9b681b89cd7d35d5c812c50ee6" alt="routing.png" width="1214" height="678" data-path="oss/images/routing.png" />

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing_extensions import Literal
  from langchain.messages import HumanMessage, SystemMessage


  # Schema for structured output to use as routing logic
  class Route(BaseModel):
      step: Literal["poem", "story", "joke"] = Field(
          None, description="The next step in the routing process"
      )


  # Augment the LLM with schema for structured output
  router = llm.with_structured_output(Route)


  # State
  class State(TypedDict):
      input: str
      decision: str
      output: str


  # Nodes
  def llm_call_1(state: State):
      """Write a story"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_2(state: State):
      """Write a joke"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_3(state: State):
      """Write a poem"""

      result = llm.invoke(state["input"])
      return {"output": result.content}


  def llm_call_router(state: State):
      """Route the input to the appropriate node"""

      # Run the augmented LLM with structured output to serve as routing logic
      decision = router.invoke(
          [
              SystemMessage(
                  content="Route the input to story, joke, or poem based on the user's request."
              ),
              HumanMessage(content=state["input"]),
          ]
      )

      return {"decision": decision.step}


  # Conditional edge function to route to the appropriate node
  def route_decision(state: State):
      # Return the node name you want to visit next
      if state["decision"] == "story":
          return "llm_call_1"
      elif state["decision"] == "joke":
          return "llm_call_2"
      elif state["decision"] == "poem":
          return "llm_call_3"


  # Build workflow
  router_builder = StateGraph(State)

  # Add nodes
  router_builder.add_node("llm_call_1", llm_call_1)
  router_builder.add_node("llm_call_2", llm_call_2)
  router_builder.add_node("llm_call_3", llm_call_3)
  router_builder.add_node("llm_call_router", llm_call_router)

  # Add edges to connect nodes
  router_builder.add_edge(START, "llm_call_router")
  router_builder.add_conditional_edges(
      "llm_call_router",
      route_decision,
      {  # Name returned by route_decision : Name of next node to visit
          "llm_call_1": "llm_call_1",
          "llm_call_2": "llm_call_2",
          "llm_call_3": "llm_call_3",
      },
  )
  router_builder.add_edge("llm_call_1", END)
  router_builder.add_edge("llm_call_2", END)
  router_builder.add_edge("llm_call_3", END)

  # Compile workflow
  router_workflow = router_builder.compile()

  # Show the workflow
  display(Image(router_workflow.get_graph().draw_mermaid_png()))

  # Invoke
  state = router_workflow.invoke({"input": "Write me a joke about cats"})
  print(state["output"])
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing_extensions import Literal
  from pydantic import BaseModel
  from langchain.messages import HumanMessage, SystemMessage


  # Schema for structured output to use as routing logic
  class Route(BaseModel):
      step: Literal["poem", "story", "joke"] = Field(
          None, description="The next step in the routing process"
      )


  # Augment the LLM with schema for structured output
  router = llm.with_structured_output(Route)


  @task
  def llm_call_1(input_: str):
      """Write a story"""
      result = llm.invoke(input_)
      return result.content


  @task
  def llm_call_2(input_: str):
      """Write a joke"""
      result = llm.invoke(input_)
      return result.content


  @task
  def llm_call_3(input_: str):
      """Write a poem"""
      result = llm.invoke(input_)
      return result.content


  def llm_call_router(input_: str):
      """Route the input to the appropriate node"""
      # Run the augmented LLM with structured output to serve as routing logic
      decision = router.invoke(
          [
              SystemMessage(
                  content="Route the input to story, joke, or poem based on the user's request."
              ),
              HumanMessage(content=input_),
          ]
      )
      return decision.step


  # Create workflow
  @entrypoint()
  def router_workflow(input_: str):
      next_step = llm_call_router(input_)
      if next_step == "story":
          llm_call = llm_call_1
      elif next_step == "joke":
          llm_call = llm_call_2
      elif next_step == "poem":
          llm_call = llm_call_3

      return llm_call(input_).result()

  # Invoke
  stream = router_workflow.stream_events("Write me a joke about cats", version="v3")
  for snapshot in stream.values:
      print(snapshot)
      print("\n")
  ```
</CodeGroup>

## Orchestrator-worker

### 👷 What Problem Does This Pattern Solve?

Use orchestrator-worker when the number or type of subtasks cannot be fully defined before execution.

The orchestrator dynamically:

1. Understands the overall request.
2. Breaks it into subtasks.
3. Delegates work.
4. Collects completed outputs.
5. Synthesizes a final result.

### Architecture

```text
Complex Request
      ↓
Orchestrator / Planner
      ↓
Creates Dynamic Task List
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
W1    W2   W3
 ↓    ↓    ↓
 └────┼────┘
      ↓
Synthesizer
      ↓
Final Output
```

This differs from simple parallelization because the work plan itself may be generated dynamically.



In an orchestrator-worker configuration, the orchestrator:

* Breaks down tasks into subtasks
* Delegates subtasks to workers
* Synthesizes worker outputs into a final result

<img src="https://mintcdn.com/langchain-5e9cc07a/ybiAaBfoBvFquMDz/oss/images/worker.png?fit=max&auto=format&n=ybiAaBfoBvFquMDz&q=85&s=2e423c67cd4f12e049cea9c169ff0676" alt="worker.png" width="1486" height="548" data-path="oss/images/worker.png" />

Orchestrator-worker workflows provide more flexibility and are often used when subtasks cannot be predefined the way they can with [parallelization](#parallelization). This is common with workflows that write code or need to update content across multiple files. For example, a workflow that needs to update installation instructions for multiple Python libraries across an unknown number of documents might use this pattern.

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing import Annotated, List
  import operator


  # Schema for structured output to use in planning
  class Section(BaseModel):
      name: str = Field(
          description="Name for this section of the report.",
      )
      description: str = Field(
          description="Brief overview of the main topics and concepts to be covered in this section.",
      )


  class Sections(BaseModel):
      sections: List[Section] = Field(
          description="Sections of the report.",
      )


  # Augment the LLM with schema for structured output
  planner = llm.with_structured_output(Sections)
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing import List


  # Schema for structured output to use in planning
  class Section(BaseModel):
      name: str = Field(
          description="Name for this section of the report.",
      )
      description: str = Field(
          description="Brief overview of the main topics and concepts to be covered in this section.",
      )


  class Sections(BaseModel):
      sections: List[Section] = Field(
          description="Sections of the report.",
      )


  # Augment the LLM with schema for structured output
  planner = llm.with_structured_output(Sections)


  @task
  def orchestrator(topic: str):
      """Orchestrator that generates a plan for the report"""
      # Generate queries
      report_sections = planner.invoke(
          [
              SystemMessage(content="Generate a plan for the report."),
              HumanMessage(content=f"Here is the report topic: {topic}"),
          ]
      )

      return report_sections.sections


  @task
  def llm_call(section: Section):
      """Worker writes a section of the report"""

      # Generate section
      result = llm.invoke(
          [
              SystemMessage(content="Write a report section."),
              HumanMessage(
                  content=f"Here is the section name: {section.name} and description: {section.description}"
              ),
          ]
      )

      # Write the updated section to completed sections
      return result.content


  @task
  def synthesizer(completed_sections: list[str]):
      """Synthesize full report from sections"""
      final_report = "\n\n---\n\n".join(completed_sections)
      return final_report


  @entrypoint()
  def orchestrator_worker(topic: str):
      sections = orchestrator(topic).result()
      section_futures = [llm_call(section) for section in sections]
      final_report = synthesizer(
          [section_fut.result() for section_fut in section_futures]
      ).result()
      return final_report

  # Invoke
  report = orchestrator_worker.invoke("Create a report on LLM scaling laws")
  from IPython.display import Markdown
  Markdown(report)
  ```
</CodeGroup>

### Creating workers in LangGraph

Orchestrator-worker workflows are common and LangGraph has built-in support for them. The `Send` API lets you dynamically create worker nodes and send them specific inputs. Each worker has its own state, and all worker outputs are written to a shared state key that is accessible to the orchestrator graph. This gives the orchestrator access to all worker output and allows it to synthesize them into a final output. The example below iterates over a list of sections and uses the `Send` API to send a section to each worker.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langgraph.types import Send


# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    # Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Show the workflow
display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

# Invoke
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})

from IPython.display import Markdown
Markdown(state["final_report"])
```

## Evaluator-optimizer

### 🔄 What Problem Does This Pattern Solve?

Some outputs cannot be reliably completed in one generation. The evaluator-optimizer pattern introduces an explicit quality loop:

```text
Generate
   ↓
Evaluate
   ↓
Accept? ── Yes → Finish
   │
   No
   ↓
Feedback
   ↓
Regenerate
   └───────────────↺
```

### When It Is Useful

Use this pattern when:

- Success criteria can be defined.
- Iterative refinement improves quality.
- Feedback can guide another generation.
- A bounded number of retries is acceptable.

### Critical Production Rule

Always define a maximum iteration count or another stopping condition. An unbounded evaluator loop can increase cost and create non-terminating executions.



In evaluator-optimizer workflows, one LLM call creates a response and the other evaluates that response. If the evaluator or a [human-in-the-loop](/oss/python/langgraph/interrupts) determines the response needs refinement, feedback is provided and the response is recreated. This loop continues until an acceptable response is generated.

Evaluator-optimizer workflows are commonly used when there's particular success criteria for a task, but iteration is required to meet that criteria. For example, there's not always a perfect match when translating text between two languages. It might take a few iterations to generate a translation with the same meaning across the two languages.

<img src="https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/evaluator_optimizer.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=9bd0474f42b6040b14ed6968a9ab4e3c" alt="evaluator_optimizer.png" width="1004" height="340" data-path="oss/images/evaluator_optimizer.png" />

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # Graph state
  class State(TypedDict):
      joke: str
      topic: str
      feedback: str
      funny_or_not: str


  # Schema for structured output to use in evaluation
  class Feedback(BaseModel):
      grade: Literal["funny", "not funny"] = Field(
          description="Decide if the joke is funny or not.",
      )
      feedback: str = Field(
          description="If the joke is not funny, provide feedback on how to improve it.",
      )


  # Augment the LLM with schema for structured output
  evaluator = llm.with_structured_output(Feedback)


  # Nodes
  def llm_call_generator(state: State):
      """LLM generates a joke"""

      if state.get("feedback"):
          msg = llm.invoke(
              f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
          )
      else:
          msg = llm.invoke(f"Write a joke about {state['topic']}")
      return {"joke": msg.content}


  def llm_call_evaluator(state: State):
      """LLM evaluates the joke"""

      grade = evaluator.invoke(f"Grade the joke {state['joke']}")
      return {"funny_or_not": grade.grade, "feedback": grade.feedback}


  # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
  def route_joke(state: State):
      """Route back to joke generator or end based upon feedback from the evaluator"""

      if state["funny_or_not"] == "funny":
          return "Accepted"
      elif state["funny_or_not"] == "not funny":
          return "Rejected + Feedback"


  # Build workflow
  optimizer_builder = StateGraph(State)

  # Add the nodes
  optimizer_builder.add_node("llm_call_generator", llm_call_generator)
  optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

  # Add edges to connect nodes
  optimizer_builder.add_edge(START, "llm_call_generator")
  optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
  optimizer_builder.add_conditional_edges(
      "llm_call_evaluator",
      route_joke,
      {  # Name returned by route_joke : Name of next node to visit
          "Accepted": END,
          "Rejected + Feedback": "llm_call_generator",
      },
  )

  # Compile the workflow
  optimizer_workflow = optimizer_builder.compile()

  # Show the workflow
  display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

  # Invoke
  state = optimizer_workflow.invoke({"topic": "Cats"})
  print(state["joke"])
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # Schema for structured output to use in evaluation
  class Feedback(BaseModel):
      grade: Literal["funny", "not funny"] = Field(
          description="Decide if the joke is funny or not.",
      )
      feedback: str = Field(
          description="If the joke is not funny, provide feedback on how to improve it.",
      )


  # Augment the LLM with schema for structured output
  evaluator = llm.with_structured_output(Feedback)


  # Nodes
  @task
  def llm_call_generator(topic: str, feedback: Feedback):
      """LLM generates a joke"""
      if feedback:
          msg = llm.invoke(
              f"Write a joke about {topic} but take into account the feedback: {feedback}"
          )
      else:
          msg = llm.invoke(f"Write a joke about {topic}")
      return msg.content


  @task
  def llm_call_evaluator(joke: str):
      """LLM evaluates the joke"""
      feedback = evaluator.invoke(f"Grade the joke {joke}")
      return feedback


  @entrypoint()
  def optimizer_workflow(topic: str):
      feedback = None
      while True:
          joke = llm_call_generator(topic, feedback).result()
          feedback = llm_call_evaluator(joke).result()
          if feedback.grade == "funny":
              break

      return joke

  # Invoke
  stream = optimizer_workflow.stream_events("Cats", version="v3")
  for snapshot in stream.values:
      print(snapshot)
      print("\n")
  ```
</CodeGroup>

## Agents

### 🤖 What Makes an Agent Different?

An agent repeatedly observes the current situation and determines the next action.

A typical loop is:

```text
User Request
      ↓
LLM
      ↓
Need a tool?
 ┌────┴────┐
 No        Yes
 ↓          ↓
Answer     Execute Tool
              ↓
          Tool Result
              ↓
             LLM
              ↺
```

### Agent Design Questions

Before creating an agent, decide:

- Which tools are available?
- Which actions are forbidden?
- What information is included in state?
- What stops the loop?
- How are tool errors represented?
- Can multiple tools run concurrently?
- Is human approval required for sensitive actions?

The following examples show the core tool execution loop explicitly.



Agents are typically implemented as an LLM performing actions using [tools](/oss/python/langchain/tools). They operate in continuous feedback loops, and are used in situations where problems and solutions are unpredictable. Agents have more autonomy than workflows, and can make decisions about the tools they use and how to solve problems. You can still define the available toolset and guidelines for how agents behave.

<img src="https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/agent.png?fit=max&auto=format&n=-_xGPoyjhyiDWTPJ&q=85&s=bd8da41dbf8b5e6fc9ea6bb10cb63e38" alt="agent.png" width="1732" height="712" data-path="oss/images/agent.png" />

<Note>
  To get started with agents, see the [quickstart](/oss/python/langchain/quickstart) or read more about [how they work](/oss/python/langchain/agents) in LangChain.
</Note>

```python Using tools theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.tools import tool


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```

<CodeGroup>
  ```python Graph API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langgraph.graph import MessagesState
  from langchain.messages import SystemMessage, HumanMessage, ToolMessage


  # Nodes
  def llm_call(state: MessagesState):
      """LLM decides whether to call a tool or not"""

      return {
          "messages": [
              llm_with_tools.invoke(
                  [
                      SystemMessage(
                          content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                      )
                  ]
                  + state["messages"]
              )
          ]
      }


  def tool_node(state: MessagesState):
      """Performs the tool call"""

      result = []
      for tool_call in state["messages"][-1].tool_calls:
          tool = tools_by_name[tool_call["name"]]
          observation = tool.invoke(tool_call["args"])
          result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
      return {"messages": result}


  # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
  def should_continue(state: MessagesState) -> Literal["tool_node", END]:
      """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

      messages = state["messages"]
      last_message = messages[-1]

      # If the LLM makes a tool call, then perform an action
      if last_message.tool_calls:
          return "tool_node"

      # Otherwise, we stop (reply to the user)
      return END


  # Build workflow
  agent_builder = StateGraph(MessagesState)

  # Add nodes
  agent_builder.add_node("llm_call", llm_call)
  agent_builder.add_node("tool_node", tool_node)

  # Add edges to connect nodes
  agent_builder.add_edge(START, "llm_call")
  agent_builder.add_conditional_edges(
      "llm_call",
      should_continue,
      ["tool_node", END]
  )
  agent_builder.add_edge("tool_node", "llm_call")

  # Compile the agent
  agent = agent_builder.compile()

  # Show the agent
  display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

  # Invoke
  messages = [HumanMessage(content="Add 3 and 4.")]
  messages = agent.invoke({"messages": messages})
  for m in messages["messages"]:
      m.pretty_print()
  ```

  ```python Functional API theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langgraph.graph import add_messages
  from langchain.messages import (
      SystemMessage,
      HumanMessage,
      ToolCall,
  )
  from langchain_core.messages import BaseMessage


  @task
  def call_llm(messages: list[BaseMessage]):
      """LLM decides whether to call a tool or not"""
      return llm_with_tools.invoke(
          [
              SystemMessage(
                  content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
              )
          ]
          + messages
      )


  @task
  def call_tool(tool_call: ToolCall):
      """Performs the tool call"""
      tool = tools_by_name[tool_call["name"]]
      return tool.invoke(tool_call)


  @entrypoint()
  def agent(messages: list[BaseMessage]):
      llm_response = call_llm(messages).result()

      while True:
          if not llm_response.tool_calls:
              break

          # Execute tools
          tool_result_futures = [
              call_tool(tool_call) for tool_call in llm_response.tool_calls
          ]
          tool_results = [fut.result() for fut in tool_result_futures]
          messages = add_messages(messages, [llm_response, *tool_results])
          llm_response = call_llm(messages).result()

      messages = add_messages(messages, llm_response)
      return messages

  # Invoke
  messages = [HumanMessage(content="Add 3 and 4.")]
  stream = agent.stream_events(messages, version="v3")
  for snapshot in stream.values:
      print(snapshot)
      print("\n")
  ```
</CodeGroup>

### ToolNode

### 🧰 Why Use a Prebuilt Tool Node?

Tool execution contains repeated infrastructure concerns:

- Executing requested tools
- Handling multiple tool calls
- Managing tool errors
- Connecting tool results back into graph state
- Injecting graph-side state or run context

A prebuilt tool execution node reduces the amount of custom orchestration code required for these responsibilities.

### Conceptual Flow

```text
LLM Response
    ↓
Contains Tool Calls?
    ├── No → Finish / Continue Normal Flow
    └── Yes
          ↓
       ToolNode
          ↓
   Execute Requested Tools
          ↓
     Tool Messages
          ↓
          LLM
```



[`ToolNode`](https://reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.ToolNode) is a prebuilt node that executes tools in LangGraph workflows. It handles parallel tool execution, error handling, and state injection automatically.

Use [`ToolNode`](https://reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.ToolNode) when you need fine-grained control over how your graph executes tools. This is the building block that powers tool execution in many LangGraph agent patterns.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode([search, calculator]))
# ... add other nodes and edges
graph = builder.compile()
```

#### Access graph state and context from tools

Tools executed by `ToolNode` receive the arguments generated by the model as
their first argument. To read graph-side data that was not generated by the
model, use one of these options:

* In Python, read state and run-scoped context from the injected
  [`ToolRuntime`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime) argument.
* In JavaScript, read state and run-scoped context from the tool's second
  argument, typed as [`ToolRuntime`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime).

<Note>
  Tools can only access the state values passed to the `ToolNode`. When
  `ToolNode` is added directly as a `StateGraph` node, that input is the current
  graph state. If you invoke a `ToolNode` manually from another node, pass the
  full state when tools need custom state fields. For example, `tool_node.invoke(state)`
  or `toolNode.invoke(state, config)` exposes the full state, while passing only
  `{"messages": state["messages"]}` or `{ messages: state.messages }` only exposes
  `messages`.
</Note>

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode


class State(MessagesState):
    user_id: str


@dataclass
class Context:
    organization_id: str


@tool
def get_user_info(runtime: ToolRuntime[Context, State]) -> str:
    """Look up user information."""
    # Read the current graph state passed to the ToolNode.
    user_id = runtime.state["user_id"]

    # Read explicit per-run values that are not part of graph state.
    organization_id = runtime.context.organization_id

    return f"User {user_id} in organization {organization_id}"


builder = StateGraph(State, context_schema=Context)
builder.add_node("tools", ToolNode([get_user_info]))
builder.add_edge(START, "tools")
graph = builder.compile()

result = graph.invoke(
    {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_user_info",
                        "args": {},
                        "id": "call_user_info",
                    }
                ],
            )
        ],
        "user_id": "user_123",
    },
    context=Context(organization_id="org_456"),
)
```

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langgraph/workflows-agents.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>

````

**LF**


---

# 🎯 Pattern Selection Guide

## Which Pattern Should You Use?

| Situation | Recommended Pattern | Why |
|---|---|---|
| Fixed sequence of transformations | Prompt chaining | Order is known |
| Independent tasks | Parallelization | Reduces latency |
| Input requires specialist handling | Routing | Selects context-specific path |
| Subtasks must be created dynamically | Orchestrator-worker | Planner can generate work |
| Output must meet quality criteria | Evaluator-optimizer | Supports iterative refinement |
| Next actions are unpredictable | Agent | Model can choose actions |
| LLM needs external capabilities | Agent + tools / ToolNode | Supports action execution |

## Quick Decision Tree

```text
Do you know the execution path in advance?
│
├── Yes
│   │
│   ├── Do tasks depend on previous outputs?
│   │   ├── Yes → Prompt Chaining
│   │   └── No  → Parallelization
│   │
│   └── Do different inputs need different specialists?
│       └── Yes → Routing
│
└── No
    │
    ├── Can a planner break work into dynamic subtasks?
    │   └── Yes → Orchestrator-Worker
    │
    └── Must the model choose actions/tools dynamically?
        └── Yes → Agent
```

---

# 🏭 Production Design Checklist

Before moving a graph from experimentation toward production, review the following areas.

## State

- [ ] Is state minimal and clearly typed?
- [ ] Does each node receive the information it actually needs?
- [ ] Are concurrent updates handled safely?
- [ ] Are shared collections using appropriate reducers or aggregation logic?

## Control Flow

- [ ] Does every conditional branch have a valid destination?
- [ ] Are fallback paths defined?
- [ ] Are retry loops bounded?
- [ ] Can the graph terminate from every valid execution path?

## Model Output

- [ ] Are structured schemas used where application logic depends on model output?
- [ ] Is invalid output handled?
- [ ] Are prompts narrow enough for the responsibility of each node?

## Tools

- [ ] Are tool permissions limited?
- [ ] Are tool arguments validated?
- [ ] Are tool failures represented clearly?
- [ ] Are dangerous actions protected by approval mechanisms?

## Reliability

- [ ] Are transient failures retried appropriately?
- [ ] Are timeouts considered?
- [ ] Is partial progress recoverable?
- [ ] Is persistence/checkpointing required?

## Observability

- [ ] Can you identify which node failed?
- [ ] Can you inspect model and tool inputs/outputs where appropriate?
- [ ] Can you measure latency and cost?
- [ ] Can you compare different workflow versions?

---

# ⚠️ Common Mistakes and Anti-Patterns

## 1. Using an Agent for a Fully Deterministic Process

If the path is already known, unnecessary autonomy can add latency, cost, and debugging complexity.

## 2. Putting Everything in One Giant Node

Large nodes are difficult to test and observe. Separate responsibilities into meaningful units.

## 3. Returning Free-Form Text for Critical Routing

If code depends on a route decision, use constrained structured output where possible.

## 4. Creating Unbounded Loops

Evaluator, retry, and agent loops need explicit stopping conditions.

## 5. Parallelizing Dependent Work

Only fan out tasks that can execute independently.

## 6. Giving Agents Too Many Tools

More tools do not automatically produce better agents. A focused toolset makes decisions easier to reason about and secure.

## 7. Ignoring Failure Paths

A successful demo path is not a complete production design. Explicitly model tool errors, invalid outputs, timeouts, and recovery.

---

# 🧪 A Practical QA Automation Interpretation

These workflow patterns can be mapped to AI-assisted quality engineering.

```text
Requirement
    ↓
Routing
    ├── UI Test Generation
    ├── API Test Generation
    └── Data Validation
            ↓
      Parallel Execution
            ↓
      Result Aggregation
            ↓
      Evaluator
       ├── Accept → Report
       └── Improve → Regenerate / Refine
```

A more autonomous architecture can use an agent when the next diagnostic action is unknown:

```text
Test Failure
     ↓
RCA Agent
     ↓
Choose Action
 ┌────┼─────────┐
 ↓    ↓         ↓
Logs  Retry   Inspect Locator
 ↓    ↓         ↓
 └────┼─────────┘
      ↓
Evaluate Evidence
      ↓
Final Diagnosis
```

The pattern should be selected based on the predictability of the task—not simply on whether an LLM is available.

---

# 📌 Final Takeaway

The central design decision is simple:

> **Use explicit workflows when you can predict the process. Use agentic autonomy when you cannot predict the process and dynamic decisions provide real value.**

Prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer, and agents are complementary patterns. A production system can combine them:

```text
Router
  ↓
Orchestrator
  ↓
Parallel Workers
  ↓
Evaluator
  ↓
Agentic Recovery
  ↓
Final Result
```

Choose the smallest amount of autonomy that solves the problem effectively. Start with clear state, focused nodes, explicit transitions, bounded loops, and strong observability. Add dynamic agent behaviour only where it genuinely improves the system.
