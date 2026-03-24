# Knowledge Graph Query & Navigation Features

## Overview

The `KnowledgeGraph` class now includes powerful query and navigation capabilities for analyzing the mobile app exploration graph. These features enable sophisticated traversal algorithms, cycle detection, and reachability analysis.

## Features Implemented

### 1. **Shortest Path Finding** (`find_path`)

Finds the shortest path between two screens using Breadth-First Search (BFS).

```python
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
await kg.load()

# Find shortest path from screen A to screen D
path = kg.find_path("hash_screen_a", "hash_screen_d", max_depth=50)

if path:
    for node_hash, edge in path:
        if edge:
            print(f"→ {edge.action_type} '{edge.action_target}' (count: {edge.count})")
        else:
            print(f"Start: {node_hash}")
else:
    print("No path found")
```

**Parameters:**
- `start_hash`: Visual hash of the starting screen
- `end_hash`: Visual hash of the target screen
- `max_depth`: Maximum search depth (default: 50) to prevent infinite exploration

**Returns:**
- List of `(node_hash, edge_taken)` tuples representing the path
- `None` if no path exists

**Time Complexity:** O(V + E) where V is nodes, E is edges

---

### 2. **All Paths Discovery** (`find_all_paths`)

Finds all possible paths between two screens up to a depth limit. Useful for understanding all possible user journeys.

```python
# Find all ways to get from login screen to purchase screen
all_paths = kg.find_all_paths(
    "hash_login",
    "hash_purchase",
    max_depth=10
)

print(f"Found {len(all_paths)} different paths")
for i, path in enumerate(all_paths, 1):
    print(f"\nPath {i} ({len(path)} steps):")
    for node, edge in path:
        if edge:
            print(f"  {node} --[{edge.action_type}]--> ...")
```

**Parameters:**
- `start_hash`: Starting screen hash
- `end_hash`: Target screen hash
- `max_depth`: Maximum search depth (default: 10, smaller than `find_path` to prevent explosion)

**Returns:**
- List of paths, where each path is a list of `(node_hash, edge_taken)` tuples

**Note:** Using a smaller `max_depth` is recommended here since the number of paths can grow exponentially.

---

### 3. **Reachability Analysis**

#### `is_reachable(start_hash, end_hash, max_depth=100)`

Quick boolean check if one screen is reachable from another.

```python
if kg.is_reachable("hash_home", "hash_checkout"):
    print("Checkout is accessible from home screen")
else:
    print("Checkout is unreachable from home screen")
```

#### `get_connected_component(start_hash)`

Gets all screens reachable from a given starting screen (forward reachability).

```python
reachable = kg.get_connected_component("hash_home")
print(f"From home screen, can reach {len(reachable)} screens")
for screen_hash in reachable:
    node = kg.get_screen(screen_hash)
    print(f"  - {node.description}")
```

**Returns:** Set of all reachable node hashes (including the start node)

#### `get_reverse_connected_component(end_hash)`

Gets all screens that can reach a given target screen (backward reachability).

```python
sources = kg.get_reverse_connected_component("hash_payment")
print(f"{len(sources)} screens can reach the payment screen:")
for screen_hash in sources:
    node = kg.get_screen(screen_hash)
    print(f"  - {node.description}")
```

**Returns:** Set of all nodes that can reach the target (including the target itself)

---

### 4. **Cycle Detection** (`detect_cycles`)

Detects all cycles in the graph using Depth-First Search (DFS).

```python
# Find all cycles in the entire graph
cycles = kg.detect_cycles()

print(f"Found {len(cycles)} cycles:")
for i, cycle in enumerate(cycles, 1):
    print(f"\nCycle {i}:")
    for node_hash in cycle[:-1]:  # Exclude repeated end node
        node = kg.get_screen(node_hash)
        print(f"  - {node.description}")
    print(f"  → (back to {kg.get_screen(cycle[0]).description})")

# Find cycles reachable from a specific screen
cycles_from_home = kg.detect_cycles(start_hash="hash_home")
```

**Parameters:**
- `start_hash`: Optional starting point. If `None`, searches entire graph.

**Returns:** List of cycles, where each cycle is a list of node hashes (first and last are the same)

**Use Cases:**
- Detecting infinite loops in user flows
- Identifying screens users can get stuck in
- Understanding navigation patterns that create cycles

---

### 5. **Graph Diameter** (`get_graph_diameter`)

Computes the longest shortest path between any two connected nodes.

```python
diameter = kg.get_graph_diameter()
if diameter:
    print(f"Maximum steps needed to reach any screen: {diameter}")
else:
    print("Graph is empty or completely disconnected")
```

**Returns:** Integer diameter, or `None` if graph is empty

**Note:** For large graphs, this is computationally expensive as it runs shortest path between all node pairs. Consider caching.

---

### 6. **Visualization Context** (`get_visualization_context`)

Generates rich context about a screen's position in the graph for visualization and analysis.

```python
context = kg.get_visualization_context("hash_cart", depth=2)

print(f"Screen: {context['node']['description']}")
print(f"Visited {context['node']['visit_count']} times")
print(f"Can reach {context['forward_reachable']} screens")
print(f"Reachable from {context['backward_reachable']} screens")
print(f"Part of a cycle: {context['in_cycle']}")

print("\nOutgoing actions:")
for edge_info in context['outgoing_edges']:
    print(f"  {edge_info['action_type']} → {edge_info['destination_description']}")

print("\nInbound edges:")
for edge_info in context['inbound_edges']:
    print(f"  from {edge_info['source_description']} via {edge_info['action_type']}")
```

**Parameters:**
- `visual_hash`: Screen to analyze
- `depth`: Levels of neighboring nodes to include (default: 2)

**Returns:** Dictionary with:
- `node`: Screen metadata (hash, activity, description, visit count, timestamps)
- `outgoing_edges`: List of actions leaving this screen
- `inbound_edges`: List of transitions entering this screen
- `forward_reachable`: Count of reachable screens
- `backward_reachable`: Count of screens that can reach this one
- `in_cycle`: Boolean indicating if part of any cycle

---

## Practical Examples

### Finding the Shortest User Journey

```python
# What's the quickest way from app launch to making a purchase?
path = kg.find_path("hash_splash", "hash_purchase", max_depth=20)

if path:
    steps = []
    for node_hash, edge in path:
        node = kg.get_screen(node_hash)
        if edge:
            steps.append(f"{edge.action_type} '{edge.action_target}'")

    print(f"Shortest purchase flow: {len(steps)} steps")
    print(" → ".join(steps))
```

### Detecting Navigation Loops

```python
# Find screens that can create loops
for cycle in kg.detect_cycles():
    if len(cycle) > 2:  # Multi-node cycle
        print(f"⚠️  Potential infinite loop with {len(cycle)-1} screens:")
        for hash_val in cycle[:-1]:
            print(f"   {kg.get_screen(hash_val).description}")
```

### Coverage Analysis

```python
# What percentage of screens are reachable from home?
from_home = kg.get_connected_component("hash_home")
total_screens = kg.node_count
coverage = (len(from_home) / total_screens) * 100
print(f"Coverage from home screen: {coverage:.1f}%")

# Which screens are orphaned?
for node in kg.nodes.values():
    if node.visual_hash not in from_home:
        print(f"Unreachable: {node.description}")
```

### Testing Navigation Robustness

```python
# Verify all critical screens are reachable from home
critical_screens = [
    "hash_checkout",
    "hash_payment",
    "hash_confirmation"
]

for screen_hash in critical_screens:
    if kg.is_reachable("hash_home", screen_hash):
        print(f"✓ {screen_hash} is reachable")
    else:
        print(f"✗ {screen_hash} is UNREACHABLE - navigation may be broken")
```

---

## Performance Considerations

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `find_path` | O(V + E) | BFS, efficient for single path |
| `find_all_paths` | O(V^E) exponential | Use small `max_depth` to limit |
| `is_reachable` | O(V + E) | Just runs `find_path` internally |
| `get_connected_component` | O(V + E) | One BFS traversal |
| `detect_cycles` | O(V + E) | DFS traversal, linear |
| `get_graph_diameter` | O(V² × (V+E)) | Expensive, cache result |
| `get_visualization_context` | O(E) | Linear in edge count |

**Recommendations:**
- Cache `get_graph_diameter()` results - recompute only on significant graph changes
- Use `max_depth` limits to prevent expensive explorations
- For large graphs (1000+ screens), consider batching path queries

---

## Integration with Exploration

These navigation features automatically integrate with the exploration workflow:

```python
# During exploration, the knowledge graph is built and queried:
kg = KnowledgeGraph()
await kg.load()  # Load prior knowledge

# During each exploration step, new screens/transitions are added:
await kg.add_screen(captured_state, description="Product search results")
await kg.record_transition(current_hash, action, next_hash)

# Navigation queries help guide exploration strategy:
if kg.detect_cycles(start_hash=current_hash):
    print("⚠️  Current screen might be in a loop")

reachable_count = len(kg.get_connected_component(current_hash))
print(f"From here, can explore {reachable_count} screens")
```

---

## Testing

Comprehensive unit tests are included in [tests/unit/test_knowledge_graph_navigation.py](../../tests/unit/test_knowledge_graph_navigation.py).

Run tests with:
```bash
pytest tests/unit/test_knowledge_graph_navigation.py -v
```

All 28 tests verify:
- ✓ Shortest path finding (7 tests)
- ✓ All paths discovery (4 tests)
- ✓ Reachability analysis (6 tests)
- ✓ Cycle detection (5 tests)
- ✓ Graph diameter (3 tests)
- ✓ Visualization context (3 tests)
