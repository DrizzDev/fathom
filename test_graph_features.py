#!/usr/bin/env python3
"""Quick test script for knowledge graph navigation features."""

import asyncio

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph


async def main():
    # Load your existing exploration data
    kg = KnowledgeGraph()
    await kg.load()

    print("📊 Graph Statistics:")
    print(f"   Screens: {kg.node_count}")
    print(f"   Transitions: {kg.edge_count}")
    print()

    # Get the first screen to use as a starting point
    screens = list(kg.nodes.values())
    if not screens:
        print("❌ No screens in knowledge graph yet!")
        return

    start_screen = screens[0]
    print(f"🎬 Starting screen: {start_screen.description or start_screen.visual_hash}")
    print()

    # Test 1: Get reachable screens
    print("=" * 60)
    print("TEST 1: Reachable Screens (Forward)")
    print("=" * 60)
    reachable = kg.get_connected_component(start_screen.visual_hash)
    print(f"From this screen, can reach {len(reachable)} screens total")
    for screen_hash in list(reachable)[:5]:
        node = kg.get_screen(screen_hash)
        print(f"  ✓ {node.description or node.visual_hash}")
    if len(reachable) > 5:
        print(f"  ... and {len(reachable) - 5} more")
    print()

    # Test 2: Detect cycles
    print("=" * 60)
    print("TEST 2: Cycle Detection")
    print("=" * 60)
    cycles = kg.detect_cycles(start_hash=start_screen.visual_hash)
    print(f"Found {len(cycles)} cycle(s) reachable from this screen:")
    for i, cycle in enumerate(cycles[:3], 1):
        cycle_screens = [kg.get_screen(h) for h in cycle[:-1]]
        print(f"\n  Cycle {i} ({len(cycle) - 1} screens):")
        for node in cycle_screens[:3]:
            print(f"    → {node.description or node.visual_hash}")
        if len(cycle) > 4:
            print(f"    ... and {len(cycle) - 3} more")
    if not cycles:
        print("  (No cycles found)")
    print()

    # Test 3: Find path to another screen
    if len(screens) > 1:
        end_screen = screens[-1]
        print("=" * 60)
        print("TEST 3: Shortest Path")
        print("=" * 60)
        print(f"From:  {start_screen.description or start_screen.visual_hash}")
        print(f"To:    {end_screen.description or end_screen.visual_hash}")

        path = kg.find_path(start_screen.visual_hash, end_screen.visual_hash)
        if path:
            print(f"✓ Path found ({len(path) - 1} steps):")
            for i, (node_hash, edge) in enumerate(path):
                node = kg.get_screen(node_hash)
                if edge:
                    print(f"  {i}. {edge.action_type} '{edge.action_target}'")
                    dest_node = kg.get_screen(edge.destination_hash)
                    print(f"     → {dest_node.description or dest_node.visual_hash}")
                else:
                    print(f"  0. START: {node.description or node.visual_hash}")
        else:
            print("✗ No path found between these screens")
        print()

    # Test 4: Visualization context
    print("=" * 60)
    print("TEST 4: Visualization Context")
    print("=" * 60)
    context = kg.get_visualization_context(start_screen.visual_hash)
    print(f"Screen:        {context['node']['description']}")
    print(f"Activity:      {context['node']['activity']}")
    print(f"Visits:        {context['node']['visit_count']}")
    print(f"Forward reach: {context['forward_reachable']} screens")
    print(f"Inbound edges: {len(context['inbound_edges'])}")
    print(f"Outbound edges: {len(context['outgoing_edges'])}")
    print(f"In cycle:      {'Yes ⚠️' if context['in_cycle'] else 'No'}")
    print()

    # Test 5: Graph diameter
    print("=" * 60)
    print("TEST 5: Graph Diameter")
    print("=" * 60)
    diameter = kg.get_graph_diameter()
    if diameter:
        print(f"Maximum steps between any two screens: {diameter}")
    else:
        print("Graph is too small or disconnected for diameter calculation")
    print()

    print("✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
