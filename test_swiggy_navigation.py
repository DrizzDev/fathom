#!/usr/bin/env python3
"""Test navigation scenarios for Swiggy app with real exploration data."""

import asyncio
from pathlib import Path

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph


async def main():
    kg = KnowledgeGraph()
    await kg.load()

    print(f"Total screens explored: {kg.node_count}")
    print(f"Total transitions: {kg.edge_count}\n")

    # If no data, check available memory
    if kg.node_count == 0:
        print("⚠️  No data in knowledge graph yet.")
        print("\nAvailable memory paths:")
        memory_dir = Path("assets/memory/com.google.android.apps.nexuslauncher")
        if memory_dir.exists():
            print(f"  - {memory_dir.name}: {len(list(memory_dir.glob('*')))} files")

        swiggy_dir = Path("assets/memory/in.swiggy.android")
        if swiggy_dir.exists():
            print(f"  - {swiggy_dir.name}: {len(list(swiggy_dir.glob('*')))} files")

        print("\n💡 Run exploration to generate graph data:")
        print("  python -m fathom.cli explore -p in.swiggy.android --max-steps 50")
        return

    # Find screens by activity
    activities = {}
    for node in kg.nodes.values():
        activity = node.activity
        if activity not in activities:
            activities[activity] = []
        activities[activity].append(node)

    print("Screens by activity:")
    for activity, nodes in sorted(activities.items()):
        print(f"  {activity}: {len(nodes)} screens")

    print("\n" + "=" * 60)
    print("Testing Navigation Scenarios")
    print("=" * 60 + "\n")

    # Get activity with most screens (likely main flow)
    if activities:
        main_activity = max(activities.items(), key=lambda x: len(x[1]))[0]
        main_screens = activities[main_activity]

        print(f"Main activity: {main_activity} ({len(main_screens)} screens)\n")

        if len(main_screens) >= 2:
            start = main_screens[0]
            end = main_screens[-1]

            print("Looking for path from:")
            print(f"  {start.description or 'START'}\n")
            print("To:")
            print(f"  {end.description or 'END'}\n")

            path = kg.find_path(start.visual_hash, end.visual_hash, max_depth=30)

            if path:
                print(f"✓ Found path with {len(path) - 1} steps:\n")
                for i, (_node_hash, edge) in enumerate(path):
                    if edge:
                        print(f"  {i}. {edge.action_type:10} '{edge.action_target}'")
                    else:
                        print(f"  {i}. START")
            else:
                print("✗ No path found between these screens")

        # Show reachability stats
        if main_screens:
            first = main_screens[0]
            reachable = kg.get_connected_component(first.visual_hash)
            print(f"\n📊 Reachability from first screen in {main_activity}:")
            print(f"   Can reach: {len(reachable)} out of {kg.node_count} total screens")
            print(f"   Coverage: {(len(reachable) / kg.node_count * 100):.1f}%")

        # Cycle detection
        all_cycles = kg.detect_cycles()
        if all_cycles:
            print(f"\n⚠️  Found {len(all_cycles)} cycle(s) in the graph:")
            for i, cycle in enumerate(all_cycles[:3], 1):
                print(f"   Cycle {i}: {len(cycle) - 1} screens in loop")
        else:
            print("\n✓ No cycles detected (graph is acyclic)")

        # Graph diameter
        diameter = kg.get_graph_diameter()
        if diameter:
            print(f"\n📏 Graph diameter: {diameter} steps")

        # Show all screens
        print(f"\n{'=' * 60}")
        print("All Explored Screens:")
        print("=" * 60)
        for i, node in enumerate(
            sorted(kg.nodes.values(), key=lambda n: n.visit_count, reverse=True)[:10], 1
        ):
            context = kg.get_visualization_context(node.visual_hash)
            print(f"{i}. {node.description or node.visual_hash[:16]}")
            print(f"   Activity: {node.activity}")
            print(
                f"   Visits: {node.visit_count}, Outgoing: {len(context['outgoing_edges'])}, Inbound: {len(context['inbound_edges'])}"
            )


if __name__ == "__main__":
    asyncio.run(main())
