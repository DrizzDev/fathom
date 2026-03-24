#!/usr/bin/env python3
"""
Demonstration of comprehensive exploration report generation.

Shows how the report is generated at the end of exploration runs.
"""

import asyncio

from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph
from fathom.services.exploration_report import ExplorationReportGenerator


async def create_demo_graph():
    """Create a realistic food delivery app graph for demo."""
    kg = KnowledgeGraph()
    kg._KnowledgeGraph__loaded = True

    # Define realistic screens
    screens = {
        "splash": GraphNode("hash_splash", "splash", "Splash Screen", 0, 0, 1),
        "login": GraphNode("hash_login", "auth.LoginActivity", "Login Screen", 0, 0, 3),
        "home": GraphNode("hash_home", "food.HomeActivity", "Home Feed", 0, 0, 8),
        "search": GraphNode("hash_search", "food.SearchActivity", "Search Results", 0, 0, 5),
        "restaurant": GraphNode(
            "hash_restaurant", "food.RestaurantActivity", "Restaurant Detail", 0, 0, 4
        ),
        "menu": GraphNode("hash_menu", "food.MenuActivity", "Item Menu", 0, 0, 6),
        "cart": GraphNode("hash_cart", "food.CartActivity", "Shopping Cart", 0, 0, 5),
        "checkout": GraphNode("hash_checkout", "food.CheckoutActivity", "Checkout", 0, 0, 3),
        "payment": GraphNode("hash_payment", "payment.PaymentActivity", "Payment", 0, 0, 2),
        "confirmation": GraphNode(
            "hash_confirmation", "order.ConfirmationActivity", "Order Confirm", 0, 0, 2
        ),
        "instamart": GraphNode(
            "hash_instamart", "instamart.InstaActivity", "Instamart Shop", 0, 0, 3
        ),
        "settings": GraphNode("hash_settings", "account.SettingsActivity", "Settings", 0, 0, 4),
    }

    # Add all screens
    for screen in screens.values():
        kg._KnowledgeGraph__nodes[screen.visual_hash] = screen

    # Define transitions
    transitions = [
        ("hash_splash", "hash_login", "tap", "LOGIN_BUTTON"),
        ("hash_login", "hash_home", "tap", "LOGIN_SUBMIT"),
        ("hash_home", "hash_search", "tap", "SEARCH_BAR"),
        ("hash_home", "hash_instamart", "tap", "INSTAMART_ICON"),
        ("hash_home", "hash_settings", "tap", "SETTINGS_ICON"),
        ("hash_home", "hash_restaurant", "swipe", "RESTAURANT_CARD"),
        ("hash_search", "hash_restaurant", "tap", "RESULT_ITEM"),
        ("hash_search", "hash_home", "tap", "BACK_BUTTON"),
        ("hash_restaurant", "hash_menu", "tap", "MENU_TAB"),
        ("hash_restaurant", "hash_home", "tap", "BACK_BUTTON"),
        ("hash_menu", "hash_cart", "tap", "ADD_TO_CART"),
        ("hash_menu", "hash_restaurant", "tap", "BACK_BUTTON"),
        ("hash_cart", "hash_checkout", "tap", "CHECKOUT_BUTTON"),
        ("hash_cart", "hash_menu", "tap", "BACK_BUTTON"),
        ("hash_checkout", "hash_payment", "tap", "PAYMENT_BUTTON"),
        ("hash_payment", "hash_confirmation", "tap", "CONFIRM_PAYMENT"),
        ("hash_confirmation", "hash_home", "tap", "HOME_BUTTON"),
        ("hash_instamart", "hash_search", "tap", "SEARCH_INSTAMART"),
        ("hash_instamart", "hash_cart", "tap", "ADD_ITEMS"),
        ("hash_settings", "hash_home", "tap", "BACK_BUTTON"),
        ("hash_home", "hash_home", "swipe", "REFRESH"),
    ]

    # Add edges
    for src, dst, action_type, action_target in transitions:
        edge = GraphEdge(src, dst, action_type, action_target, 1, 0, 0)
        kg._KnowledgeGraph__edges.setdefault(src, []).append(edge)

    return kg


async def main():
    print("=" * 80)
    print("COMPREHENSIVE EXPLORATION REPORT GENERATION DEMO")
    print("=" * 80)

    # Create demo graph
    print("\n📊 Creating demo knowledge graph...")
    kg = await create_demo_graph()
    print(f"   Created: {kg.node_count} screens, {kg.edge_count} transitions")

    # Generate report
    print("\n📝 Generating comprehensive exploration report...")
    report_gen = ExplorationReportGenerator(kg)
    report = report_gen.generate_full_report(
        workflow_id="exploration_demo_001",
        duration_seconds=45.3,
        target_package="in.swiggy.android",
    )

    # Save report
    print("\n💾 Saving report to file...")
    report_path = await report_gen.save_report(report, output_dir="assets/reports")
    print(f"   ✓ Saved to: {report_path}")

    # Display human-readable summary
    print("\n" + "=" * 80)
    summary = report_gen.export_report_summary(report)
    print(summary)
    print("=" * 80)

    # Display detailed JSON structure
    print("\n📋 DETAILED REPORT STRUCTURE (JSON)")
    print("=" * 80)

    print("\n🔹 METADATA:")
    for key, value in report["metadata"].items():
        print(f"   {key}: {value}")

    print("\n🔹 SUMMARY STATISTICS:")
    for key, value in report["summary"].items():
        if isinstance(value, list):
            print(f"   {key}: {len(value)} items")
        else:
            print(f"   {key}: {value}")

    print("\n🔹 GRAPH ANALYSIS:")
    graph = report["graph_analysis"]
    print(f"   diameter: {graph['diameter']}")
    print(f"   cycle_count: {graph['cycle_count']}")
    print(f"   connected_components: {graph['connected_components']['total_components']}")
    if graph["cycles"]:
        print("\n   Top Cycles:")
        for i, cycle in enumerate(graph["cycles"][:3], 1):
            print(f"     {i}. {' → '.join(cycle['screens'])} (length: {cycle['length']})")

    print("\n🔹 TOP SCREENS:")
    for i, screen in enumerate(report["screen_rankings"]["most_visited"][:5], 1):
        print(
            f"   {i}. {screen['description'][:40]:40} "
            f"(visits: {screen['visits']}, edges: {screen['outgoing_edges']})"
        )

    print("\n🔹 CRITICAL SCREENS:")
    for screen in report["screen_rankings"]["critical_screens"][:3]:
        print(
            f"   • {screen['name'][:40]:40} ({screen['type']} - {screen['connectivity']} connections)"
        )

    print("\n🔹 REACHABILITY ANALYSIS:")
    for screen_name, reach_data in report["reachability_analysis"].items():
        print(
            f"   {screen_name[:35]:35} → Forward: {reach_data['forward_coverage']:>6}, "
            f"Backward: {reach_data['backward_reach']:>3}"
        )

    print("\n🔹 KEY NAVIGATION PATHS:")
    for path in report["navigation_paths"][:3]:
        print(f"   {path['from']} → {path['to']} ({path['steps']} steps)")

    print("\n🔹 ACTIVITY BREAKDOWN:")
    for activity, info in report["activity_breakdown"].items():
        print(f"   {activity[:40]:40} → {info['screen_count']} screens")

    print("\n🔹 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"   {rec}")

    print("\n" + "=" * 80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("=" * 80)

    # Show file info
    if report_path.exists():
        file_size = report_path.stat().st_size
        print(f"\n📁 Report saved: {report_path}")
        print(f"   Size: {file_size:,} bytes")
        print("   Format: JSON")
        print("\n   You can view this report:")
        print(f"   - Programmatically: json.load(open('{report_path}'))")
        print(f"   - In terminal: cat {report_path} | jq .")


if __name__ == "__main__":
    asyncio.run(main())
