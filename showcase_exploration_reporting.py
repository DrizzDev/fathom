#!/usr/bin/env python3
"""
Complete showcase of exploration reporting integration.

Demonstrates:
1. Known graph from previous explorations
2. Comprehensive report generation
3. CLI-style display
4. JSON export
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph
from fathom.services.exploration_report import ExplorationReportGenerator

console = Console()


async def create_realistic_app_graph():
    """Create a realistic graph from a real app exploration."""
    kg = KnowledgeGraph()
    kg._KnowledgeGraph__loaded = True

    # Real app scenario: Food delivery app with full user journey
    screens = {
        # Auth flow
        "splash": GraphNode("hash_splash", "com.app.SplashActivity", "App Splash", 0, 0, 1),
        "login": GraphNode("hash_login", "com.app.auth.LoginActivity", "Login Screen", 0, 0, 5),
        "register": GraphNode("hash_reg", "com.app.auth.RegisterActivity", "Registration", 0, 0, 2),
        # Main navigation
        "home": GraphNode("hash_home", "com.app.home.HomeActivity", "Home Feed", 0, 0, 25),
        "search": GraphNode(
            "hash_search", "com.app.food.SearchActivity", "Search Results", 0, 0, 12
        ),
        "categories": GraphNode(
            "hash_cat", "com.app.food.CategoriesActivity", "Categories", 0, 0, 8
        ),
        # Shopping flow
        "restaurant": GraphNode(
            "hash_rest", "com.app.restaurant.DetailActivity", "Restaurant", 0, 0, 18
        ),
        "menu": GraphNode("hash_menu", "com.app.menu.MenuActivity", "Item Menu", 0, 0, 15),
        "cart": GraphNode("hash_cart", "com.app.cart.CartActivity", "Shopping Cart", 0, 0, 12),
        "review_order": GraphNode(
            "hash_review", "com.app.order.ReviewActivity", "Review Order", 0, 0, 8
        ),
        # Checkout & Payment
        "location": GraphNode(
            "hash_loc", "com.app.checkout.LocationActivity", "Delivery Location", 0, 0, 7
        ),
        "payment": GraphNode("hash_pay", "com.app.payment.PaymentActivity", "Payment", 0, 0, 6),
        "confirmation": GraphNode(
            "hash_conf", "com.app.order.ConfirmationActivity", "Order Confirmed", 0, 0, 6
        ),
        # Order tracking
        "orders": GraphNode(
            "hash_orders", "com.app.orders.OrderListActivity", "My Orders", 0, 0, 10
        ),
        "order_detail": GraphNode(
            "hash_odet", "com.app.orders.OrderDetailActivity", "Order Detail", 0, 0, 8
        ),
        # Account
        "account": GraphNode("hash_acc", "com.app.account.ProfileActivity", "Account", 0, 0, 9),
        "settings": GraphNode("hash_set", "com.app.account.SettingsActivity", "Settings", 0, 0, 5),
    }

    for screen in screens.values():
        kg._KnowledgeGraph__nodes[screen.visual_hash] = screen

    # Realistic transitions
    transitions = [
        # Auth
        ("hash_splash", "hash_login", "tap", "LOGIN_BUTTON"),
        ("hash_login", "hash_home", "tap", "SUBMIT_LOGIN"),
        ("hash_login", "hash_reg", "tap", "SIGNUP_LINK"),
        ("hash_reg", "hash_home", "tap", "COMPLETE_SIGNUP"),
        # Home navigation
        ("hash_home", "hash_search", "tap", "SEARCH_BAR"),
        ("hash_home", "hash_categories", "tap", "CATEGORIES_ICON"),
        ("hash_home", "hash_orders", "tap", "ORDERS_ICON"),
        ("hash_home", "hash_account", "tap", "ACCOUNT_ICON"),
        ("hash_home", "hash_restaurant", "swipe", "RESTAURANT_CARD"),
        # Search and discovery
        ("hash_search", "hash_restaurant", "tap", "RESULT_ITEM"),
        ("hash_search", "hash_home", "tap", "BACK"),
        ("hash_search", "hash_categories", "tap", "FILTER"),
        # Restaurant browsing
        ("hash_restaurant", "hash_menu", "tap", "VIEW_ITEMS"),
        ("hash_restaurant", "hash_home", "tap", "BACK"),
        ("hash_restaurant", "hash_search", "tap", "BACK"),
        # Ordering
        ("hash_menu", "hash_cart", "tap", "ADD_TO_CART"),
        ("hash_menu", "hash_restaurant", "tap", "CLOSE_DETAIL"),
        ("hash_cart", "hash_review_order", "tap", "PROCEED"),
        ("hash_cart", "hash_menu", "tap", "CONTINUE_SHOPPING"),
        ("hash_cart", "hash_home", "tap", "BACK"),
        # Checkout
        ("hash_review_order", "hash_location", "tap", "SELECT_LOCATION"),
        ("hash_review_order", "hash_cart", "tap", "BACK"),
        ("hash_location", "hash_payment", "tap", "CONFIRM_LOCATION"),
        ("hash_location", "hash_review_order", "tap", "BACK"),
        ("hash_payment", "hash_confirmation", "tap", "COMPLETE_PAYMENT"),
        ("hash_payment", "hash_location", "tap", "CHANGE_LOCATION"),
        # After order
        ("hash_confirmation", "hash_orders", "tap", "VIEW_ALL_ORDERS"),
        ("hash_confirmation", "hash_home", "tap", "HOME"),
        # Order history
        ("hash_orders", "hash_order_detail", "tap", "ORDER_ITEM"),
        ("hash_orders", "hash_home", "tap", "BACK"),
        ("hash_order_detail", "hash_orders", "tap", "BACK"),
        ("hash_order_detail", "hash_payment", "tap", "REPEAT_ORDER"),
        # Account
        ("hash_account", "hash_settings", "tap", "SETTINGS"),
        ("hash_account", "hash_orders", "tap", "MY_ORDERS"),
        ("hash_account", "hash_home", "tap", "BACK"),
        ("hash_settings", "hash_account", "tap", "BACK"),
        # Cycles/loops
        ("hash_home", "hash_home", "swipe", "REFRESH"),
        ("hash_restaurant", "hash_restaurant", "swipe", "REFRESH"),
    ]

    for src, dst, action_type, action_target in transitions:
        edge = GraphEdge(src, dst, action_type, action_target, 1, 0, 0)
        kg._KnowledgeGraph__edges.setdefault(src, []).append(edge)

    return kg


async def main():
    console.clear()

    print("╭" + "─" * 78 + "╮")
    print("│" + " " * 78 + "│")
    print("│" + " FATHOM EXPLORATION REPORTING - COMPLETE INTEGRATION SHOWCASE ".center(78) + "│")
    print("│" + " " * 78 + "│")
    print("╰" + "─" * 78 + "╯")
    print()

    # Step 1: Create graph
    print("📊 [Step 1] Loading knowledge graph from exploration...")
    kg = await create_realistic_app_graph()
    print(f"   ✓ Loaded: {kg.node_count} screens, {kg.edge_count} transitions")
    print()

    # Step 2: Generate report
    print("📝 [Step 2] Generating comprehensive exploration report...")
    report_gen = ExplorationReportGenerator(kg)
    record = report_gen.generate_full_report(
        workflow_id="food_delivery_exploration_v1",
        duration_seconds=127.5,
        target_package="com.food.delivery.app",
    )
    print(f"   ✓ Report generated ({record['summary']['unique_screens']} screens analyzed)")
    print()

    # Step 3: Save report
    print("💾 [Step 3] Persisting report to disk...")
    report_path = await report_gen.save_report(record, output_dir="assets/reports")
    file_size = report_path.stat().st_size
    print(f"   ✓ Saved: {report_path.name} ({file_size:,} bytes)")
    print()

    # Step 4: Display CLI-style output (as users see it)
    print("=" * 80)
    print("EXPLORATION COMPLETE - GENERATED REPORT DISPLAYED IN CLI".center(80))
    print("=" * 80)
    print()

    # Graph Analysis table
    graph_table = Table(title="Graph Analysis & Insights", border_style="magenta")
    graph_table.add_column("Metric", style="cyan")
    graph_table.add_column("Value", style="magenta")

    graph_analysis = record["graph_analysis"]
    graph_table.add_row("Graph Diameter", str(graph_analysis["diameter"] or "N/A"))
    graph_table.add_row("Cycles Detected", str(graph_analysis["cycle_count"]))
    graph_table.add_row(
        "Connected Components", str(graph_analysis["connected_components"]["total_components"])
    )
    graph_table.add_row("Total Edges", str(kg.edge_count))

    console.print(graph_table)
    print()

    # Critical screens
    critical = report_gen._identify_critical_screens()
    if critical:
        critical_table = Table(title="Critical Screens (Hubs & Bottlenecks)", border_style="yellow")
        critical_table.add_column("Screen", style="cyan", width=35)
        critical_table.add_column("Type", style="yellow")
        critical_table.add_column("Connections", style="magenta", justify="right")

        for screen in critical[:8]:
            critical_table.add_row(
                screen["name"][:35],
                screen["type"],
                str(screen["connectivity"]),
            )

        console.print(critical_table)
        print()

    # Reachability
    reachability = report_gen._analyze_reachability()
    if reachability:
        reach_table = Table(title="Reachability Analysis", border_style="cyan")
        reach_table.add_column("Screen", style="cyan", width=30)
        reach_table.add_column("Forward", style="magenta", justify="right")
        reach_table.add_column("Backward", style="yellow", justify="right")

        for screen_name, reach_data in reachability.items():
            reach_table.add_row(
                screen_name[:30],
                reach_data["forward_coverage"],
                str(reach_data["backward_reach"]),
            )

        console.print(reach_table)
        print()

    # Recommendations
    stats_for_rec = {
        "unique_screens": record["summary"]["unique_screens"],
        "unexplored": record["summary"]["unexplored_screens"],
    }
    recommendations = report_gen._generate_recommendations(
        stats_for_rec, record["graph_analysis"]["cycles"]
    )
    if recommendations:
        rec_panel = Panel(
            "\n".join([f"• {rec}" for rec in recommendations]),
            title="[bold]Recommendations[/bold]",
            border_style="green",
        )
        console.print(rec_panel)
        print()

    # Step 5: Show report details
    print("=" * 80)
    print()
    print("📋 REPORT METADATA:")
    meta = record["metadata"]
    for key, value in meta.items():
        print(f"   {key:35} {str(value):40}")

    print()
    print("📊 SUMMARY STATISTICS:")
    summary = record["summary"]
    for key, value in summary.items():
        if key != "activities":
            print(f"   {key:35} {str(value):40}")

    print()
    print("🔍 TOP SCREENS BY VISITS:")
    for i, screen in enumerate(record["screen_rankings"]["most_visited"][:5], 1):
        print(
            f"   {i}. {screen['description'][:40]:40} (visits: {screen['visits']}, edges: {screen['outgoing_edges']})"
        )

    print()
    print("🔄 DETECTED CYCLES:")
    for i, cycle in enumerate(record["graph_analysis"]["cycles"][:4], 1):
        path = " → ".join(cycle["screens"][:3]) + ("..." if cycle["length"] > 3 else "")
        print(f"   {i}. {path}")

    print()
    print("🛣️  KEY NAVIGATION PATHS (Entry → Exit):")
    if record.get("navigation_paths"):
        entry_paths = [
            p for p in record["navigation_paths"] if p.get("type") == "entry_exit_journey"
        ]
        longest_paths = [p for p in record["navigation_paths"] if p.get("type") == "longest_path"]

        if entry_paths:
            for i, navigation in enumerate(entry_paths[:2], 1):
                path_str = " → ".join(navigation["path"][:5])
                if len(navigation["path"]) > 5:
                    path_str += " → ..."
                print(
                    f"   {i}. {navigation['from']:20} ⟶ {navigation['to']:20} ({navigation['steps']} steps)"
                )
                print(f"      {path_str}")

        if longest_paths:
            print()
            print("📏 LONGEST PATHS IN GRAPH (Diameter Paths):")
            for i, longest in enumerate(longest_paths, 1):
                path_str = " → ".join(longest["path"][:6])
                if len(longest["path"]) > 6:
                    path_str += " → ..."
                print(
                    f"   {i}. {longest['from']:20} ⟶ {longest['to']:20} ({longest['steps']} steps)"
                )
                print(f"      {path_str}")
    else:
        print("   (No significant entry-exit paths found)")

    print()
    print("=" * 80)
    print()

    # Step 6: Show JSON structure
    print("📁 GENERATED JSON REPORT:")
    print(f"   Path: {report_path}")
    print(f"   Size: {file_size:,} bytes")
    print("   Format: Prettified JSON")
    print()
    print("   Top-level sections:")
    print("   ├── metadata (generation info, workflow id, duration)")
    print("   ├── summary (screens, transitions, activities)")
    print("   ├── graph_analysis (diameter, cycles, components)")
    print("   ├── screen_rankings (most visited, critical)")
    print("   ├── reachability_analysis (forward/backward reach)")
    print("   ├── navigation_paths (key user journeys)")
    print("   ├── activity_breakdown (screens by activity)")
    print("   └── recommendations (actionable insights)")

    print()
    print("=" * 80)
    print()

    # Step 7: Display markdown version
    print("📄 HUMAN-READABLE MARKDOWN REPORT:")
    md_path = report_path.with_suffix(".md")
    if md_path.exists():
        from rich.markdown import Markdown

        markdown_content = md_path.read_text()
        console.print(Markdown(markdown_content))
        print()
        print(f"   Markdown file: {md_path}")
    else:
        print(f"   Markdown file not found: {md_path}")

    print()
    print("=" * 80)
    print()
    print("✅ INTEGRATION COMPLETE!")
    print()
    print("Features implemented:")
    print("   ✓ Knowledge graph analysis (nav, cycles, reachability)")
    print("   ✓ Comprehensive report generation")
    print("   ✓ Automatic report persistence")
    print("   ✓ CLI integration with rich tables")
    print("   ✓ JSON export for programmatic access")
    print("   ✓ Actionable recommendations")
    print()
    print("Next exploration run will automatically:")
    print("   → Analyze the knowledge graph")
    print("   → Generate comprehensive report")
    print("   → Display insights in CLI")
    print("   → Save report to assets/reports/")
    print()


if __name__ == "__main__":
    asyncio.run(main())
