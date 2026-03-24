#!/usr/bin/env python3
"""
Comprehensive demonstration of knowledge graph navigation features.
Creates a realistic app flow and shows all query capabilities.
"""

import asyncio

from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph


async def main():
    # Create a demo graph representing a food delivery app flow
    kg = KnowledgeGraph()

    # Bypass SQLite, work with in-memory structures directly for demo
    kg._KnowledgeGraph__loaded = True

    # Define realistic screen nodes
    screens = {
        "splash": GraphNode("hash_splash", "splash", "Splash Screen", 0, 0, 1),
        "login": GraphNode("hash_login", "auth.LoginActivity", "Login Screen", 0, 0, 2),
        "home": GraphNode("hash_home", "food.HomeActivity", "Home Feed", 0, 0, 5),
        "search": GraphNode("hash_search", "food.SearchActivity", "Search Results", 0, 0, 3),
        "restaurant": GraphNode(
            "hash_restaurant", "food.RestaurantActivity", "Restaurant Detail", 0, 0, 2
        ),
        "menu": GraphNode("hash_menu", "food.MenuActivity", "Item Menu", 0, 0, 4),
        "cart": GraphNode("hash_cart", "food.CartActivity", "Shopping Cart", 0, 0, 3),
        "checkout": GraphNode("hash_checkout", "food.CheckoutActivity", "Checkout", 0, 0, 2),
        "payment": GraphNode("hash_payment", "payment.PaymentActivity", "Payment", 0, 0, 1),
        "confirmation": GraphNode(
            "hash_confirmation", "order.ConfirmationActivity", "Order Confirm", 0, 0, 1
        ),
        "instamart": GraphNode(
            "hash_instamart", "instamart.InstaActivity", "Instamart Shop", 0, 0, 2
        ),
        "settings": GraphNode("hash_settings", "account.SettingsActivity", "Settings", 0, 0, 3),
    }

    # Add all screens to graph
    for screen in screens.values():
        kg._KnowledgeGraph__nodes[screen.visual_hash] = screen

    # Define realistic transitions (edges)
    transitions = [
        # Splash -> Login flow
        ("hash_splash", "hash_login", "tap", "LOGIN_BUTTON"),
        ("hash_login", "hash_home", "tap", "LOGIN_SUBMIT"),
        # Home flows
        ("hash_home", "hash_search", "tap", "SEARCH_BAR"),
        ("hash_home", "hash_instamart", "tap", "INSTAMART_ICON"),
        ("hash_home", "hash_settings", "tap", "SETTINGS_ICON"),
        ("hash_home", "hash_restaurant", "swipe", "RESTAURANT_CARD"),
        # Search and discovery
        ("hash_search", "hash_restaurant", "tap", "RESULT_ITEM"),
        ("hash_search", "hash_home", "tap", "BACK_BUTTON"),
        # Restaurant browsing
        ("hash_restaurant", "hash_menu", "tap", "MENU_TAB"),
        ("hash_restaurant", "hash_home", "tap", "BACK_BUTTON"),
        ("hash_restaurant", "hash_search", "tap", "BACK_BUTTON"),
        # Menu to cart
        ("hash_menu", "hash_cart", "tap", "ADD_TO_CART"),
        ("hash_menu", "hash_restaurant", "tap", "BACK_BUTTON"),
        # Cart flow
        ("hash_cart", "hash_checkout", "tap", "CHECKOUT_BUTTON"),
        ("hash_cart", "hash_menu", "tap", "BACK_BUTTON"),
        ("hash_cart", "hash_home", "tap", "BACK_BUTTON"),
        # Checkout and payment
        ("hash_checkout", "hash_payment", "tap", "PAYMENT_BUTTON"),
        ("hash_checkout", "hash_cart", "tap", "BACK_BUTTON"),
        ("hash_payment", "hash_confirmation", "tap", "CONFIRM_PAYMENT"),
        # Confirmation and home (cycle back)
        ("hash_confirmation", "hash_home", "tap", "HOME_BUTTON"),
        # Instamart flow
        ("hash_instamart", "hash_search", "tap", "SEARCH_INSTAMART"),
        ("hash_instamart", "hash_cart", "tap", "ADD_ITEMS"),
        ("hash_instamart", "hash_home", "tap", "BACK_BUTTON"),
        # Settings
        ("hash_settings", "hash_home", "tap", "BACK_BUTTON"),
        # Back navigation shortcuts creating cycles
        ("hash_home", "hash_home", "swipe", "REFRESH"),
    ]

    # Add edges
    for src, dst, action_type, action_target in transitions:
        edge = GraphEdge(src, dst, action_type, action_target, 1, 0, 0)
        kg._KnowledgeGraph__edges.setdefault(src, []).append(edge)

    print("=" * 70)
    print("🍕 FOOD DELIVERY APP - KNOWLEDGE GRAPH NAVIGATION DEMO")
    print("=" * 70)
    print("\n📊 Graph Stats:")
    print(f"   📱 Screens: {len(screens)}")
    print(f"   🔗 Transitions: {len(transitions)}")

    # =========================================================================
    # TEST 1: SHORTEST PATH
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: SHORTEST PATH FINDING (Home → Payment)")
    print("=" * 70)
    path = kg.find_path("hash_home", "hash_payment", max_depth=20)
    if path:
        print(f"\n✓ Found path with {len(path) - 1} steps:\n")
        for i, (node_hash, edge) in enumerate(path):
            scene = screens.get(node_hash)
            if scene is None:
                scene = next((s for s in screens.values() if s.visual_hash == node_hash), None)
            if edge:
                dest_scene = screens.get(edge.destination_hash, None)
                if dest_scene is None:
                    dest_scene = next(
                        (s for s in screens.values() if s.visual_hash == edge.destination_hash),
                        None,
                    )
                dest_name = dest_scene.description if dest_scene else "unknown"
                print(f"  {i:2d}. {edge.action_type:6} '{edge.action_target:20}' → {dest_name}")
            else:
                scene_name = scene.description if scene else "unknown"
                print(f"  {i:2d}. [START] {scene_name}")

    # =========================================================================
    # TEST 2: ALL POSSIBLE PATHS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ALL POSSIBLE PATHS (Home → Cart)")
    print("=" * 70)
    all_paths = kg.find_all_paths("hash_home", "hash_cart", max_depth=8)
    print(f"\n✓ Found {len(all_paths)} different path(s):\n")
    for path_num, path in enumerate(all_paths[:3], 1):
        print(f"  Path {path_num} ({len(path) - 1} steps):")
        for node_hash, edge in path:
            scene = screens.get(node_hash)
            if scene is None:
                scene = next((s for s in screens.values() if s.visual_hash == node_hash), None)
            if edge:
                dest_scene = screens.get(edge.destination_hash, None)
                if dest_scene is None:
                    dest_scene = next(
                        (s for s in screens.values() if s.visual_hash == edge.destination_hash),
                        None,
                    )
                dest_name = dest_scene.description if dest_scene else "unknown"
                print(f"    → {edge.action_type:6} → {dest_name}")
            else:
                scene_name = scene.description if scene else "unknown"
                print(f"    START: {scene_name}")
    if len(all_paths) > 3:
        print(f"  ... and {len(all_paths) - 3} more paths")

    # =========================================================================
    # TEST 3: REACHABILITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: REACHABILITY ANALYSIS")
    print("=" * 70)

    # Forward reachability from home
    forward = kg.get_connected_component("hash_home")
    reachable_screens = []
    for h in forward:
        scene = screens.get(h)
        if scene is None:
            scene = next((s for s in screens.values() if s.visual_hash == h), None)
        if scene:
            reachable_screens.append(scene.description)
    print("\n📍 From Home Screen:")
    print(
        f"   Can reach {len(forward)} out of {len(screens)} screens ({len(forward) / len(screens) * 100:.0f}%)"
    )
    print(f"   Reachable: {', '.join(reachable_screens[:5])}")

    # Backward reachability to payment
    backward = kg.get_reverse_connected_component("hash_payment")
    backward_screens = []
    for h in backward:
        if h != "hash_payment":
            scene = screens.get(h)
            if scene is None:
                scene = next((s for s in screens.values() if s.visual_hash == h), None)
            if scene:
                backward_screens.append(scene.description)
    print("\n📍 To Payment Screen:")
    print(f"   Reachable from {len(backward)} out of {len(screens)} screens")
    print(f"   Can reach from: {', '.join(backward_screens[:4])}")

    # Quick reachability check
    is_reachable = kg.is_reachable("hash_home", "hash_confirmation")
    print("\n📍 Can user complete order?")
    print(f"   Home → Confirmation: {'✓ YES' if is_reachable else '✗ NO'}")

    # =========================================================================
    # TEST 4: CYCLE DETECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: CYCLE DETECTION")
    print("=" * 70)
    cycles = kg.detect_cycles()
    print(f"\n🔄 Found {len(cycles)} cycle(s):")
    for i, cycle in enumerate(cycles[:5], 1):
        cycle_screens = []
        for c in cycle:
            scene = screens.get(c)
            if scene is None:
                scene = next((s for s in screens.values() if s.visual_hash == c), None)
            if scene:
                cycle_screens.append(scene.description)
        cycle_str = " → ".join(cycle_screens[:3])
        if len(cycle) > 3:
            cycle_str += "..."
        print(f"  Cycle {i}: {cycle_str}")

    # =========================================================================
    # TEST 5: VISUALIZATION CONTEXT
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: VISUALIZATION CONTEXT (Restaurant Screen)")
    print("=" * 70)
    context = kg.get_visualization_context("hash_restaurant")
    node_info = context["node"]
    print("\n📊 Screen Details:")
    print(f"   Name: {node_info['description']}")
    print(f"   Activity: {node_info['activity']}")
    print(f"   Visits: {node_info['visit_count']}")
    print("\n🔗 Connectivity:")
    print(f"   Outgoing edges: {len(context['outgoing_edges'])}")
    for edge_info in context["outgoing_edges"][:3]:
        print(f"     → {edge_info['action_type']:6} → {edge_info['destination_description']}")
    print(f"   Inbound edges: {len(context['inbound_edges'])}")
    for edge_info in context["inbound_edges"][:3]:
        print(f"     ← from {edge_info['source_description']}")
    print("\n📈 Reachability:")
    print(f"   Can reach {context['forward_reachable']} screens")
    print(f"   Reachable from {context['backward_reachable']} screens")
    print(f"   Part of cycle: {'⚠️ YES' if context['in_cycle'] else 'No'}")

    # =========================================================================
    # TEST 6: GRAPH DIAMETER
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: GRAPH DIAMETER")
    print("=" * 70)
    diameter = kg.get_graph_diameter()
    if diameter:
        print(f"\n📏 Maximum steps between any two screens: {diameter}")
        print("   (Longest shortest path in the graph)")

    # =========================================================================
    # TEST 7: COVERAGE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 7: COVERAGE & BOTTLENECK ANALYSIS")
    print("=" * 70)
    print("\nScreen Coverage from each starting point:")
    for start_hash in ["hash_login", "hash_home", "hash_splash"]:
        reachable = kg.get_connected_component(start_hash)
        coverage = len(reachable) / len(screens) * 100
        start_scene = screens.get(start_hash)
        if start_scene is None:
            start_scene = next((s for s in screens.values() if s.visual_hash == start_hash), None)
        start_name = start_scene.description if start_scene else start_hash
        print(
            f"  From {start_name:20} → {coverage:5.1f}% ({len(reachable)}/{len(screens)} screens)"
        )

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
