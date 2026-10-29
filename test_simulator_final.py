#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Автотести симулятора ШХТ: python3 -m unittest -v test_simulator"""
import random
import unittest

import simulator_final as S


class TestTopology(unittest.TestCase):
    def test_formula_distance_equals_bfs(self):
        for n in (4, 6, 8, 10):
            a, b = S.chords_for(n)
            T = S.ChessChordalTorus(n, a, b)
            for u in range(0, T.N, 3):
                ref = T.bfs(u)
                for v in range(T.N):
                    self.assertEqual(T.dist(u, v), ref[v], (n, u, v))

    def test_basic_metrics_n16(self):
        T = S.ChessChordalTorus(16, 2, 5)
        d, avg = T.metrics()
        self.assertEqual(d, 5)
        self.assertAlmostEqual(avg, 3.6706, places=3)
        self.assertTrue(all(len(x) == 6 for x in T.nbrs))

    def test_fault_free_routes_are_shortest(self):
        T = S.ChessChordalTorus(8, 1, 4)
        for u in range(T.N):
            for v in range(T.N):
                if u != v:
                    r = S.route_one_to_one(T, u, v)
                    self.assertTrue(r.delivered)
                    self.assertEqual(r.hops, T.dist(u, v))


class TestTrees(unittest.TestCase):
    def test_tree_height_equals_eccentricity(self):
        T = S.ChessChordalTorus(16, 2, 5)
        for root in (0, 1, 137):
            for bal in (False, True):
                t = S.build_broadcast_tree(T, root, balance=bal)
                self.assertTrue(S.check_tree(T, t))
                self.assertEqual(t.covered, T.N)
                self.assertEqual(t.height, 5)

    def test_tree_covers_component_with_faults(self):
        T = S.ChessChordalTorus(16, 2, 5)
        rng = random.Random(1)
        for _ in range(30):
            f = S.sample_exact_faults(T, rng, 20, 40)
            root = rng.choice([u for u in range(T.N) if f.alive_node(u)])
            comp = sum(1 for x in T.bfs(root, f) if x >= 0)
            t = S.build_broadcast_tree(T, root, f)
            self.assertTrue(S.check_tree(T, t, f))
            self.assertEqual(t.covered, comp)


class TestFaultTolerantRouting(unittest.TestCase):
    def test_guaranteed_delivery_and_unreachability_detection(self):
        T = S.ChessChordalTorus(16, 2, 5)
        rng = random.Random(7)
        for hc in (5, 7):
            for _ in range(25):
                f = S.sample_exact_faults(T, rng, rng.choice([0, 10, 40]), rng.choice([0, 60, 150]))
                forest = S.build_escape_forest(T, f)
                alive = [u for u in range(T.N) if f.alive_node(u)]
                for _ in range(40):
                    s, d = rng.sample(alive, 2)
                    ref = T.bfs(s, f)
                    r = S.route_one_to_one_ft(T, s, d, f, forest, hc)
                    if ref[d] < 0:
                        self.assertFalse(r.delivered)
                    else:
                        self.assertTrue(r.delivered, (hc, s, d))
                        self.assertEqual(r.path[0], s)
                        self.assertEqual(r.path[-1], d)
                        for u, v in zip(r.path, r.path[1:]):
                            self.assertIn(v, T.nbrs[u])
                            self.assertTrue(f.alive_link(u, v))

    def test_node_with_one_live_link_is_reached(self):
        T = S.ChessChordalTorus(16, 2, 5)
        dst = 100
        keep = T.nbrs[dst][0]
        f = S.Faults(set(), {(min(dst, v), max(dst, v)) for v in T.nbrs[dst] if v != keep})
        forest = S.build_escape_forest(T, f)
        for s in range(T.N):
            if s != dst:
                self.assertTrue(S.route_one_to_one_ft(T, s, dst, f, forest, 7).delivered)

    def test_isolated_node_detected(self):
        T = S.ChessChordalTorus(16, 2, 5)
        dst = 100
        f = S.Faults(set(), {(min(dst, v), max(dst, v)) for v in T.nbrs[dst]})
        forest = S.build_escape_forest(T, f)
        r = S.route_one_to_one_ft(T, 0, dst, f, forest, 7)
        self.assertFalse(r.delivered)

    def test_fault_free_never_uses_escape(self):
        T = S.ChessChordalTorus(16, 2, 5)
        forest = S.build_escape_forest(T)
        rng = random.Random(3)
        for _ in range(2000):
            s, d = rng.sample(range(T.N), 2)
            r = S.route_one_to_one_ft(T, s, d, S.NO_FAULTS, forest, 5)
            self.assertFalse(r.escaped)
            self.assertEqual(r.hops, T.dist(s, d))


class TestDeadlockFreedom(unittest.TestCase):
    def test_cdg_acyclic_with_faults_and_cyclic_with_one_vc(self):
        T = S.ChessChordalTorus(8, 1, 4)
        rng = random.Random(5)
        f = S.sample_exact_faults(T, rng, 4, 12)
        forest = S.build_escape_forest(T, f)
        alive = [u for u in range(T.N) if f.alive_node(u)]
        deps, deps1 = {}, {}
        for s in alive:
            for d in alive:
                if s != d and forest.same_component(s, d):
                    r = S.route_one_to_one_ft(T, s, d, f, forest, 6)
                    S.route_cdg_add(deps, r.path, r.vcs)
                    S.route_cdg_add(deps1, r.path, r.vcs, single_vc=True)
        self.assertEqual(S.cdg_cycle_nodes(deps), 0)
        self.assertGreater(S.cdg_cycle_nodes(deps1), 0)
        self.assertEqual(S.cdg_cycle_nodes(S.tree_cdg(T, forest)), 0)

    def test_multicast_cdg_acyclic(self):
        T = S.ChessChordalTorus(8, 1, 4)
        rng = random.Random(6)
        f = S.sample_exact_faults(T, rng, 6, 10)
        alive = [u for u in range(T.N) if f.alive_node(u)]
        deps, _ = S.multicast_cdg(T, f, alive)
        self.assertEqual(S.cdg_cycle_nodes(deps), 0)


class TestCollectives(unittest.TestCase):
    def test_broadcast_time_formula(self):
        T = S.ChessChordalTorus(16, 2, 5)
        cfg = S.SimConfig(flits=17280)
        res = S.simulate_collective(T, cfg, [0])
        self.assertEqual(res["cycles"], cfg.t_init + 5 * (cfg.t_router + cfg.t_link) + 17280)


class TestModels(unittest.TestCase):
    def test_p3_channels_follow_yaml_scales(self):
        self.assertEqual({k: S.p3_channels(k) for k in "nsmlx"}, {"n": 128, "s": 256, "m": 512, "l": 512, "x": 768})

    def test_flits(self):
        self.assertEqual(S.model_flits("l"), (17280, 6144))
        self.assertEqual(S.model_flits("x"), (25920, 9216))
        self.assertEqual(S.model_flits("n"), (4320, 1536))

    def test_smallest_model_is_n(self):
        self.assertEqual(min(S.YOLO26_MODELS.values(), key=lambda m: m.gflops_640).key, "n")

    def test_default_simconfig_follows_active_model(self):
        old = S.ACTIVE_MODEL
        try:
            S.set_model("s")
            self.assertEqual(S.SimConfig().flits, 8640)
        finally:
            S.set_model(old)

    def test_flops_per_node_uses_model_specific_value(self):
        self.assertAlmostEqual(S.node_flops_per_frame("l") / S.node_flops_per_frame("x"), 94.2 / 210.1, places=6)


class TestFlowBalancing(unittest.TestCase):
    def test_trees_valid_all_roots_shht(self):
        T = S.ChessChordalTorus(16, 2, 5)
        for r in range(T.N):
            t = S.build_broadcast_tree(T, r, balance="flow")
            self.assertTrue(S.check_tree(T, t))
            self.assertEqual(t.covered, T.N)
            self.assertEqual(t.height, 5)
            self.assertEqual(t.repair_rounds, 0)

    def test_max_load_reaches_ceiling_of_average(self):
        cfg = S.SimConfig(flits=1000)
        for T, expect in ((S.ChessChordalTorus(16, 2, 5), 43), (S.Torus((8, 8, 4)), 43),
                          (S.Torus((16, 16)), 64), (S.Hypercube(8), 32)):
            res = S.simulate_collective(T, cfg, list(range(T.N)), balance="flow", sched="critical")
            self.assertEqual(res["max_link_load"], expect, T.name)
            self.assertLessEqual(res["cycles"], cfg.t_init + expect * 1000 + (cfg.t_router + cfg.t_link) + 1000)

    def test_critical_not_worse_than_fifo_and_formula_for_broadcast(self):
        T = S.ChessChordalTorus(16, 2, 5)
        cfg = S.SimConfig(flits=2000)
        a = S.simulate_collective(T, cfg, [0], sched="critical")["cycles"]
        self.assertEqual(a, cfg.t_init + 5 * (cfg.t_router + cfg.t_link) + 2000)
        f = S.simulate_collective(T, cfg, list(range(T.N)), balance="flow")["cycles"]
        c = S.simulate_collective(T, cfg, list(range(T.N)), balance="flow", sched="critical")["cycles"]
        self.assertLessEqual(c, f)

    def test_flow_with_faults_covers_component(self):
        T = S.ChessChordalTorus(16, 2, 5)
        rng = random.Random(21)
        for _ in range(15):
            f = S.sample_exact_faults(T, rng, 15, 40)
            root = rng.choice([u for u in range(T.N) if f.alive_node(u)])
            comp = sum(1 for x in T.bfs(root, f) if x >= 0)
            t = S.build_broadcast_tree(T, root, f, balance="flow")
            self.assertTrue(S.check_tree(T, t, f))
            self.assertEqual(t.covered, comp)

    def test_unsupported_topology_falls_back(self):
        T = S.Mesh2D(8)
        self.assertIsNone(S.flow_balance_table(T))
        t = S.build_broadcast_tree(T, 5, balance="flow")
        self.assertTrue(S.check_tree(T, t))
        self.assertEqual(t.covered, T.N)

    def test_multicast_cdg_acyclic_with_flow_trees(self):
        T = S.ChessChordalTorus(8, 1, 4)
        rng = random.Random(8)
        f = S.sample_exact_faults(T, rng, 5, 10)
        alive = [u for u in range(T.N) if f.alive_node(u)]
        deps, _ = S.multicast_cdg(T, f, alive, balance="flow")
        self.assertEqual(S.cdg_cycle_nodes(deps), 0)

    def test_ideal_estimates(self):
        cfg = S.SimConfig(flits=1000)
        H = S.Hypercube(4)
        self.assertEqual(S.ideal_allgather_estimate(H, cfg), 4 * 13 + -(-15 * 1000 // 4))
        self.assertIsNone(S.ideal_allgather_estimate(S.ChessChordalTorus(8, 1, 4), cfg))


class TestTrafficWithFaults(unittest.TestCase):
    def test_simulate_traffic_delivers_all_reachable(self):
        T = S.ChessChordalTorus(8, 1, 4)
        rng = random.Random(4)
        f = S.sample_exact_faults(T, rng, 6, 20)
        alive = [u for u in range(T.N) if f.alive_node(u)]
        msgs = [(s, d, 0) for s in alive[:20] for d in alive[-20:] if s != d]
        lat = S.simulate_traffic(T, S.SimConfig(flits=100), msgs, f)
        for (s, d, _), x in zip(msgs, lat):
            self.assertEqual(x is not None, T.bfs(s, f)[d] >= 0)


class TestV5Halo(unittest.TestCase):
    def test_side_sizes_and_no_wrap(self):
        self.assertEqual((S.side_flits(0, "x"), S.side_flits(1, "x")), (1728, 2880))
        T = S.ChessChordalTorus(16, 2, 5)
        messages, paths = S.halo_plan(T, "x")
        self.assertEqual(len(messages), 960)
        self.assertEqual(sum(m[3] for m in messages), 480 * (1728 + 2880))
        for (u, v, t0, m), p in zip(messages, paths):
            self.assertEqual(p, [u, v])
            xu, yu = divmod(u, 16); xv, yv = divmod(v, 16)
            self.assertEqual(abs(xu - xv) + abs(yu - yv), 1)

    def test_halo_time_and_unsupported(self):
        cfg = S.SimConfig()
        r = S.simulate_halo(S.ChessChordalTorus(16, 2, 5), cfg, "x")
        self.assertEqual(r["cycles"], cfg.t_init + cfg.t_router + cfg.t_link + 2880)
        self.assertIsNone(S.simulate_halo(S.Hypercube(8), cfg, "x"))
        self.assertIsNone(S.simulate_halo(S.Torus((8, 8, 4)), cfg, "x"))

    def test_paths_are_validated(self):
        T = S.ChessChordalTorus(8, 1, 4); cfg = S.SimConfig()
        with self.assertRaises(ValueError):
            S.simulate_paths(T, cfg, [(0, 1, 0, 10)], [[0, 63, 1]])
        d = T.nbrs[0][0]
        self.assertEqual(S.simulate_paths(T, cfg, [(0, d, 0, 10), (0, d, 0, 30)], [[0, d], [0, d]]), [23, 53])


class TestV5Collective(unittest.TestCase):
    def test_empty_duplicates_and_dead_root(self):
        T = S.ChessChordalTorus(4, 1, 2); cfg = S.SimConfig(flits=10)
        self.assertEqual(S.simulate_collective(T, cfg, [])["cycles"], 0)
        with self.assertRaises(ValueError):
            S.simulate_collective(T, cfg, [0, 0])
        with self.assertRaises(ValueError):
            S.simulate_collective(T, cfg, [1], S.Faults({1}))

    def test_disconnected_has_no_bound(self):
        T = S.ChessChordalTorus(4, 1, 2); d = 5
        f = S.Faults(links={tuple(sorted((d, v))) for v in T.nbrs[d]})
        r = S.simulate_collective(T, S.SimConfig(flits=10), [u for u in range(T.N) if u != d], f)
        self.assertFalse(r["complete"]); self.assertIsNone(r["lb_port_cycles"])

    def test_time_never_below_bound(self):
        for T in (S.ChessChordalTorus(8, 1, 4), S.Torus((4, 4)), S.Hypercube(4)):
            for bal, sc in ((False, "fifo"), (True, "fifo"), ("flow", "fifo"), ("flow", "critical")):
                r = S.simulate_collective(T, S.SimConfig(flits=100), range(T.N), balance=bal, sched=sc)
                self.assertTrue(r["complete"]); self.assertGreaterEqual(r["cycles"], r["lb_port_cycles"])


class TestV5DeadlockAndFaults(unittest.TestCase):
    def test_shared_pool_cdg(self):
        T = S.ChessChordalTorus(8, 1, 4); rng = random.Random(9)
        for fn, fl in ((0, 0), (4, 10), (10, 25)):
            f = S.sample_exact_faults(T, rng, fn, fl)
            r = S.combined_cdg(T, f, T.metrics()[0] + 2)
            self.assertEqual(r["residual"], 0)
            self.assertEqual(r["vcs_required"], max(T.metrics()[0] + 3, r["hmax"]))

    def test_hc_boundaries(self):
        T = S.ChessChordalTorus(8, 1, 4); fo = S.build_escape_forest(T)
        self.assertTrue(S.route_one_to_one_ft(T, 0, 30, S.NO_FAULTS, fo, 0).delivered)
        with self.assertRaises(ValueError):
            S.route_one_to_one_ft(T, 0, 30, S.NO_FAULTS, fo, -1)

    def test_all_pairs_delivered_in_component(self):
        T = S.ChessChordalTorus(8, 1, 4); rng = random.Random(42)
        for _ in range(20):
            f = S.sample_exact_faults(T, rng, 8, 25); fo = S.build_escape_forest(T, f)
            alive = list(fo.comp)
            for _ in range(40):
                a, b = rng.sample(alive, 2)
                r = S.route_one_to_one_ft(T, a, b, f, fo, 6)
                self.assertEqual(r.delivered, T.bfs(a, f)[b] >= 0)
                self.assertEqual(len(r.vcs), r.hops)

    def test_dynamic_experiments_removed(self):
        for name in ("simulate_dynamic", "reconfig_cost", "exp_dynamic", "exp_reconfig"):
            self.assertFalse(hasattr(S, name), name)


class TestV5Topology(unittest.TestCase):
    def test_analytic_baselines_match_bfs(self):
        for dims in ((4, 4), (3, 4, 5)):
            T = S.Torus(dims); D, avg = T.metrics(); m = S.torus_metrics(dims)
            self.assertEqual(D, m["D"]); self.assertAlmostEqual(avg, m["avg"], places=9)
        H = S.Hypercube(5); D, avg = H.metrics(); m = S.hypercube_metrics(5)
        self.assertEqual(D, m["D"]); self.assertAlmostEqual(avg, m["avg"], places=9)
        self.assertEqual(S.best_torus3(256), (8, 8, 4))

    def test_shht_table_1_1(self):
        for n, (D, avg) in S.TABLE_1_1.items():
            T = S.ChessChordalTorus(n, *S.chords_for(n)); d0, d1 = T.bfs(0), T.bfs(1)
            self.assertEqual(max(max(d0), max(d1)), D)
            self.assertAlmostEqual((sum(d0) + sum(d1)) / (2 * (T.N - 1)), avg, delta=6e-4)

    def test_embedding_table_1_4(self):
        expected = {(16, "row"): (3, 1.833, 2, 5), (16, "gray"): (3, 1.667, 1, 5), (64, "row"): (4, 2.083, 3, 5),
                    (64, "gray"): (3, 1.667, 1, 4), (256, "row"): (7, 2.792, 4, 6), (256, "gray"): (5, 2.167, 1, 5)}
        for (N, method), (dm, da, dg, cg) in expected.items():
            n = int(N ** 0.5)
            r = S.embed_hypercube(S.ChessChordalTorus(n, *S.chords_for(n)), method)
            self.assertEqual((r["dilation_max"], r["grid_dilation_max"], r["congestion_max"], r["expansion"]), (dm, dg, cg, 1.0))
            self.assertAlmostEqual(r["dilation_mean"], da, places=3)

    def test_memory_helpers(self):
        T = S.ChessChordalTorus(16, 2, 5); cfg = S.SimConfig(flits=17280)
        self.assertEqual(S.vc_buffer_cost(cfg, 6)["node_B"], 4976640)
        self.assertEqual(S.directory_bytes(T, S.build_escape_forest(T))["directory_B"], 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
