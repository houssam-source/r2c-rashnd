"""Coefficient-dependent dynamic programming over the cluster graph.

Value iteration for the bridging MDP: states = clusters, actions = bridge
along a corridor edge, stage cost = time-to-bridge penalized by the believed
jammer power along the corridor. The cost is parameterized by a coefficient
vector theta = (gamma, kappa_age): gamma scales the belief penalty, and
kappa_age lets the penalty GROW with belief staleness, so the optimal POLICY
changes as the coefficients change. That is the point of the DP formulation
over a plain shortest-path search: the whole value function is re-solved
under the current coefficients, and coefficient sweeps expose policy flips.

For fixed coefficients the DP optimum equals Dijkstra/A*; the DP pays off
when coefficients are state/time-dependent (e.g. stale beliefs must be
penalized more) or when costs become stochastic.
"""
import numpy as np


def stage_cost(length, p_belief_norm, v_max, gamma, kappa_age=0.0,
               age_steps=0):
    """time-to-bridge x (1 + gamma_eff * believed-jammer-power), where
    gamma_eff = gamma * (1 + kappa_age * age_steps / 100)."""
    gamma_eff = gamma * (1.0 + kappa_age * min(age_steps, 200) / 100.0)
    return length / v_max * (1.0 + gamma_eff * p_belief_norm)


def value_iteration(nodes, edges, edge_cost, goal, discount=1.0, iters=200,
                    tol=1e-9):
    """nodes: iterable; edges: {s: [(s2, edge_key), ...]}; edge_cost(s, s2,
    key) -> float; goal(s) -> bool. Returns (V, policy)."""
    V = {s: 0.0 if goal(s) else np.inf for s in nodes}
    policy = {}
    for _ in range(iters):
        delta = 0.0
        newV, newP = dict(V), dict(policy)
        for s in nodes:
            if goal(s):
                continue
            best, best_a = np.inf, None
            for s2, key in edges.get(s, ()):
                c = edge_cost(s, s2, key)
                if not np.isfinite(c):
                    continue
                q = c + discount * V[s2]
                if q < best - 1e-12:
                    best, best_a = q, (s2, key)
            if best_a is not None:
                newV[s] = best
                newP[s] = best_a
                delta = max(delta, abs(newV[s] - V[s]))
        V, policy = newV, newP
        if delta < tol:
            break
    return V, policy


def extract_path(policy, start):
    path, seen = [start], {start}
    s = start
    while s in policy:
        s2, _key = policy[s]
        if s2 in seen:
            break
        path.append(s2)
        seen.add(s2)
        s = s2
    return path


def dp_plan(origin, graph, belief_power, norm_power, v_max, gamma,
            kappa_age=0.0, age_of=None, goal=None):
    """Full plan under coefficient vector (gamma, kappa_age).

    belief_power(k, m): believed jammer power (mW) on corridor k->m.
    age_of(k): staleness of the belief used for k's outgoing corridors.
    Returns (path_edges, value_function) or (None, V).
    """
    if goal is None:
        goal = graph.adjacent_to_largest
    edges = {k: [(m, length) for m, length in graph.edges.get(k, ())] for k in
             graph.parts}
    nodes = list(graph.parts)

    def cost(k, m, length):
        age = age_of(k) if age_of else 0
        return stage_cost(length, norm_power(belief_power(k, m)),
                          v_max, gamma, kappa_age, age)

    V, policy = value_iteration(nodes, edges, cost, goal)
    if origin not in policy:
        return None, V
    path = extract_path(policy, origin)
    edges_path = list(zip(path[:-1], path[1:]))
    return edges_path, V


def coefficient_sweep(origin, graph, belief_power, norm_power, v_max,
                      gammas, kappas):
    """Show that the optimal plan CHANGES with the coefficients.
    Returns { (gamma, kappa): path }."""
    out = {}
    for g in gammas:
        for k in kappas:
            path, _V = dp_plan(origin, graph, belief_power, norm_power, v_max,
                               gamma=g, kappa_age=k)
            out[(g, k)] = path
    return out
