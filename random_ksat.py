import random, time, math
from collections import defaultdict
import pandas as pd

random.seed(0)

def gen_random_k_sat(k, n, m):
    clauses = []
    for _ in range(m):
        vars_in_clause = random.sample(range(n), k)
        clause = []
        for v in vars_in_clause:
            if random.random() < 0.5:
                clause.append(v+1)     # positive literal (index+1)
            else:
                clause.append(-(v+1))  # negative literal
        clauses.append(tuple(clause))
    return clauses

def eval_clause(clause, assignment):
    for lit in clause:
        v = abs(lit)-1
        val = assignment[v]
        if lit > 0 and val: return True
        if lit < 0 and (not val): return True
    return False

def satisfied_clauses(clauses, assignment):
    return sum(1 for c in clauses if eval_clause(c, assignment))

def build_occurrence_map(clauses, n):
    pos_occ = [[] for _ in range(n)]
    neg_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit)-1
            if lit > 0:
                pos_occ[v].append(ci)
            else:
                neg_occ[v].append(ci)
    return pos_occ, neg_occ

def flip_delta(clauses, assignment, v, pos_occ, neg_occ):
    before = 0
    after = 0
    for ci in pos_occ[v]:
        sat_before = eval_clause(clauses[ci], assignment)
        assignment[v] = not assignment[v]
        sat_after = eval_clause(clauses[ci], assignment)
        assignment[v] = not assignment[v]
        before += 1 if sat_before else 0
        after += 1 if sat_after else 0
    for ci in neg_occ[v]:
        sat_before = eval_clause(clauses[ci], assignment)
        assignment[v] = not assignment[v]
        sat_after = eval_clause(clauses[ci], assignment)
        assignment[v] = not assignment[v]
        before += 1 if sat_before else 0
        after += 1 if sat_after else 0
    return after - before

def hill_climbing(clauses, n, max_flips=2000, max_restarts=20, heuristic='sat_count', pos_occ=None, neg_occ=None):
    m = len(clauses)
    for restart in range(max_restarts):
        assignment = [random.choice([False, True]) for _ in range(n)]
        for flip in range(max_flips):
            best_vs = []
            best_score = -10**9
            for v in range(n):
                if heuristic == 'sat_count':
                    assignment[v] = not assignment[v]
                    score = satisfied_clauses(clauses, assignment)
                    assignment[v] = not assignment[v]
                else:
                    score = flip_delta(clauses, assignment, v, pos_occ, neg_occ)
                if score > best_score:
                    best_score = score
                    best_vs = [v]
                elif score == best_score:
                    best_vs.append(v)
            choice_v = random.choice(best_vs)
            assignment[choice_v] = not assignment[choice_v]
            if satisfied_clauses(clauses, assignment) == m:
                return True, assignment, flip+1
    return False, None, max_restarts*max_flips

def beam_search(clauses, n, beam_width=3, max_steps=500):
    m = len(clauses)
    beam = []
    for _ in range(beam_width):
        a = [random.choice([False, True]) for _ in range(n)]
        beam.append((satisfied_clauses(clauses, a), a))
    for step in range(max_steps):
        for score, a in beam:
            if score == m:
                return True, a, step
        candidates = []
        for score, a in beam:
            for v in range(n):
                na = a.copy()
                na[v] = not na[v]
                candidates.append((satisfied_clauses(clauses, na), na))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
    return False, None, max_steps

def vnd(clauses, n, pos_occ, neg_occ, max_iter=500):
    m = len(clauses)
    assignment = [random.choice([False, True]) for _ in range(n)]
    best_score = satisfied_clauses(clauses, assignment)
    for it in range(max_iter):
        # N1: best single flip
        best_v, best_delta = None, -10**9
        for v in range(n):
            d = flip_delta(clauses, assignment, v, pos_occ, neg_occ)
            if d > best_delta:
                best_delta = d
                best_v = v
        if best_delta > 0:
            assignment[best_v] = not assignment[best_v]
            best_score += best_delta
            if best_score == m:
                return True, assignment, it
            continue
        # N2: sample pairs
        pair_samples = [tuple(random.sample(range(n), 2)) for _ in range(min(50, n*(n-1)//2))]
        best_pair, best_pair_delta = None, -10**9
        for (i, j) in pair_samples:
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]
            s_after = satisfied_clauses(clauses, assignment)
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]
            delta = s_after - best_score
            if delta > best_pair_delta:
                best_pair_delta = delta; best_pair=(i,j)
        if best_pair_delta > 0:
            i,j = best_pair
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]
            best_score += best_pair_delta
            if best_score == m: return True, assignment, it
            continue
        # N3: sample triples
        triple_samples = [tuple(random.sample(range(n),3)) for _ in range(min(100, math.comb(n,3) if n>=3 else 0))]
        best_triple, best_triple_delta = None, -10**9
        for (i,j,k) in triple_samples:
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]; assignment[k] = not assignment[k]
            s_after = satisfied_clauses(clauses, assignment)
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]; assignment[k] = not assignment[k]
            delta = s_after - best_score
            if delta > best_triple_delta:
                best_triple_delta = delta; best_triple=(i,j,k)
        if best_triple_delta > 0:
            i,j,k = best_triple
            assignment[i] = not assignment[i]; assignment[j] = not assignment[j]; assignment[k] = not assignment[k]
            best_score += best_triple_delta
            if best_score == m: return True, assignment, it
            continue
        break
    return best_score == m, (assignment if best_score==m else None), -1

def run_single_setting(n,m,n_instances=10):
    results = []
    for inst in range(n_instances):
        clauses = gen_random_k_sat(3,n,m)
        pos_occ,neg_occ = build_occurrence_map(clauses,n)
        configs = [
            ('Hill','sat_count', lambda: hill_climbing(clauses, n, max_flips=500, max_restarts=5, heuristic='sat_count', pos_occ=pos_occ, neg_occ=neg_occ)),
            ('Hill','net_gain', lambda: hill_climbing(clauses, n, max_flips=500, max_restarts=5, heuristic='net_gain', pos_occ=pos_occ, neg_occ=neg_occ)),
            ('Beam-3','sat', lambda: beam_search(clauses, n, beam_width=3, max_steps=200)),
            ('Beam-4','sat', lambda: beam_search(clauses, n, beam_width=4, max_steps=200)),
            ('VND','sat', lambda: vnd(clauses, n, pos_occ, neg_occ, max_iter=200)),
        ]
        for name, heur, fn in configs:
            start=time.time()
            solved, assignment, info = fn()
            elapsed=time.time() - start
            results.append({'instance':inst, 'n':n, 'm':m, 'solver':name, 'heuristic':heur, 'solved':1 if solved else 0, 'time':elapsed})
    df = pd.DataFrame(results)
    summary = df.groupby(['solver','heuristic']).agg(
        penetrance=('solved','mean'),
        solved_count=('solved','sum'),
        instances=('solved','count'),
        avg_time=('time','mean')
    ).reset_index()
    return df, summary

if __name__ == "__main__":
    df_details, df_summary = run_single_setting(20, 80, n_instances=5)
    print(df_summary)
