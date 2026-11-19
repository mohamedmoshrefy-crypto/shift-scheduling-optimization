"""Workforce shift scheduling with Mixed-Integer Linear Programming (operations research).

Problem: a customer-service centre must cover a weekly demand (7 days x 3 shifts) with 20 employees.
Decision variables:
  x[e,d,s] = 1 if employee e works shift s on day d        (binary)
  a[d,s]   = number of temporary agency workers hired       (integer)
Objective: minimise the total weekly staffing cost.
Constraints:
  - demand of every shift is covered (employees + agency workers)
  - at most 1 shift per employee per day
  - each employee works between 3 and 5 shifts per week (contract)
  - no "night -> morning" sequence (11 h rest rule)
  - every shift has at least one senior agent
  - employees respect their declared unavailability
Soft goal: assign people to their preferred shift (penalty in the objective).
Solver: HiGHS through scipy.optimize.milp (no licence needed).
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["Morning", "Afternoon", "Night"]
E, D, S = 17, len(DAYS), len(SHIFTS)
demand = np.array([[5, 5, 2], [5, 4, 2], [4, 4, 2], [5, 4, 2], [6, 5, 3], [4, 4, 3], [3, 3, 2]])  # people needed
cost_shift = np.array([1.0, 1.0, 1.3])      # night premium (x base shift cost)
SHIFT_COST, AGENCY_COST = 100, 220           # EUR per shift
senior = np.arange(E) < 6                    # 6 senior agents: every shift needs at least 1 senior
pref = rng.integers(0, S, E)                 # preferred shift of each employee
PREF_PENALTY = 15                            # EUR-equivalent penalty when not on the preferred shift
unavailable = {(e, d) for e in range(E) for d in range(D) if rng.random() < 0.12}

nx = E * D * S; na = D * S; n = nx + na
X = lambda e, d, s: (e * D + d) * S + s
A = lambda d, s: nx + d * S + s

c = np.zeros(n)
for e in range(E):
    for d in range(D):
        for s in range(S):
            c[X(e, d, s)] = SHIFT_COST * cost_shift[s] + PREF_PENALTY * (s != pref[e])
for d in range(D):
    for s in range(S):
        c[A(d, s)] = AGENCY_COST * cost_shift[s]
        # (agency workers are never senior)

rows, lo, hi = [], [], []
def add(coefs, l, h):
    r = np.zeros(n)
    for i, v in coefs: r[i] = v
    rows.append(r); lo.append(l); hi.append(h)

for d in range(D):                                   # coverage
    for s in range(S):
        add([(X(e, d, s), 1) for e in range(E)] + [(A(d, s), 1)], demand[d, s], np.inf)
for d in range(D):                                   # at least one senior on every shift
    for s in range(S):
        add([(X(e, d, s), 1) for e in range(E) if senior[e]], 1, np.inf)
for e in range(E):
    for d in range(D):                               # one shift per day
        add([(X(e, d, s), 1) for s in range(S)], 0, 1)
    add([(X(e, d, s), 1) for d in range(D) for s in range(S)], 3, 5)   # weekly contract
    for d in range(D - 1):                           # rest rule: no night followed by morning
        add([(X(e, d, 2), 1), (X(e, d + 1, 0), 1)], 0, 1)

ub = np.full(n, 1.0); ub[nx:] = np.inf
for e, d in unavailable:
    for s in range(S): ub[X(e, d, s)] = 0

res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(n), bounds=Bounds(0, ub))
assert res.success, res.message
x = res.x[:nx].round().reshape(E, D, S); a = res.x[nx:].round().reshape(D, S)

# ---- Baseline: greedy rota (least-loaded available employee first, then agency) ----
base_x = np.zeros((E, D, S)); load = np.zeros(E)
for d in range(D):
    for s in range(S):
        for e in np.argsort(load):
            if base_x[e, d, s:].sum() + base_x[e, d, :s].sum() == 0 and (e, d) not in unavailable and load[e] < 5 \
               and not (s == 0 and d > 0 and base_x[e, d - 1, 2]) and base_x[:, d, s].sum() < demand[d, s]:
                base_x[e, d, s] = 1; load[e] += 1
base_a = np.maximum(demand - base_x.sum(0), 0)
# a shift without any senior must call a senior from the agency
for d in range(D):
    for s in range(S):
        if base_x[senior, d, s].sum() == 0: base_a[d, s] += 1
cost = lambda xx, aa: float((xx.sum(0) * cost_shift * SHIFT_COST).sum() + (aa * cost_shift * AGENCY_COST).sum())
pref_rate = lambda xx: sum(xx[e, :, pref[e]].sum() for e in range(E)) / xx.sum()

opt, base = cost(x, a), cost(base_x, base_a)
summary = pd.DataFrame({"Greedy rota": [f"{base:,.0f}", int(base_a.sum()), f"{pref_rate(base_x):.0%}"],
                        "MILP optimum": [f"{opt:,.0f}", int(a.sum()), f"{pref_rate(x):.0%}"]},
                       index=["Weekly cost (EUR)", "Agency shifts", "Shifts on preferred slot"])
print(f"Cost saving vs greedy rota: {(base - opt) / base:.1%}")
print(summary, "\n")

plan = pd.DataFrame([[", ".join(SHIFTS[s][0] for s in range(S) if x[e, d, s]) or "-" for d in range(D)] for e in range(E)],
                    index=[f"Emp {e + 1:02d}" for e in range(E)], columns=DAYS)
plan.to_csv(OUT / "schedule.csv"); summary.to_csv(OUT / "summary.csv")
print(plan.head(8))
print(f"\nAll {D * S} shifts covered: {bool(((x.sum(0) + a) >= demand).all())}")
