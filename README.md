# Workforce Shift Scheduling – Operations Research (MILP)

Optimises the weekly schedule of a customer-service team with **Mixed-Integer Linear Programming**, and compares it with a simple greedy rota.

> Data (demand, availabilities, preferences) is simulated for portfolio purposes.

## Problem
- 17 agents (6 seniors), 7 days × 3 shifts (morning, afternoon, night), weekly demand between 2 and 6 agents per shift.
- **Decision variables:** `x[e,d,s]` = 1 if agent *e* works shift *s* on day *d*; `a[d,s]` = temporary agency workers.
- **Objective:** minimise staffing cost (night premium +30%, agency shift = 2.2× an internal shift) + a penalty when an agent is not on their preferred shift.
- **Constraints:** demand covered · max 1 shift per day · 3 to 5 shifts per week (contract) · 11 h rest (no night → morning) · at least one senior per shift · unavailability respected.
- **Solver:** HiGHS via `scipy.optimize.milp` (open source, no licence).

## Results
| | Greedy rota | MILP optimum |
|---|---|---|
| Weekly cost | 9,698 EUR | **8,180 EUR (-15.7%)** |
| Agency shifts | 6 | **0** |
| Shifts on preferred slot | 23% | **73%** |

The optimised schedule is exported to `outputs/schedule.csv`.

## Run it
```bash
pip install -r requirements.txt
python src/optimize.py
```

## Tech stack
Python · Operations research · Linear / integer programming (MILP) · SciPy (HiGHS) · NumPy · Pandas
