# Mars Aerocapture Model

A 3-DOF point-mass simulation of aerocapture at Mars for a blunt sphere-cone
capsule, developed for AERO4800 (Space Engineering) at the University of
Queensland. Aerocapture trades propellant for atmospheric drag: by flying a
carefully chosen hyperbolic pass through the upper Martian atmosphere, the
vehicle sheds enough orbital energy to be captured into a bound orbit at the
cost of significant aeroheating. This model quantifies that trade end to end.

## What it does

- Propagates the entry arc with a 3-DOF spherical equation-of-motion solver
  and a selectable (constant or inverse-square) gravity model.
- Computes aeroheating along the trajectory: West-Brandis convective and
  Tauber-Sutton radiative stagnation-point correlations, with a
  radiative-equilibrium wall-temperature estimate.
- Plans the post-aerocapture capture: a two-burn retarget (raise periapsis,
  then circularise) to a parking orbit, and reports total Delta-v.
- Runs a parametric sweep over entry speed, flight-path angle and lift-to-drag
  to trade Delta-v, peak/total heating, g-loads and minimum altitude.

## Files

| File | Description |
|------|-------------|
| `model_mars.py` | Mars environment: exponential atmosphere, gravity model, gravitational parameter. |
| `main.py` | Main aerocapture simulation, heating models and two-burn capture plan. |

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.9+.

## Running

```
python main.py
```

Figures and summary tables are generated directly from the run.

## Validation

The solver was checked against three benchmarks: ballistic entry scaling, a
Hayabusa stagnation-heating snapshot, and a published NASA aerocapture
corridor.

## Author

Jason-Dylan Hassard - University of Queensland
