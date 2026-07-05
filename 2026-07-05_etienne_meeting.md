# Meeting With Etienne: Negative-Drag Work Extraction

Prepared by: Yunsik Choe  
Date: 2026-07-05  
Project: negative-drag work extraction from an active bath

## One-Sentence Thesis

Negative drag from an active bath should be treated not just as an anomalous mobility, but as a dynamical resource that can perform mechanical work on a controlled spring without measuring the active noise or applying feedback.

## What I Want To Get From The Meeting

1. Check whether the reduced Langevin picture is a physically clean way to present the RTP simulations.
2. Decide how to frame the unstable branch: as a limit-cycle engine regime rather than a failure of the fixed-point force curve.
3. Clarify whether the finite-`N` correction is generically detrimental, or only detrimental in the stable cubic branch we have analyzed.
4. Decide whether the paper should foreground constant-speed pulling, a periodic spring-center engine, or both.
5. Get advice on thermodynamic accounting: extracted mechanical work versus the energetic cost of maintaining the active bath.

## Current Model

The microscopic setup is a passive tracer coupled to non-interacting 1D run-and-tumble particles and attached to a controlled spring center `lambda(t)`.

The coarse-grained tracer model is

```text
M Xddot = F(Xdot) - k [X - lambda(t)] + eta_N(t),
```

where `F(v)` is the active-bath force-velocity curve. Near the negative-drag region we use the odd cubic normal form

```text
F(v) ~= a v - b v^3,  a > 0, b > 0.
```

The operational work convention is

```text
P_ext = k [X - lambda(t)] lambda_dot(t).
```

Positive `P_ext` means the tracer pushes the operator along the imposed motion.

## Constant-Speed Pulling

For `lambda(t)=v0 t`, define `y=X-lambda` and `u=Xdot-v0`. Then

```text
M udot = F(v0 + u) - k y + noise.
```

The fixed point is

```text
y* = F(v0)/k,  u* = 0.
```

Three regimes:

1. `F'(v0)>0`: fixed point is unstable; mean work depends on the limit cycle.
2. `F'(v0)<0` and `F(v0)>0`: stable fixed point; work extraction.
3. `F'(v0)<0` and `F(v0)<0`: stable fixed point; work injection.

For `F(v)=v-v^3`, the stable extraction branch is `1/sqrt(3)<v0<1`.

## Finite-N Correction

In the stable branch, expanding the noisy equation around the stable fixed point gives

```text
k <y> = F(v0) + F''(v0) B(v0) / [2 N M (-F'(v0))] + O(N^-2).
```

For the cubic normal form and positive `v0` in the stable extraction branch, `F''(v0)<0`, so the leading finite-`N` correction reduces extracted work.

Notebook result at `v0=0.75`, `M=1`:

- Mean-field force: `0.3281`.
- Simulated finite-`N` force approaches the mean-field value from below.
- The `1/N` prediction matches the Langevin simulation across `N=50..2000`.

See:

- `negative_drag_langevin_engine.ipynb`, section 11.
- `data/langevin_engine_summary/finite_N_correction.csv`.
- `image/langevin_engine_summary/finite_N_correction.svg`.

## Unstable Branch And Large-M Limit

The unstable branch cannot be interpreted by the sign of `F(v0)` alone. The large-`M` note gives a useful averaged picture:

```text
T_osc ~= 2 pi sqrt(M/k).
```

For a cubic normal form, the limit-cycle center shift is

```text
Delta F_center = (1/4) F''(v0) A_LC^2 = 2 v0 F'_cub(v0).
```

For positive `v0` in the unstable branch this correction shifts the mean spring force downward. This may explain why large-mass RTP simulations can show weaker or negative extraction even where the naive mean-field force is positive.

Question for Etienne:

Is this the right physical language, or should we avoid calling the unstable branch an "engine regime" until the full work distribution and bath entropy budget are measured?

## Periodic Engine

The periodic engine uses a back-and-forth spring-center protocol. The key observables are not only mean work, but also finite-duration reliability:

```text
mean W per cycle,
power = W/T,
P(W>0).
```

Compact Langevin scan:

- Triangular protocol.
- Amplitude `A=1.5`.
- Finite bath noise `N=250`.
- Duration scan over `10, 20, 40, 80` measured cycles.
- Positive-power window centered near engine speed `4A/T ~= 1.75`.

Best robust row in the compact summary:

```text
speed = 1.75
T = 3.429
duration = 20 cycles
power = +0.1482 +/- 0.0064
P(power > 0) = 1.000
```

See:

- `data/langevin_engine_summary/periodic_speed_duration.csv`.
- `image/langevin_engine_summary/periodic_speed_duration.svg`.

## Main Questions For Etienne

1. Does the reduced Langevin equation need an explicit memory kernel to be credible, or is the fast-bath/large-`N` Markovian reduction acceptable as a first theory?
2. Is the finite-`N` correction formula the right object to compare with RTP simulations, or should we measure a full velocity-dependent noise kernel `B(v)` first?
3. How should we present the active-bath energy budget? Should the headline be "mechanical work extracted from a maintained active bath" rather than simply "work extraction"?
4. In the unstable branch, should we emphasize the large-`M` perturbation theory, the limit-cycle geometry, or direct RTP evidence?
5. Is a periodic protocol necessary for publication framing, or can constant-speed pulling already count as a clean work-extraction demonstration?
6. Would an information-thermodynamic framing help, or would it distract from the no-feedback mechanism?

## Proposed Meeting Flow

### First 10 Minutes: Physical Message

Show the setup and thesis:

- Active bath gives nonlinear `F(v)`.
- Negative drag destabilizes ordinary friction.
- Controlled spring can extract mechanical work without measuring active noise.

### Next 15 Minutes: Theory

Walk through:

- Constant-speed fixed point.
- Three regimes using `F(v0)` and `F'(v0)`.
- Stable-branch finite-`N` correction.
- Large-`M` limit-cycle center shift.

### Next 15 Minutes: Numerics

Show:

- Langevin finite-`N` verification.
- Periodic speed-duration heatmap.
- RTP pipeline status via `run_kv.py`, `run_saw.py`, and `kv_visualization.py`.

### Final 10 Minutes: Decisions

Ask Etienne to choose the most convincing paper path:

- constant-speed pulling first,
- periodic engine first,
- or both, with constant-speed as theory and cycle as engine demonstration.

## Possible Paper Framing

Title direction:

```text
Work extraction from negative drag in an active bath
```

Core claim:

```text
An active bath with a negative-drag force-velocity curve can drive a passive tracer against an externally controlled spring. Work extraction is controlled by fixed-point stability, limit-cycle dynamics, and finite-bath fluctuations.
```

Important caveat:

```text
The extracted work is mechanical work delivered by a maintained active bath. It is not a violation of the second law and should be reported together with the bath-maintenance cost or an explicit statement that the bath is treated as a nonequilibrium reservoir.
```

## Concrete Next Steps After Meeting

1. Replace the cubic `F(v)` in section 11 with measured RTP `F(v)`.
2. Measure `B(v)` from fixed-speed RTP trajectories.
3. Re-run finite-`N` correction tests using measured `F(v)` and `B(v)`.
4. Add passive/control curves to the periodic protocol plots.
5. Decide whether to write the first draft around constant-speed pulling or the periodic engine.
