# High-rate inertial preintegration reference

## Purpose

The calibration playground now includes an independent high-rate inertial truth
oracle for intervals where component-wise mean angular rate and specific force
are not sufficient. The runtime `imu_error_model` package still owns sensor
corruption, not navigation mechanization.

The oracle is used to answer three questions:

1. Does a candidate sample schedule preserve coning and sculling motion?
2. Does a downstream correction algorithm recover the major-interval inertial
   increment from high-rate raw increments?

3. At what tested minor-sample rate does a configured deterministic error target
   become achievable?

## Reference semantics

For one minor interval with constant body angular rate `omega`, specific force
`f`, and duration `dt`, the exact relative attitude is

```text
R_next = R @ Exp([omega]x dt)
```

and the velocity increment expressed in the major-interval starting body frame
is

```text
delta_v += R @ integral_0^dt Exp([omega]x tau) d tau @ f.
```

The integral is evaluated analytically. A continuous input is approximated by a
very fine midpoint partition and then integrated with these exact
piecewise-constant updates. Step-size convergence is therefore explicit and
measurable.

The playground also implements a classical small-angle recursive
coning/sculling correction for comparison. It is an approximation, not the
truth oracle. The recursive form follows the coning/sculling accumulation
structure summarized in the open-access uncertainty treatment at
https://pmc.ncbi.nlm.nih.gov/articles/PMC8708497/ .

## Canonical excitation cases

### Coning

A circular two-axis rate input

```text
omega(t) = A [cos(Omega t), sin(Omega t), 0]
```

has zero component-wise mean over one cycle, but rotations do not commute. The
major-interval attitude increment is therefore nonzero. A single interval mean
rate loses the geometric rotation entirely.

### Sculling

The canonical orthogonal excitation

```text
omega_x(t) = B Omega cos(Omega t)
f_y(t)     = C sin(Omega t)
```

has zero raw angle and raw velocity sums over one cycle, while the coupled
rotation/specific-force integral produces a nonzero third-axis velocity
increment. This makes sculling loss directly observable.

## Calibration boundary

Calibration should still operate on high-rate sensor samples before navigation
preintegration. A full intrinsic matrix, bias, filter, or timing correction is
not generally equivalent to applying one correction to a major-interval
rotation vector after coning/sculling has already occurred.

The intended verification route is therefore:

```text
continuous truth
  -> high-rate ideal minor increments
  -> IMU error model / acquisition model
  -> calibration correction on minor samples
  -> coning/sculling preintegration
  -> compare with independent high-rate truth oracle.
```

The current tranche establishes the oracle and convergence machinery. Applying
nonlinear sensor effects inside the analog aperture and propagating stochastic
preintegration covariance remain later backlog items.
