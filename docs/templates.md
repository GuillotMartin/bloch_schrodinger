# Template cheat sheet

A **template** is a plain dictionary describing how one layer of a plot should look. It is the
one argument every plotting function in `bloch_schrodinger.plotting` takes, and it is just a
dict: there is no class to subclass and nothing to register. Build one literally, or start from
a preset and pass keyword overrides.

```python
from bloch_schrodinger.plotting import cmesh_tmpl, plot_eigenvector

plot_eigenvector([[abs(eigve)**2]], [[pot]], [[cmesh_tmpl("amplitude")]])
```

## The whole shape

Every key is optional. Nothing here is required, including `fkwargs`.

```python
{
    "fkwargs":   {...},          # passed straight to the matplotlib function
    "colorbar":  {
        "kwargs":     {...},     # passed to Figure.colorbar
        "cax":        {...},     # passed to AxesDivider.append_axes
        "ticks":      [...],     # tick positions   (only applied together with tickslabel)
        "tickslabel": [...],     # tick text
    },
    "axes":      {"xlabel": ..., "ylabel": ..., "aspect": ...},
    "clim":      (lo, hi),       # fixed colour limits
    "autoscale": bool,           # rescale as the sliders move
    "slider_start": "left",      # or "mid"
    "density":   int,            # quiver only: keep every n-th arrow
}
```

## Which function reads which key

Not every key is read by every function. A key the function ignores is silently ignored, so
this table is the thing to check first when a template seems to do nothing.

| Key | `create_map` | `create_line` | `create_quiver` | Default |
|---|---|---|---|---|
| `fkwargs` | yes | yes | yes | `{}` |
| `axes` | yes | yes | yes | nothing set |
| `slider_start` | yes | yes | yes | `"left"` |
| `autoscale` | yes | yes | yes | **off**, except `create_line` where it is **on** |
| `colorbar` | yes | no | yes | no colorbar |
| `clim` | yes | no | no | unset |
| `density` | no | no | yes | `1` |

`plot_eigenvector` and `dashboard` pass templates through to these three, so the same table
applies to a template you hand them.

## Where `fkwargs` ends up

`fkwargs` is forwarded unchanged, so anything the underlying matplotlib function accepts works,
and its documentation is the reference for what is allowed.

| You called | With | `fkwargs` goes to |
|---|---|---|
| `create_map` | `method="pcolormesh"` | [`Axes.pcolormesh`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html) |
| `create_map` | `method="contour"` | `Axes.contour` |
| `create_map` | `method="contourf"` | `Axes.contourf` |
| `create_line` | | `Axes.plot` |
| `create_quiver` | | `Axes.quiver` |

So `cmap`, `norm`, `vmin`, `vmax`, `rasterized`, `shading` for a mesh; `levels`, `colors`,
`linewidths`, `linestyles` for contours; `color`, `linestyle`, `linewidth` for a line;
`scale`, `width`, `pivot`, `headwidth` for arrows.

## Four things that will catch you out

**A `norm` should be a factory.** Write `lambda: colors.LogNorm()`. Two maps
must not share one `Normalize`, because autoscaling either one would rescale the other. Passing
an instance also works and is copied for you, but the factory form is what every preset uses
and is unambiguous.

```python
{"fkwargs": {"norm": lambda: colors.LogNorm()}}      # preferred
{"fkwargs": {"norm": colors.LogNorm()}}              # also fine, copied on use
```

**`clim` and `autoscale` contradict each other, and `autoscale` wins.** Autoscaling is applied
after the limits on every slider move, so a template with both honours `clim` on the first draw
and then loses it. Set `autoscale` to `False` whenever you set `clim`. Using
`cmesh_tmpl(..., clim=...)` does this for you.

```python
{"clim": (0, 6e-4), "autoscale": False}              # limits that actually hold
```

**`ticks` does nothing without `tickslabel`.** Tick positions are only applied when there are
labels to go with them. To move the ticks but keep numeric labels, give both.

**`autoscale` means different things per function.** For a map or a quiver it rescales the
*colour* range; for a line it rescales the *y* limits. `create_line` is also the one function
where it defaults to on, so pass `autoscale=False` to hold a line's y axis still.

## The `axes` block

Carries the labelling that would otherwise be set by hand after every call.

```python
{"axes": {"xlabel": r"$x$ ($\mu$m)", "ylabel": r"$y$ ($\mu$m)", "aspect": "equal"}}
```

Note `plot_eigenvector` sets `aspect="equal"` on two-axis cells by default, and steps aside if
your template names an aspect of its own.

## The presets

`cmesh_tmpl(name)` takes one of:

| Name | For | Colour scale |
|---|---|---|
| `"amplitude"` | a positive field | sequential, light to dark |
| `"amplitude - log"` | a positive field over decades | sequential log, dark to light |
| `"real"` | a signed field | diverging, blank at zero |
| `"real - log"` | a signed field over decades | symmetric log |
| `"phase"` | a wrapped angle | cyclic, fixed to plus or minus pi |
| `"potential"` | a trap | diverging, labelled V over E-sub-r |
| `"difference"` | a signed difference | diverging, signed number format |
| `"spin"` | a component on minus-one to one | diverging, sigma and pi tick labels |

An unrecognised name raises and lists the valid ones.

Two of these describe the same job. `"real - log"` is a continuous symmetric-log scale with a
fixed linear threshold of 1e-12, which suits data already scaled near that threshold; for a
field of order one decaying over several decades, almost the whole range falls in the log
regime and the picture saturates. `signed_log_tmpl` covers that case instead, with discrete
bands on an explicit ladder, and is usually the clearer choice for a decaying signed field such
as a Wannier function.

Overrides, so you never have to patch the returned dict:

```python
cmesh_tmpl("amplitude - log", clim=(1e-6, 1e0), cmap="magma", label=r"$|\psi|^2$")
cmesh_tmpl("spin", ticks=[-1, 1], tickslabel=[r"$\sigma_-$", r"$\sigma_+$"])
```

`cmap`, `label`, `clim`, `fmt`, `ticks`, `tickslabel`. Passing `clim` also turns `autoscale`
off. Called with a name alone, `cmesh_tmpl` returns exactly what it always has.

The other factories:

```python
contour_tmpl(3)                                  # or contour_tmpl([5], colors="k", linewidths=0.2)
quiver_tmpl(density=2)                           # plus any Axes.quiver kwarg
signed_log_tmpl(6)                               # a signed field spanning many decades
```

## Layers in `plot_eigenvector`

Each cell of the `templates` matrix styles up to three layers, in this order: the field, the
potential's contours, and the quiver. Give as few as you like and the rest take sensible
defaults.

```python
templates=[["amplitude"]]                                   # field only
templates=[[("amplitude", contour_tmpl(5))]]                # field + contours
templates=[[("amplitude", contour_tmpl(5), quiver_tmpl())]] # all three
templates=[[None]]                                          # no styling at all
templates=[[{"fkwargs": {"cmap": "magma"}}]]                # your own dict
```

## Two passes over one field

Some styles need the same data drawn twice, such as filled bands with their edges outlined.
That is two calls, so `signed_log_tmpl` hands back both templates at once. They share their
levels, which is what keeps the outlines on the band edges.

```python
filled, outline = signed_log_tmpl(6, label=r"$\mathrm{sign}(w)|w|^2$")

fig, ax = plt.subplots()
s1, up1, ax = create_map(fig, ax, [0, 1], w, "contourf", template=filled)
s2, up2, ax = create_map(fig, ax, [0, 1], w, "contour",  template=outline)
```

This is not a `plot_eigenvector` two-tuple. A two-tuple means (field, potential contour), so
passing the pair there styles the *potential's* contours with the outline. `plot_eigenvector`
raises and says so rather than drawing the wrong picture.

## Writing your own preset

A preset is a function returning a fresh dict. Return a new one on every call rather than
holding a module-level constant: callers are handed the dict itself and may patch it, and a
shared one would leak those edits into every later plot.

```python
def berry_tmpl(label=r"$\Omega_{n}$"):
    """A diverging map for Berry curvature, which is signed and centred on zero."""
    return {
        "fkwargs": {
            "cmap": cm.vik,                          # neutral midpoint, so zero reads as blank
            "rasterized": True,
            "norm": lambda: colors.CenteredNorm(),
        },
        "autoscale": True,
        "colorbar": {"kwargs": {"label": label, "format": "{x:+.1e}"}},
        "axes": {"aspect": "equal"},
    }
```

Templates are deep-copied before use, so one you build can be shared safely between subplots.
