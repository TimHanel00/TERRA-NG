# Distributed MMOC transport

This implementation combines the MMOC solver from `suganth/mmoc-transport-v1`
(`260989b246ca914617d089be92c154079098df94`) with bundled owner queries. The mantle
application passes its actual subdomain-to-rank distribution into `MMOCTransport`.
The original Runge–Kutta schemes, global pseudo-time velocity blending, monotone
cubic temperature reconstruction, Q1 velocity interpolation, limiters, and
separate diffusion step are retained.

## Sampling workflow

1. Fill temperature ghosts once at the start of a timestep. Copy current and
   previous velocity values into the interior of their padded buffers without
   exchanging velocity ghosts: Q1 samples only the destination cell's vertices.
2. Form physical sampling positions for the current Runge–Kutta stage.
3. Locate positions inside their original owned subdomain using the direct
   physical-wedge locator. Ghost cells are excluded from this containment test.
4. For departures, determine the diamond and lateral/radial subdomain. Evaluate
   on this rank if it holds the destination; otherwise bundle requests by owner
   rank, exchange them, and run the same locator and interpolator there.
5. Return sampled values to their original indices. Advance the trajectories and
   repeat for the remaining stages/substeps, then sample temperature.
6. Write owning field nodes and synchronize shared node copies with the boundary
   SUM exchange. This exchange is separate from particle query/reply traffic.

All ranks participate in each sampling stage, including ranks with no outgoing
queries or no subdomains. A collective detects stages with no remote work and
skips the query exchange. A successful step has no unresolved samples. Invalid
positions or failed owner containment abort the communicator; they do not use a
clamped local escape approximation. Physical shell radial clamping is retained.

## Geometry and ghosts

Ghost values can extend the interpolation stencil of an owned boundary cell.
A particle landing in a ghost cell is sampled by that cell's owning subdomain.
Internal lateral subdivisions of the same diamond and internal radial
subdivisions may supply stencil values. Diamond chart seams and physical radial
boundaries use inward stencils, excluding extrapolated physical-boundary data.

Direct fine-cell lookup uses stored owner geometry. Sender-side routing normally
needs only partition-resolution geometry. Near radial interfaces, the physical
wedge radius can disagree with the norm of the position. A compacted exceptional
pass resolves those cases with exact fine geometry, using stored local lateral
coordinates when available and procedural coordinate reconstruction otherwise.
That reconstruction still depends on refinement depth; the complete routing
operation is not claimed to be uniformly O(1).

## Shared-memory interpolation

Located queries are histogrammed, prefix-scanned, and scattered into 4×4×4-cell
tiles. One CUDA team of 128 threads handles each occupied tile. Threads load the
union stencil cooperatively, synchronize, and reuse it for the tile's queries.
Temperature scratch holds up to 343 scalar values and seven radial coordinates
(2,800 bytes for doubles). Velocity scratch holds up to 125 nodes from each of
the current and previous three-component velocity fields (6,000 bytes).

`set_shared_interpolation(false)` selects a diagnostic global-load reference.
Both paths call the same interpolation formulas with the same stencil bounds.
The shared path includes grouping overhead, so its speed must be measured for
the complete sampling stage rather than inferred from reduced field loads.

## Storage and validation

Query, reply, location, and grouping buffers retain capacity across calls.
`sampling_buffer_bytes()` reports allocated device sampling and trajectory
buffers. It excludes fields, ghost fields, halo communication, host request
batches, MPI internals, and runtime allocations; it is not total GPU memory.
`last_remote_queries()` counts requests across the last timestep's RK and final
temperature samples. Replies carry the original index and scalar/vector value.

Focused tests cover shared/global equivalence, complete-mesh reference sampling,
multi-stage RK with distinct old/current velocities, mixed radial ownership,
empty ranks, and the ghost-stencil versus halo-landing rule. Run them with:

Velocity ghost entries are poisoned with NaN in the sampling tests to detect
accidental reads outside the owned-cell vertices in either interpolation path.

```sh
ctest --output-on-failure -R 'mmoc_hybrid|mmoc_ghost_stencil|direct_diamond|point_location_random'
```

`test_mmoc_transport_benchmark` runs complete RK4 MMOC timesteps. Its source also
builds against v1 without `TERRA_MMOC_HYBRID_BENCH`, providing the same prescribed
velocity, temperature, timestep, mesh, and step count for comparisons.
It supports `--level`, `--steps`, `--lateral-subdivision`,
`--radial-subdivision`, and `--global-interpolation` for the hybrid reference.
