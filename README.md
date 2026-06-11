# x6-public-release
Public release for the scoped X6-minimal construction: standalone theorem/audit generator, finite X6​=Z_3^4​ RCFT/source-clock artifacts, and manuscript-facing data for the worldsheet, IR phenomenology, and gravity/cosmology companion papers.
Public release for the scoped X6-minimal standalone theorem/audit generator.
This repository contains the frozen public executable and paper-facing release artifacts for the X6-minimal construction. The code starts from the primitive three-branch `3(11,1)` figure-eight seed, builds the finite `X6 = Z3^4` structure, realizes the strict `A2^4 + T\_{-2}` BRST-reduced physical sector, and exports manuscript-facing outputs for the worldsheet, IR phenomenology, and gravity/cosmology papers.
# Scope
This release verifies the scoped X6 physical sector using:
the strict `A2^4` lattice-VOA cover,
the label-trivial `c = -2` BRST/topological reduction,
finite discriminant/projector data,
local observable closure,
source-clock transport,
finite quadratic stability,
and downstream paper-facing artifact layers.
It does not claim:
unrestricted global UV completion,
absolute exclusion of all nonlocal/external completions,
or anchor-free SI gravity.
Those boundaries are explicit in the release checks and metadata.
# Entry point
Main executable:
```bash
python X6\_PUBLIC\_RELEASE\_v1\_10\_06\_2026.py --mode release --release-prefix X6\_PUBLIC\_RELEASE
