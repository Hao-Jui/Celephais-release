# Kadath_point_h

This folder provides convenience "umbrella" headers for Kadath. Each header pulls in

the core Kadath types and then a single domain/geometry-specific header. This gives

users a simple entry point depending on the geometry or configuration they need.

Design overview

- `kadath.hpp` is the all-in-one include. It brings in the common core headers and then
  all supported domain headers (spheric, polar, adapted, binary systems, periodic, etc.).
- Each `kadath_*.hpp` file is a targeted entry point: it includes the same core stack
  and then exactly one domain header. For example:
  - `kadath_spheric.hpp` -> `spheric.hpp`
  - `kadath_symphi.hpp` -> `spheric_symphi.hpp`
  - `kadath_adapted_bh.hpp` -> `adapted_bh.hpp`

Why this layout

- Keep user includes short and explicit.
- Avoid manual assembly of dozens of core headers in client code.
- Allow "pick one geometry" use while keeping a single "everything" option.

Notes

- The headers are thin wrappers; they do not add logic, only include ordering.
- Use `kadath.hpp` if you want all domains, or a specific `kadath_*.hpp` for faster
  compilation and clearer dependencies.
