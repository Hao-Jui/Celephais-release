# Kadath Public Headers

The public headers are organized by module. The `kadath` CMake target exposes
the module directories as public include paths, so existing internal flat
includes such as `#include "array.hpp"` still resolve without keeping duplicate
root-level forwarding headers.

| Directory | Purpose |
| --- | --- |
| `primitives/` | Low-level building blocks: arrays, indices, points, memory helpers, exceptions. |
| `spectral/` | Spectral bases and coefficient-space helpers. |
| `fields/` | Field containers and tensor/scalar value types. |
| `domains/` | Domain geometry headers. |
| `spaces/` | Space and multi-domain layout headers. |
| `equations/` | Equation DSL, metrics, parameters, and Jacobian helpers. |
| `linear_algebra/` | Dense matrix helper types. |
| `io/` | Binary/file/memory source and sink APIs. |
| `utilities/` | Runtime utilities, naming helpers, exporters, and solver runtime config. |
| `config/` | TOML-backed application configuration. |
| `third_party/` | Vendored standalone headers. |
| `Kadath_point_h/` | Umbrella convenience headers such as `kadath.hpp`. |

Header-included implementation fragments are co-located with their owning
headers and use `.inl`, for example `primitives/array.inl`.
