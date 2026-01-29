This folder contains prebuilt helper binaries for PTO-ISA.

- `bin/ptoas` is a small wrapper that dispatches to an OS/arch-specific binary:
  - Linux aarch64: `bin/linux-aarch64/ptoas.tar.gz` (**included**, auto-extracted)
  - Linux x86_64: `bin/linux-x86_64/ptoas` (**not included**)
  - macOS aarch64: `bin/macos-aarch64/ptoas` (**not included**)

On Linux aarch64, the repo ships `bin/linux-aarch64/ptoas.tar.gz`; `bin/ptoas` will auto-extract it to `bin/linux-aarch64/ptoas` on first use.

If your platform binary is missing, place a compatible `ptoas` executable at the path above and ensure it is executable (or provide a `ptoas.tar.gz` next to it).

Quick check:

```bash
./bin/ptoas --help
```

Rebuild from a local `~/llvm-project` checkout:

```bash
./scripts/rebuild_ptoas.sh
```
