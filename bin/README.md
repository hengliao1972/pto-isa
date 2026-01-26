This folder contains prebuilt helper binaries for PTO-ISA.

- `bin/ptoas` is a small wrapper that dispatches to an OS/arch-specific binary:
  - Linux aarch64: `bin/linux-aarch64/ptoas` (**included**)
  - Linux x86_64: `bin/linux-x86_64/ptoas` (**not included**)
  - macOS aarch64: `bin/macos-aarch64/ptoas` (**not included**)

If your platform binary is missing, place a compatible `ptoas` executable at the path above and ensure it is executable.

Quick check:

```bash
./bin/ptoas --help
```
