# xi-bundle-quantum

Quantum simulation of the xi-bundle topological structure.

## Overview

This repository contains quantum computing modules for verifying the xi-bundle framework:
- **DQPT**: Dynamical quantum phase transitions with kappa-injected couplings
- **Floquet**: Riemann zero detection via Loschmidt echo
- **VQE**: Variational quantum eigensolver with Gauss-Bonnet constraints
- **Hamiltonian constraints**: 8 selection rules for Berry-Keating candidates
- **Bethe-GB equivalence**: Bethe Ansatz phase quantization = Gauss-Bonnet condition
- **Topological code**: xi-bundle stabilizer code for quantum error correction

## Relationship to Main Repository

This is a **specialized export** from the monorepo [rdl-resonant-detection](https://github.com/younghu-kim/rdl-resonant-detection), which contains the complete codebase and all results.

- **Full code**: See [rdl-resonant-detection](https://github.com/younghu-kim/rdl-resonant-detection)
- **Mathematical framework**: See Paper I (xi-bundle geometry) in the main repo
- **GL(2) extension**: See [xi-bundle-gl2](https://github.com/younghu-kim/xi-bundle-gl2)

## Modules

```
quantum/
├── utils/quantum_utils.py          # Pauli matrices, fidelity, entropy
├── hamiltonian/
│   ├── bethe_ansatz_gb.py          # Bethe Ansatz <-> Gauss-Bonnet equivalence
│   └── constraints.py              # 8 Hamiltonian selection rules (C1-C8)
├── dqpt/
│   ├── dqpt_zeta.py                # DQPT with kappa-injection
│   └── floquet_zeros.py            # Floquet protocol zero detection
├── vqe/
│   └── vqe_gauss_bonnet.py         # VQE with GB constraint optimization
├── topological/
│   └── xi_bundle_code.py           # Topological stabilizer code
└── run_all.py                      # Integration runner
```

## Auto-sync

Files are automatically synced from the monorepo via `sync_repos.sh` (daily at 03:00).

## License

MIT
