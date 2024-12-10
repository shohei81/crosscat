GenJAXMix is a GenJAX-accelerated Dirichlet Process Mixture Modeling package for heterogenous tabular data. 

# Usage

GenJAXMix is currently under active development. Here is the current state of the repo:

- `main` is the latest stable GenJAXMix package.
- `pure-jax` contains a barebone JAX implementation dedicated for experiments.
- Various feature branches contain incoming PRs into `main`.

# Installation

Currently GenJAXMix is not registered on a package index. Please install `main`.

Example:
```
pip install git+https://github.com/OpenGen/genjaxmix.git
```

# Documentation
wip

# Examples
wip

# Development

`genjaxmix` is currently private and depends on `genjax` which is also private. To configure your machine for permissions, please 

- Follow the GCP authentication instructions [listed here](https://genjax.gen.dev/#quickstart).

- Run `uv sync`.
