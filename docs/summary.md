**🔥 tensor abstractions for fast analysis of graph relations 🔥**

```shell
pip install tart
tart init
```

tart is a library for learning approximate and abstract relations between graphs. It is currently designed for a very general
graph relation -- the subgraph relation (⊆). tart learns tensor representations that encodes the subgraph relation between graphs and provides various APIs to train, evaluate, and predict over these representations.

tart is:
1. 🌐 General Purpose (JSON graphs and type-agnostic features)
2. 🪢 Extensible (supports custom models and encoders)
3. 🔥 Fast (tensor comparisons for relation analysis)

<!-- tart is used by:
1. CodeScholar: exploring code idiom search as frequent subgraph mining.
2. tartSAT: capturing common UNSAT cores as frequent subgraphs in formulae.
3. egg-tart: exploring common subgraphs in e-graphs. -->