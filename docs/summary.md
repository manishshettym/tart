**🔥 tensor abstractions for fast analysis of graph relations 🔥**

```shell
pip install tart
tart init
```

tart is a library for learning approximate and abstract relations between graphs. It is currently designed for a very general
graph relation -- the subgraph relation (⊆). tart learns tensor representations that encodes the subgraph relation between graphs and provides various APIs to train, evaluate, and predict over these representations.

<h3><span style="background-image: linear-gradient(-294deg, #1E98FD, #FF00F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">tart</span> is:</h3>

1. 🌐 General Purpose (json graphs and type-agnostic features)
2. 🪢 Extensible (supports custom models and encoders)
3. 🔥 Fast (tensor comparisons for relation analysis)

</br>


<h3><span style="background-image: linear-gradient(-294deg, #1E98FD, #FF00F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">tart</span> in 3 steps:</h3>
<div float="left" style="display: flex; flex-wrap: wrap">
    <figure style="width:30%; vertical-align:middle">
        <figcaption style="text-align:center"><em><b>Load JSON Graphs</b></em></figcaption>
        <img src="_static/graph.svg">
    </figure>
    <figure style="width:34%; vertical-align:middle">
        <figcaption style="text-align:center"><em><b>Specify Configs</b></em></figcaption>
        <img src="_static/configs.svg">
    </figure>
    <figure style="width:34%; vertical-align:middle">
        <figcaption style="text-align:center"><em><b>Use APIs</b></em></figcaption>
        <img src="_static/apis.svg">
    </figure>
    
</div>

<!-- tart is used by:
1. CodeScholar: exploring code idiom search as frequent subgraph mining.
2. tartSAT: capturing common UNSAT cores as frequent subgraphs in formulae.
3. egg-tart: exploring common subgraphs in e-graphs. -->