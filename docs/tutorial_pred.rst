Prediction using tart
====================================

This tutorial explains how to use a trained ``tart`` model for inference.
We will use the model for two prediction tasks (1) subgraph matching, and (2) subgraph counting.

Task 1: Subgraph Matching
--------------------------

**Task Description**. Given a query graph, and a target graph, find if the
query graph is a subgraph of the target graph.

**Prerequisites**. To run this tutorial, you will need the following:

1. A trained ``tart`` model. See the tutorial on training a model.
2. Query and target graphs in the JSON format expected by your trained model. See the steps on data preparation in the 
`learning tutorial <https://tart.readthedocs.io/en/latest/tutorial_learn.html#step-1-creating-a-dataset>`_  section.

.. code-block:: python
    
    import torch
    from tart.inference.predict import tart_predict

    config_file = "config.json"
    query_json = "g1.json"
    target_json = "g2.json"
    tart_predict(config_file, query_json, search_json, outcome="is_subgraph")


Task 2: Subgraph Counting
--------------------------

**Task Description**: Given a query graph, and a dataset of target graphs, find
the number of target graphs that contain the query graph as a subgraph.

**Prerequisites**. To run this tutorial, you will need the following:

1. A trained ``tart`` model. See the tutorial on training a model.
2. Query and target graphs in the JSON format expected by your trained model. See the steps on data preparation in the
`learning tutorial <https://tart.readthedocs.io/en/latest/tutorial_learn.html#step-1-creating-a-dataset>`_  section.

.. code-block:: python
    
    import torch
    from tart.inference.predict import tart_predict

    config_file = "config.json"
    query_json = "g1.json"

    # step 1: embed the target graphs and saves it to disk
    tart_embed(config_file)

    # step 2: point to the directory with target embeddings
    target_embs_dir = "path/.../to/embeddings"

    # step 3: run inference
    tart_predict(config_file, query_json, target_embs_dir, outcome="count_subgraphs")

