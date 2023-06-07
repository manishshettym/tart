Learning using tart
====================================

.. note::
    coming soon!!

This tutorial explains how to train a ``tart`` model that can perform
subgraph matching and prediction on a graph dataset.

Step 1: Creating a dataset
--------------------------

Let's first organize your dataset. Each sample in the dataset is a graph.
`tart` expects all graphs to be in a JSON format, as follows:

.. code-block:: json
    
    {
        "directed": true,
        "graph": {},                    // any graph-level attributes
        "nodes": [                      // list of nodes. Each node is a list of [node attributes]
            ["node1 feat1", 10],            
            ["node2 feat1", 20],
            ["node3 feat1", 30],
            ["node4 feat1", 20]
        ],
        "edges": [                      // list of edges. Each edge is a list of [src, dst, [edge attributes]]
            [0, 1, ["0-1 edge", 1]],
            [1, 2, ["1-2 edge", 2]],
            [3, 0, ["3-0 edge", 3]],
            [0, 2, ["0-2 edge", 4]],
            [1, 3, ["1-3 edge", 5]]
        ]
    }

You can organize your dataset in the `/raw` subdirectory of your data directory. 
Here's a good following directory structure to follow:

.. code-block:: bash
    :linenos:

    my-dataset/
        train/
            raw/
                graph1.json
                graph2.json
                ...
        test/
            raw/
                graph1.json
                graph2.json
                ...


Step 2: Specify configs
----------------------------

Next, you need to specify the configs for tart. You can do this by creating a
`config.json` file in your data directory. Here's an example:

.. code-block:: json

    {   
        "dataset": "my-dataset",
        "data_dir": "/path/to/my-dataset",
        "feat_encoder": "CodeBert",                 // encoder to use for node and edge features
        "node_feats": ["f1", "f2"],                 // there are 2 node features
        "node_feat_types": ["str", "int"],          // the first feature is a string, the second is an int
        "node_feat_dims": [768, 1],                 // f1 maps to a 768-dim tensor, f2 maps to a 1-dim tensor
        "edge_feats": ["e1"],                       // similarly for edges ...
        "edge_feat_dims": [768],
        "edge_feat_types": ["str"],
        "n_batches": 5,                             // number of batches to use for training
        "eval_interval": 2,                         // evaluate every 2 batches
        "batch_size": 4,                            // batch size
        "min_size": 2,                              // minimum subgraph size to sample for training
        "max_size": 3,                              // maximum subgraph size to sample for training
    }

tart provides several other configs and hyperparameters that you can specify. It also provides inbuilt
support for tuning these hyperparameters using `hyperopt <https://github.com/hyperopt/hyperopt>`_. You can find the full list of configs and hyperparameters
at 
`model-configs <https://tart.readthedocs.io/en/latest/_modules/tart/representation/config.html#build_model_configs>`_ 
and `optimizer-configs <https://tart.readthedocs.io/en/latest/_modules/tart/representation/config.html#build_optimizer_configs>`_.

Step 3: Learn a tart model
------------------------

Write a super simple script using tart's APIs to train your model. Here's an example:

.. code-block:: python

    import torch
    from tart.representation.train import tart_train
    from tart.representation.test import tart_test
    from tart.inference.embed import tart_embed

    config_file = "config.json"
    tart_train(config_file)
    tart_test(config_file)

Run this script as usual ``python learn.py``. 
On doing so, tart first ensures your configuration is valid, then loads your dataset.
It then samples positive and negative examples of subgraphs from your dataset and
process them for training. Lastly, it encodes them into tensor representations and 
trains a graph neural network that can perform subgraph matching.

**Et voila! Your model is now trained and should be available in the `/ckpt` subdirectory 
of your root directory.**