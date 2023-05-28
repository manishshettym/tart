***************
Installation
***************

Install tart from PyPI
--------------------------

.. code-block:: bash
        
        $ pip install tart-lib

Install tart from GitHub
----------------------------

.. note::
        We recommend using a virtual environment for installing tart. 
        Particularly, we recommend using `poetry <https://python-poetry.org/>`_ 
        for dependency management. However, you could also use your favorite
        package and virtual environment manager to install tart.

.. code-block:: bash

        $ pip install poetry
        $ # setup a virtual environment using poetry

.. code-block:: bash

        $ git clone git@github.com:tart-proj/tart.git
        $ cd tart
        $ poetry install
        $ poetry run sh setup.sh
        $ poetry run pip install -r requirements-dev.txt

.. note:: we are working on a single line command to simplify tart installation!!