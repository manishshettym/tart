# Configuration file for the Sphinx documentation builder.


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information

project = "tart"
copyright = "2021, The tart authors"
author = "The tart authors"

release = "0.1"
version = "0.1.1"

# -- General configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxext.opengraph",
    "sphinx_click",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# autodoc
autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_options = {
    # Make sure that any autodoc declarations show the right members
    "members": True,
    "inherited-members": True,
    "private-members": False,
    "show-inheritance": True,
}
autosummary_generate = True  # Make _autosummary files and include them
napoleon_numpy_docstring = False  # Force consistency, leave only Google
napoleon_use_rtype = False  # More legible

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Options for HTML output

html_theme = "furo"
html_theme_options = {
    "light_logo": "tart-title.svg",
    "dark_logo": "tart-title.svg",
    "sidebar_hide_name": True,
}
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"
