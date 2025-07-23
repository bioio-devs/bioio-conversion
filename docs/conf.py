# -*- coding: utf-8 -*-
import os
import sys
import bioio_conversion as _bc

# add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "bioio-conversion"
author = "Your Name or Org"
copyright = "2025"
version = _bc.__version__
release = _bc.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "numpydoc",
    "myst_parser",
]
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "substitution"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# NumPyDoc and autodoc settings
numpydoc_show_class_members = False
autoclass_content = "both"
autodoc_mock_imports = []  # add if you need to mock heavy deps

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
