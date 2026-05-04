import sys
from pathlib import Path

# Add custom Sphinx extensions to module search path.
here = Path(__file__).parent
sys.path.insert(0, str(here.parent / "extensions"))
sys.path.insert(0, str(here.parents[1] / "src"))
from comsol_module import meta  # ingore

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Load Sphinx extensions.
extensions = [
    "myst_parser",  # Markdown support in documents
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    # "myst_docstring",  # Markdown support in doc-strings
    # "myst_summary",  # Markdown support in summary tables
    "sphinx.ext.intersphinx",  # inter-project cross-references
]


# Meta information
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = meta.name
version = meta.version
release = version

# Web site
html_title = f"{project} {version}"  # document title
# html_logo = "images/logo.svg"  # project logo
# html_favicon = "images/logo.svg"  # browser icon


# Source parsing
root_doc = "index"  # start page
nitpicky = True  # Warn about missing references?
exclude_patterns = ["ReadMe.md"]  # Ignore ReadMe in this folder here.

# Code documentation
autodoc_default_options = {
    "members": True,  # Include module/class members.
    "member-order": "bysource",  # Order members as in source file.
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "none"  # Rendering type hints doesn't work.
autosummary_generate = True  # Stub files are created by hand.
add_module_names = False  # Drop module prefix from signatures.

# External link targets
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}


# Rendering options
myst_heading_anchors = 2  # Generate link anchors for sections.
html_copy_source = False  # Copy documentation source files?
html_show_copyright = False  # Show copyright notice in footer?
html_show_sphinx = False  # Show Sphinx blurb in footer?

# Rendering style
html_theme = "furo"  # custom theme with light and dark mode
pygments_style = "friendly"  # syntax highlight style in light mode
pygments_dark_style = "stata-dark"  # syntax highlight style in dark mode
# html_static_path = ["style"]  # folders to include in output
# html_css_files = ["custom.css"]  # extra style files to apply

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
