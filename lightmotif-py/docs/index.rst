.. lightmotif documentation master file, created by
   sphinx-quickstart on Tue Jul 18 16:36:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Dark| |Light| LightMotif |Stars|
=================================

.. |Dark| image:: _images/logo-dark.png
   :class: only-dark 
   :scale: 50%

.. |Light| image:: _images/logo-light.png
   :class: only-light
   :scale: 50%

.. |Stars| image:: https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&maxAge=3600&label=Star
   :target: https://github.com/althonos/lightmotif/stargazers
   :class: dark-light

*A lightweight* `platform-accelerated <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`_ *library for* `biological motif <https://en.wikipedia.org/wiki/Sequence_motif>`_ *scanning using* `position weight matrices <https://en.wikipedia.org/wiki/Position_weight_matrix>`_.

|Actions| |Coverage| |License| |Docs| |Crate| |PyPI| |Wheel| |Bioconda| |Python Versions| |Python Impls| |Source| |Mirror| |Issues| |Changelog| |Downloads|

.. |Actions| image:: https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/python.yml?branch=main&logo=github&style=flat-square&maxAge=300
   :target: https://github.com/althonos/lightmotif/actions

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600
   :target: https://codecov.io/gh/althonos/lightmotif/

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400
   :target: https://choosealicense.com/licenses/mit/

.. |Docs| image:: https://img.shields.io/readthedocs/lightmotif/stable?style=flat-square&maxAge=600
   :target: https://lightmotif.readthedocs.io/en/stable/

.. |Crate| image:: https://img.shields.io/crates/v/lightmotif-py.svg?maxAge=600&style=flat-square
   :target: https://crates.io/crates/lightmotif-py

.. |PyPI| image:: https://img.shields.io/pypi/v/lightmotif.svg?style=flat-square&maxAge=600
   :target: https://pypi.org/project/lightmotif

.. |Wheel| image:: https://img.shields.io/pypi/wheel/lightmotif.svg?style=flat-square&maxAge=2678400
   :target: https://pypi.org/project/lightmotif/#files

.. |Bioconda| image:: https://img.shields.io/conda/vn/bioconda/lightmotif?style=flat-square&maxAge=3600
   :target: https://anaconda.org/bioconda/lightmotif

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/lightmotif.svg?style=flat-square&maxAge=600
   :target: https://pypi.org/project/lightmotif/#files

.. |Python Impls| image:: https://img.shields.io/pypi/implementation/lightmotif.svg?style=flat-square&maxAge=600
   :target: https://pypi.org/project/lightmotif/#files

.. |Source| image:: https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/althonos/lightmotif/tree/main/lightmotif-py

.. |Mirror| image:: https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400
   :target: https://git.embl.de/larralde/lightmotif/

.. |Issues| image:: https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600
   :target: https://github.com/althonos/lightmotif/issues

.. |Changelog| image:: https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md

.. |Downloads| image:: https://img.shields.io/pypi/dm/lightmotif?style=flat-square&color=303f9f&maxAge=86400&label=downloads
   :target: https://pepy.tech/project/lightmotif


Overview
--------

`Motif <https://en.wikipedia.org/wiki/Sequence_motif>`_ scanning with 
`position weight matrices <https://en.wikipedia.org/wiki/Position_weight_matrix>`_
(also known as position-specific scoring matrices) is a robust method for 
identifying motifs of fixed length inside a 
`biological sequence <https://en.wikipedia.org/wiki/Sequence_(biology)>`_. They can be 
used to identify `transcription factor <https://en.wikipedia.org/wiki/Transcription_factor>`_ 
`binding sites in DNA <https://en.wikipedia.org/wiki/DNA_binding_site>`_, 
or `protease <https://en.wikipedia.org/wiki/Protease>`_ `cleavage <https://en.wikipedia.org/wiki/Proteolysis>`_ site in `polypeptides <https://en.wikipedia.org/wiki/Protein>`_. 
Position weight matrices are often viewed as `sequence logos <https://en.wikipedia.org/wiki/Sequence_logo>`_:

.. image:: https://raw.githubusercontent.com/althonos/lightmotif/main/docs/_static/prodoric_logo_mx000274.svg
   :target: https://www.prodoric.de/matrix/MX000274.html
   :class: dark-light

The ``lightmotif`` library provides a Python module to run very efficient
searches for a motif encoded in a position weight matrix. The position
scanning combines several techniques to allow high-throughput processing
of sequences:

.. grid:: 1 2 4 4
   :gutter: 1

   .. grid-item-card:: :fas:`fa-solid fa-arrow-down-a-z` Constant alphabets

      Compile-time definition of alphabets and matrix dimensions, allowing
      constant pointer incrementation in loops over strided arrays.

   .. grid-item-card:: :fas:`fa-solid fa-arrow-down-1-9` Ordinal encoding

      Sequence symbol encoding as indices fo fast table look-ups, as 
      implemented in `HMMER <https://hmmer.org>`_ or 
      `MEME <https://meme-suite.org/>`_.

   .. grid-item-card:: :fas:`fa-solid fa-table` Striped matrices

      Striped sequence matrices for parallel processing, 
      inspired by Michael Farrar (:pmid:`17110365`).

   .. grid-item-card:: :fas:`fa-solid fa-microchip` Platform code

      Vectorized matrix-row look-up using ``permute`` instructions of 
      `AVX2 <https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions>`_.

*This is the Python version, there is a* `Rust crate <https://docs.rs/lightmotif>`_ *available as well.*


Setup
-----

Run ``pip install lightmotif`` in a shell to download the latest release and all
its dependencies from PyPi, or have a look at the
:doc:`Installation page <guide/install>` to find other ways to install the 
``lightmotif`` Python package.


Library
-------

.. toctree::
   :maxdepth: 2

   User Guide <guide/index>
   API Reference <api/index>


Related Projects
----------------

The following Python libraries may be of interest for bioinformaticians.

.. include:: related.rst


License
-------

This library is provided under the open-source
`MIT license <https://choosealicense.com/licenses/mit/>`_.

*This project was developed by* `Martin Larralde <https://github.com/althonos/>`_
*during his PhD project at the* `European Molecular Biology Laboratory <https://www.embl.de/>`_
*in the* `Zeller team <https://github.com/zellerlab>`_.

