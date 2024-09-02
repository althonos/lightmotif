Installation
============

.. note::

    Wheels are provided for Windows, Linux and OSX x86-64 platforms, as well as 
    Linux and OSX Aarch64 platforms. Other machines will have to build the wheel 
    from the source distribution. Building ``lightmotif`` involves compiling 
    Rust code, which requires a Rust compiler to be available.


PyPi
^^^^

``lightmotif`` is hosted on GitHub, but the easiest way to install it is to download
the latest release from its `PyPi repository <https://pypi.python.org/pypi/lightmotif>`_.
It will install all dependencies then install ``lightmotif`` either from a wheel if
one is available, or from source after compiling the Rust code :

.. code:: console

	$ pip install --user lightmotif


Arch User Repository
^^^^^^^^^^^^^^^^^^^^

A package recipe for Arch Linux can be found in the Arch User Repository
under the name `python-lightmotif <https://aur.archlinux.org/packages/python-lightmotif>`_.
It will always match the latest release from PyPI.

Steps to install on ArchLinux depend on your `AUR helper <https://wiki.archlinux.org/title/AUR_helpers>`_
(``yaourt``, ``aura``, ``yay``, etc.). For ``aura``, you'll need to run:

.. code:: console

   $ aura -A python-lightmotif


GitHub + ``pip``
^^^^^^^^^^^^^^^^

If, for any reason, you prefer to download the library from GitHub, you can clone
the repository and install the repository by running (with the admin rights):

.. code:: console

	$ pip install -U git+https://github.com/althonos/lightmotif

.. caution::

    Keep in mind this will install always try to install the latest commit,
    which may not even build, so consider using a versioned release instead.


GitHub + ``setuptools``
^^^^^^^^^^^^^^^^^^^^^^^

If you do not want to use ``pip``, you can still clone the repository and
run the ``setup.py`` file manually, although you will need to install the
build dependencies (mainly ``setuptools-rust``):

.. code:: console

	$ git clone --recursive https://github.com/althonos/lightmotif
	$ cd lightmotif
	$ python setup.py build
	# python setup.py install

.. Danger::

    Installing packages without ``pip`` is strongly discouraged, as they can
    only be uninstalled manually, and may damage your system.
