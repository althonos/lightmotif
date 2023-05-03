#!/usr/bin/env python3

import configparser
import os
import shutil
import subprocess
import sys
import urllib.request
from distutils.errors import DistutilsPlatformError
from distutils.log import INFO

import setuptools
import setuptools_rust as rust
from setuptools.command.sdist import sdist as _sdist
from setuptools_rust.build import build_rust as _build_rust

try:
    from setuptools_rust.rustc_info import get_rust_version
except ImportError:
    from setuptools_rust.utils import get_rust_version


class sdist(_sdist):

    def run(self):
        # build `pyproject.toml` from `setup.cfg`
        c = configparser.ConfigParser()
        c.add_section("build-system")
        c.set("build-system", "requires", str(self.distribution.setup_requires))
        c.set("build-system", 'build-backend', '"setuptools.build_meta"')
        with open("pyproject.toml", "w") as pyproject:
            c.write(pyproject)

        # run the rest of the packaging
        _sdist.run(self)


class build_rust(_build_rust):

    def run(self):
        rustc = get_rust_version()
        if rustc is not None:
            nightly = rustc is not None and "nightly" in rustc.prerelease
        else:
            self.setup_temp_rustc_unix(toolchain="stable", profile="minimal")
            nightly = False

        if self.inplace:
            self.extensions[0].strip = rust.Strip.No
        if nightly:
            self.extensions[0].features = (*self.extensions[0].features, "nightly")

        _build_rust.run(self)

    def setup_temp_rustc_unix(self, toolchain, profile):
        rustup_sh = os.path.join(self.build_temp, "rustup.sh")
        os.environ["CARGO_HOME"] = os.path.join(self.build_temp, "cargo")
        os.environ["RUSTUP_HOME"] = os.path.join(self.build_temp, "rustup")

        self.mkpath(os.environ["CARGO_HOME"])
        self.mkpath(os.environ["RUSTUP_HOME"])

        self.announce("downloading rustup.sh install script", level=INFO)
        with urllib.request.urlopen("https://sh.rustup.rs") as res:
            with open(rustup_sh, "wb") as dst:
                shutil.copyfileobj(res, dst)

        self.announce("installing Rust compiler to {}".format(self.build_temp), level=INFO)
        proc = subprocess.run([
            "sh",
            rustup_sh,
            "-y",
            "--default-toolchain",
            toolchain,
            "--profile",
            profile,
            "--no-modify-path"
        ])
        proc.check_returncode()

        self.announce("updating $PATH variable to use local Rust compiler", level=INFO)
        os.environ["PATH"] = ":".join([
            os.path.abspath(os.path.join(os.environ["CARGO_HOME"], "bin")),
            os.environ["PATH"]
        ])

    def get_dylib_ext_path(self, ext, module_name):
        ext_path = _build_rust.get_dylib_ext_path(self, ext, module_name)
        # if self.inplace:
        #     base = os.path.basename(ext_path)
        #     folder = os.path.dirname(os.path.realpath(__file__))
        #     prefix = os.path.sep.join(ext.name.split(".")[:-1])
        #     ext_path = os.path.join(folder, prefix, base)
        return ext_path


# HACK: Use the `configparser` from Python to read the `Cargo.toml`
#       manifest file (this... works) so that the package version
#       can be extracted from there and synchronized everywhere.
parser = configparser.ConfigParser()
parser.read(os.path.join(os.path.dirname(__file__), "lightmotif-py", "Cargo.toml"))
version = parser.get("package", "version").strip('"')

setuptools.setup(
    version=version,
    setup_requires=["setuptools", "setuptools_rust"],
    cmdclass=dict(sdist=sdist, build_rust=build_rust),
    rust_extensions=[rust.RustExtension(
        "lightmotif.lib",
        path=os.path.join("lightmotif-py", "Cargo.toml"),
        binding=rust.Binding.PyO3,
        strip=rust.Strip.Debug,
        features=["extension-module"],
    )],
)
