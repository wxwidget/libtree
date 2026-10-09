"""Build the small native library into platform-specific Python wheels."""
import os
from pathlib import Path
import subprocess

from setuptools import setup, Distribution
from setuptools.command.build_py import build_py


class NativeBuild(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent.resolve()
        destination = Path(self.build_lib) / "libtree" / "_native.so"
        subprocess.run([
            os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-DNDEBUG",
            "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
            f"-I{root / 'include'}", str(root / "src/gbdt.cc"),
            str(root / "src/c_api.cc"), str(root / "src/model.cc"), "-o", str(destination),
        ], check=True)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": NativeBuild}, distclass=BinaryDistribution)
