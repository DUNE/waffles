To run the ``00_HDF5toROOT.py`` it is necessary to build ROOT on the dbt environment, due to versions incompatibility.
To build ROOT you need to source the dbt ``env.sh`` first:

```bash
source env.sh

git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

mkdir root_build root_install && cd root_build

cmake -DCMAKE_INSTALL_PREFIX=../root_install -Ddataframe=OFF ../root_src

cmake --build . -- install -j1
```

If some error appears during the installation related with Roofit, just disable it by adding the command ``-Droofit=OFF`` on the cmake.

After that, everytime you login, you need to source ROOT, or you can edit ``env.sh`` and add this command:

``source ../root_install/bin/thisroot.sh``
