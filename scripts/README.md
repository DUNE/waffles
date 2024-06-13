To run the 00_HDF5toROOT.py it is necessary to build Root on the dbt environment, due to Python versions incompatibility.
To build Root you need to source the dbt env.sh first:
-  source env.sh
-  git clone git@github.com:root-project/root.git root_src
-  mkdir root_build root_install && cd root_build
-  cmake -DCMAKE_INSTALL_PREFIX=../root_install -D daframe=OFF ../root_src
-  make install

After that, everytime you login, you need to source Root:
-  source ../root_install/bin/thisroot.sh 
