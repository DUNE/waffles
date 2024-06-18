To run the 00_HDF5toROOT.py it is necessary to build Root on the dbt environment, due to Python versions incompatibility.
To build Root you need to source the dbt env.sh first:

source env.sh

git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

mkdir root_build root_install && cd root_build

cmake -DCMAKE_INSTALL_PREFIX=../root_install -D dataframe=OFF ../root_src

make install

After that, everytime you login, you need to source Root, or you can edit env.sh and add this command:

source ../root_install/bin/thisroot.sh 
