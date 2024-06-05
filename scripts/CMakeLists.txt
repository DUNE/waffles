cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(hdf5libs VERSION 2.8.2)

find_package(daq-cmake REQUIRED)

daq_setup_environment()

#find_package(ROOT 6.30 CONFIG REQUIRED)
find_package(ROOT REQUIRED COMPONENTS RIO Tree Hist Imt)
find_package(logging REQUIRED)
find_package(ers REQUIRED)
find_package(HighFive REQUIRED)
find_package(daqdataformats REQUIRED)
find_package(detdataformats REQUIRED)
find_package(fddetdataformats REQUIRED)
find_package(trgdataformats REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost COMPONENTS iostreams unit_test_framework REQUIRED)

daq_codegen( *.jsonnet TEMPLATES Structs.hpp.j2 Nljs.hpp.j2 )

##############################################################################
# Main library
daq_add_library (HDF5FileLayout.cpp HDF5SourceIDHandler.cpp HDF5RawDataFile.cpp LINK_LIBRARIES stdc++fs ers::ers HighFive daqdataformats::daqdataformats detdataformats::detdataformats fddetdataformats::fddetdataformats trgdataformats::trgdataformats logging::logging nlohmann_json::nlohmann_json ROOT::RIO ROOT::Tree ROOT::Hist ROOT::Imt)

##############################################################################
# Unit tests
daq_add_unit_test( HDF5WriteReadTriggerRecord_test LINK_LIBRARIES hdf5libs )
daq_add_unit_test( HDF5WriteReadTimeSlice_test LINK_LIBRARIES hdf5libs )

##############################################################################
# Python bindings
daq_add_python_bindings(*.cpp LINK_LIBRARIES ${PROJECT_NAME} daqdataformats::daqdataformats detdataformats::detdataformats)

###############################################################################
# Applications
daq_add_application(HDF5LIBS_TestReader HDF5LIBS_TestReader.cpp TEST LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(HDF5LIBS_TestWriter HDF5LIBS_TestWriter.cpp TEST LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(HDF5LIBS_TestDumpRecord HDF5LIBS_TestDumpRecord.cpp TEST LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(HDF5LIBS_DumptoROOT_filtered HDF5LIBS_DumptoROOT_filtered.cpp TEST LINK_LIBRARIES ${PROJECT_NAME})

daq_install()