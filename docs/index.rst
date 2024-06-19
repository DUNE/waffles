.. waffles documentation master file, created by
   sphinx-quickstart on Mon May 20 16:37:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <style> 
      .tealtitle {color:#008080; font-weight:bold; font-size:60px} 
      .tealsttle {color:#008080; font-weight:bold; font-size:30px} 
      .tealstext {color:#008080; font-weight:bold; font-size:17px} 
      .tealtexts {color:#008080; font-weight:bold; font-size:12px} 
   </style>

.. role:: tealtitle
.. role:: tealsttle
.. role:: tealstext
.. role:: tealtexts

====================================================
   :tealtitle:`WELCOME TO WAFFLES`
====================================================

The **Waveform Analysis Framework For Light Emission Studies** (:tealstext:`WAFFLES`) is a Python library used for the PDS analysis in NP04.

You can navigate through the documentation using the table of contents below, and you can search for specific keywords using the search tab placed at left side.

---------------------------------------------------------------------------------------------------------------------------------------------

**CONTENTS**
============
.. toctree::   
    :maxdepth: 1

    1_Intro  
    2_Scripts  
    3_Libraries

    examples/4_Examples

---------------------------------------------------------------------------------------------------------------------------------------------

.. warning::
    🚧 This project is still under development. Please, contact the authors for more information.🚧

---------------------------------------------------------------------------------------------------------------------------------------------

**SUMMARY**
-------------

For a quick summary or just as a reminder follow the next steps:


0.- Clone the repository into a local directory and create your branch:

.. code-block:: bash

   git clone https://github.com/DUNE/waffles.git
   cd waffles
   git checkout -b <your_branch_name>


1.- You will need a ``daq environment`` to run the library from the beginning to the end. [You may just want to analyse the already generated ``root`` files making this requirement optional].

Depending on the scope of your analysis you may need a different machine:
   
* **OFFLINE** analysis: log-in to lxplus with your CERN user: ``ssh CERNuser@lxplus.cern.ch``
* **ONLINE** + **DATA TAKING**: log-in to np04-srv-015 with your CERN user: ``ssh CERNuser@np04-srv-015``

Inside the daq machines, to access the network we need to do: ``source ~np04daq/bin/web_proxy.sh``

To install the environment check for the last version and run:

.. code-block:: bash

   source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh

   setup_dbt latest
   dbt-create -l 
   dbt-create fddaq-v4.4.2-a9 <my_dir>

2.- Install packages needed for the library to run:

.. code-block:: bash

   cd waffles/scripts
   sh setup.sh


3.- Make sure you have access to data to analyze:
   
* **RUCIO PATHS**: ``/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`` 
* **RAW ROOT FILES**: ``/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root``


4.- Have a look on the ``notebooks`` folder to see how to visualize data and run the macros (?):

.. code-block:: bash

   cd ../notebooks
   juptyer notebook 00TUTORIAL.ipynb

This folder is prepared **NOT** to be synchronized with the repository, so you can make your own analysis without affecting the rest of the team.
If you think your analysis can be useful for the rest of the team contact the authors to add it to the repository.

If you want to have a personal folder to store your test files locally you can create a ``test`` folder (it won't be synchronized with the repository).
Otherwise, you can create a folder for your custom scripts and add them to the ``.gitignore`` file:

.. code-block:: bash

   mkdir <your_folder_name>
   echo "<your_folder_name/*>" >> .gitignore


5.- To run the scripts:

.. code-block:: bash

   cd ../scripts
   python XXmacro.py (--flags input)


.. admonition:: **Preferred work-flow**
   
   Convert the ``hdf5`` files to ``root`` files using ``00_HDF5toROOT`` scripts. (For using the bash script you need to have the cpp tools previously compiled, see ``cpp_utils`` folder).
   This will generate the rucio paths if they are not already created and store them in the ``1_rucio_paths`` folder. The output will be stored in the ``2_daq_root`` folder.

---------------------------------------------------------------------------------------------------------------------------------------------

**INDICES AND TABLES**
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`