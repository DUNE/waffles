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

.. image:: _static/waffles_logo.png
   :width: 500px
   :align: center
   :alt: waffles_logo

.. raw:: html

   <br />

The **Waveform Analysis Framework For Light Emission Studies** (:tealstext:`WAFFLES`) is a Python library used for the PDS analysis in NP04.

You can navigate through the documentation using the table of contents below, and you can search for specific keywords using the search tab placed at left side.

---------------------------------------------------------------------------------------------------------------------------------------------

**CONTENTS**
============
.. toctree::   
    :maxdepth: 2

    10_Intro  
    20_Scripts  
    30_Libraries
    examples/40_Examples

---------------------------------------------------------------------------------------------------------------------------------------------

.. warning::
    🚧 This project is still under development. Please, contact the authors for more information.🚧

---------------------------------------------------------------------------------------------------------------------------------------------

**CONNECTION TO LXPLUS@CERN.CH ARE NEEDED** -- Check your `CERN account configuration <https://resources.web.cern.ch/resources/>`_.

**QUICK START**
================

-------------------------
0.- Clone the repository
-------------------------

To clone the repository and create your own branch run:

.. code-block:: bash

   git clone https://github.com/DUNE/waffles.git 
   cd waffles
   git checkout -b <your_branch_name>

If you want to have a personal folder to store your test files locally you can create a ``test`` folder (it won't be synchronized with the repository).
Otherwise, you can create a folder for your custom scripts and add them to the ``.gitignore`` file:

.. code-block:: bash

   mkdir <your_folder_name>
   echo "<your_folder_name/*>" >> .gitignore

-----------------------------------------------------------------------------------
1.- Install (first time) or source your virtual environment (every time you log-in)
-----------------------------------------------------------------------------------

* If you need to decode ``.hdf5`` files you need to install the ``dbt`` environment. This environment is needed to extract PDS data from the raw DAQ data.

  To install the environment check for the last version and run:

  .. code-block:: bash
  
     source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh  # Source the dunedaq environment
  
     setup_dbt latest                                            # Setup the latest version of dbt
     dbt-create -l                                               # List the available versions of dbt
     dbt-create fddaq-v4.4.3-a9 <my_dir>                         # Create a new dbt environment with the version fddaq-v4.4.3-a9
     source <my_dir>/env.sh                                      # Source the dbt environment


* If you just want to analyze the already generated ``ROOT`` files this requirement becomes optional and you can create a python virtual environment with:

  .. code-block:: bash
  
     python3 -m venv /path/to/new/virtual/environment      # Create a new virtual environment
     source /path/to/new/virtual/environment/bin/activate  # Activate the environment
  
    
  In order to have access to the ``ROOT`` library you need to have it sourced in your environment. Add these lines at the end of the ``bin/activate`` file:

  .. code-block:: bash
  
     source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh
     export JUPYTER_CONFIG_DIR=$VIRTUAL_ENV


----------------------------------------------------
2.- Install packages needed for the library to run
----------------------------------------------------

Once you have your environment sourced (``source env.sh`` or ``source bin/activate``) you can install the packages needed for the library with:

.. code-block:: bash

   cd /path/to/waffles # Go to the waffles main folder
   pip install -r docs/requirements.txt # Install the requirements for the library


----------------------
3.- Access the data  
----------------------

The PDS data is stored in the ``eos`` CERN storage system. The paths would be useful:

* **TUTORIAL INFO**: ``/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/0_TUTORIAL`` 
* **RUCIO PATHS**: ``/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`` 
* **RAW ROOT FILES**: ``/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root``


.. admonition:: **Preferred work-flow**
   
   Convert the ``hdf5`` files to ``ROOT`` files using ``00_HDF5toROOT`` scripts. (For using the bash script you need to have the cpp tools previously compiled, see ``cpp_utils`` folder).
   This will generate the ``rucio`` paths if they are not already created and store them in the ``1_rucio_paths`` folder. The output will be stored in the ``2_daq_root`` folder.

Have a look at the examples tab for more information on how to run waffles.

Depending on the scope of your analysis you may need a different machine:
   
* **OFFLINE** analysis: log-in to lxplus with your CERN user: ``ssh CERNuser@lxplus.cern.ch``
* **ONLINE** + **DATA TAKING**: log-in to np04-srv-015 with your CERN user: ``ssh CERNuser@np04-srv-015``

Inside the daq machines, to access the network we need to do: ``source ~np04daq/bin/web_proxy.sh``

---------------------------------------------------------------------------------------------------------------------------------------------

**INDICES AND TABLES**
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`