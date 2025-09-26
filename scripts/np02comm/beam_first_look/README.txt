Workflow:
Choose the run you want to process;
Update the file pds_beam_run_infos.csv; 
Run in sequence:
python beam_saturated_check_cathode.py --run RUN_NUMBER #creates the csv file for the cathode
python beam_saturated_check_membrane.py --run RUN_NUMBER #creates the csv file for the membranes
create_dataframe.py --runs RUN_NUMBERS #creates the pandas data frame --- serve a qualcosa??

