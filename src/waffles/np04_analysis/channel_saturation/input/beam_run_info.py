

A = { 'Name' : 'A',
    'Beam period' : 1,
    'Collimator' : 20,
    'PID' : 'No',
    'Polarity' : '+1',
    'Runs' : {1 : 27338, 2 : 27355, 3 : 27361, 5 : 27367, 7 : 27374}}

B = { 'Name' : 'B',
    'Beam period' : 1,
    'Collimator' : 20,
    'PID' : 'No',
    'Polarity' : '-1',
    'Runs' : {1 : 27347, 2 : 27358, 3 : 27353, 5 : 27371, 7 : 27378}} #changed 27351  27352 (not found) with 27353 (10 collimator)


run_set_list = [A, B]
run_set_dict = {entry['Name']: entry for entry in run_set_list}
run_set_list_A = [A]
run_set_list_B = [B]

