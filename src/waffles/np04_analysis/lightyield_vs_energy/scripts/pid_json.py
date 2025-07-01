import json

particle_dic = { 'e' : { 1 : {'tof' : {'min': 0, 'max': 105}, 'c0' : 1, 'c1' : None}, 
                    2 : {'tof' : {'min': 0, 'max': 105}, 'c0' : 1, 'c1' : None}, 
                    3 : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1}, 
                    5 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, #boo??
                    7 : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1}},
            'mu_pi' : {1 : {'tof' : {'min': 0, 'max': 110}, 'c0' : 0, 'c1' : None}, 
                    2 : {'tof' : {'min': 0, 'max': 103}, 'c0' : 0, 'c1' : None}, 
                    3 : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 1}, 
                    5 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, #boo??
                    7 : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1}},
            'K' : {1 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, 
                    2 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, 
                    3 : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0}, 
                    5 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, #boo??
                    7 : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 1}},
            'p' : { 1 : {'tof' : {'min': 110, 'max': 160}, 'c0' : 0, 'c1' : None}, 
                    2 : {'tof' : {'min': 103, 'max': 160}, 'c0' : 0, 'c1' : None}, 
                    3 : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0}, 
                    5 : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}, #boo??
                    7 : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0}}
        }

energy_dic = { 1 : { 'e' : {'tof' : {'min': 0, 'max': 105}, 'c0' : 1, 'c1' : None},
                    'mu_pi' : {'tof' : {'min': 0, 'max': 110}, 'c0' : 0, 'c1' : None},
                    'K' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None},
                    'p' : {'tof' : {'min': 110, 'max': 160}, 'c0' : 0, 'c1' : None}},
              2 : { 'e' : {'tof' : {'min': 0, 'max': 105}, 'c0' : 1, 'c1' : None},
                    'mu_pi' : {'tof' : {'min': 0, 'max': 103}, 'c0' : 0, 'c1' : None},
                    'K' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None},
                    'p' : {'tof' : {'min': 103, 'max': 160}, 'c0' : 0, 'c1' : None}},
              3 : { 'e' : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1},
                    'mu_pi' : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 1},
                    'K' : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0},
                    'p' : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0}},
              5 : { 'e' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None},
                    'mu_pi' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None},
                    'K' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None},
                    'p' : {'tof' : {'min': None, 'max': None}, 'c0' : None, 'c1' : None}},
              7 : { 'e' : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1},
                    'mu_pi' : {'tof' : {'min': None, 'max': None}, 'c0' : 1, 'c1' : 1},
                    'K' : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 1},
                    'p' : {'tof' : {'min': None, 'max': None}, 'c0' : 0, 'c1' : 0}}
        }
    
with open("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/pid_1.json", "w", encoding="utf-8") as file:
    json.dump(particle_dic, file, ensure_ascii=False, indent=4)
    
with open("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/pid_2.json", "w", encoding="utf-8") as file:
    json.dump(energy_dic, file, ensure_ascii=False, indent=4)