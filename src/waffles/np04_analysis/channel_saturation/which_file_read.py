from tools_analysis import *

pickle_folder = "/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles"
output_folder = "/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output"
output_filename = "reading_pickles_info"

data = {}

for run_set in run_set_list:
    set_dic = {}
    print(f"\n\n ----------- SET {run_set['Name']} ----------- \n")
    for energy, run in run_set['Runs'].items():
        print(f"\n--- RUN {run} ({energy} GeV) ---")
        files = os.listdir(f"{pickle_folder}/set_{run_set['Name']}/run_{run}")
        pattern = re.compile(r"self_(\d+)\.pkl")
        indices = sorted(int(pattern.match(f).group(1)) for f in files if pattern.match(f))

        if indices:
            run_dic = {'Files read' : len(indices), 'Min index' : min(indices), 'Max index' : max(indices), 'Missing index' : [i for i in range(min(indices), max(indices) + 1) if i not in indices]}
            for key, value in run_dic.items():
                print(f"{key} : {value}")
        else:
            run_dic = {}
            print("No file self_<index>.pkl found")

        set_dic[run] = run_dic
    
    data[run_set['Name']] = set_dic
    
with open(f"{output_folder}/{output_filename}.json", "w") as file:
    json.dump(data, file, indent=4)