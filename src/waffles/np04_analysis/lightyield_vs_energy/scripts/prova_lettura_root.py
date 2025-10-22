import uproot

def explore_root(filename):
    with uproot.open(filename) as f:
        print(f"📂 File: {filename}")
        
        def explore_group(group, path=""):
            for key in group.keys():
                obj = group[key]
                fullpath = f"{path}{key}"

                # Caso TTree
                if isinstance(obj, uproot.behaviors.TTree.TTree):
                    print(f"🌳 TTree: {fullpath} con {obj.num_entries} entries")
                    print("   Branch disponibili:", list(obj.keys())[:10], "...")
                
                # Caso istogramma (TH1, TH2, ecc.)
                elif hasattr(obj, "to_numpy"):
                    try:
                        values, edges = obj.to_numpy()
                        print(f"📊 Istogramma: {fullpath}, somma contenuti = {values.sum()}")
                    except Exception as e:
                        print(f"📊 Istogramma: {fullpath}, ma errore nella lettura: {e}")
                
                # Caso directory
                elif isinstance(obj, uproot.reading.ReadOnlyDirectory):
                    print(f"📁 Directory: {fullpath}")
                    explore_group(obj, path=fullpath + "/")
                
                # Caso generico
                else:
                    print(f"❓ Oggetto non riconosciuto: {fullpath}, tipo {type(obj)}")
        
        explore_group(f)

# Uso:
filename = "/afs/cern.ch/user/a/anbalbon/pid_larsoft/beamevent_hist.root"
# filename = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/beam_example/data/028676_20files_beam.root"

explore_root(filename)
