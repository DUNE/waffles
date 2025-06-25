#classs to compare two channels

import numpy as np
import uproot

class ChTimeAlignment:
    def __init__(self, channel):
        self.channel = channel
        self.t0s = np.array([])
        self.pes = np.array([])
        self.tss = np.array([])

    def set_quantities(self, root_file_name, root_directory):
        in_root_file_name = root_file_name
        root_file = uproot.open(in_root_file_name)
        try:
            directory = root_file[root_directory]
        except:
            print(f"Directory {root_directory} not found in {in_root_file_name}")
            return
        tree = directory["time_resolution"]
        branches = tree.keys()
        arrays = tree.arrays(branches, library="np")

        self.t0s = arrays["t0"]
        self.pes = arrays["pe"]
        self.tss = arrays["timestamp"]



# --- CLASS DEFINITION ------------------------------------------------
class TimeAligner:
    def __init__(self,
                 ref_ch: int,
                 com_ch: int) -> None:
        """
        This class is used to estimate the time alignment between two channels.
        """
        self.ref_ch = ChTimeAlignment(ref_ch)
        self.com_ch = ChTimeAlignment(com_ch)

    def set_quantities(self, ref_file, com_file, root_directory):
        """
        Args:
        - folder_path: path to the root file
        """
        self.ref_ch.set_quantities(ref_file, root_directory)
        self.com_ch.set_quantities(com_file, root_directory)
       
    def allign_events(self):
        """
        Align events from two channels.
        """
        print("Aligning events")
        n_ref_evts = len(self.ref_ch.t0s)
        n_com_evts = len(self.com_ch.t0s)
        
        # Sort arrays
        sorted_indices = np.argsort(self.ref_ch.tss)
        self.ref_ch.t0s, self.ref_ch.pes, self.ref_ch.tss = self.ref_ch.t0s[sorted_indices], self.ref_ch.pes[sorted_indices], self.ref_ch.tss[sorted_indices]

        sorted_indices = np.argsort(self.com_ch.tss)
        self.com_ch.t0s, self.com_ch.pes, self.com_ch.tss = self.com_ch.t0s[sorted_indices], self.com_ch.pes[sorted_indices], self.com_ch.tss[sorted_indices]
        

        common_ts = np.intersect1d(self.ref_ch.tss, self.com_ch.tss)

        mask_ref = np.isin(self.ref_ch.tss, common_ts)
        mask_com = np.isin(self.com_ch.tss, common_ts)

        self.ref_ch.t0s = self.ref_ch.t0s[mask_ref]
        self.ref_ch.pes = self.ref_ch.pes[mask_ref]
        self.ref_ch.tss = self.ref_ch.tss[mask_ref]

        self.com_ch.t0s = self.com_ch.t0s[mask_com]
        self.com_ch.pes = self.com_ch.pes[mask_com]
        self.com_ch.tss = self.com_ch.tss[mask_com]

        nf_ref_evts = len(self.ref_ch.t0s)
        nf_com_evts = len(self.com_ch.t0s)

        print(f"Ref evts {n_ref_evts} -> {nf_ref_evts} \nCom evts {n_com_evts} -> {nf_com_evts}")
