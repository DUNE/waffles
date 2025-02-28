
#classs to compare two channels

def calculate_t0_differences(self) -> np.array:
        """
        Calculate differences in t0 values for wf objects with matching ts values and selection==True.
        Args:
        
        Returns:
            np.ndarray: Array of t0 differences for matching ts values.
        """
        
        # Filter wf objects where selection is True
        wf1_filtered = {wf.timestamp: wf.t0 for wf in self.wfs if wf.time_resolution_selection}
        wf2_filtered = {wf.timestamp: wf.t0 for wf in self.com_wfs if wf.time_resolution_selection}
        
        # Find common ts values and calculate t0 differences
        common_ts = set(wf1_filtered.keys()).intersection(wf2_filtered.keys())
        t0_differences = [wf1_filtered[ts] - wf2_filtered[ts] for ts in common_ts]
        
        return np.array(t0_differences)
