import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
import waffles.input_output.hdf5_structured as reader
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TH1D, TFile
import mplhep

homedir =  "/afs/cern.ch/user/f/fegalizz/"
filename =  "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/files/Run_31602_ChSiPM_11101_ChST_11100_structured.hdf5"

if __name__ == "__main__":
    print("Reading file ", filename)
    wfset = reader.load_structured_waveformset(filename)
    ch_sipm = filename.split("ChSiPM_")[-1].split("_")[0]
    ch_st = filename.split("ChST_")[-1].split("_")[0]
    st = self_trigger.SelfTrigger(ch_sipm=int(ch_sipm),
                                          ch_st=int(ch_st),
                                          wf_set=wfset)
    st.create_wfs()
    
    fig = plt.figure(figsize=(10, 8))
    h_st = st.create_self_trigger_distribution()
    mplhep.histplot(h_st, label="Self Trigger Distribution", color="blue")
    plt.xlabel("Ticks")
    plt.ylabel("Counts")
    plt.xlim(200, 280)
    plt.savefig(homedir+"NI_self_trigger_distribution.png")


   
    h_trgPerWf = TH1D("h_trgPerWf", "h_trgPerWf;#TRG;Counts", 10, 0, 10)
    h_trgDist  = TH1D("h_trgDist",  "h_trgDist;#Delta T [ticks];Counts", 1024, 0, 1024)
    h_1sttrgDist = TH1D("h_1sttrgDist", "h_1sttrgDist;#Delta T [ticks];Counts", 1024, 0, 1024)

    for wf in st.wfs_st:
        n_trig = np.sum(wf.adcs) 
        h_trgPerWf.Fill(n_trig)
        if n_trig != 0:
            h_1sttrgDist.Fill(np.flatnonzero(wf.adcs)[0])
        if n_trig > 1:
            trig_idxs = np.flatnonzero(wf.adcs)
            for i in range(len(trig_idxs)-1):
                h_trgDist.Fill(trig_idxs[i+1]-trig_idxs[i])
    
    # Trigger Rate Calculation ------------------------------------------------
    n_triggers = np.sum(h_st.values())
    n_wfs = len(st.wfs_st)
    full_trigger_rate = (n_triggers * 10**9) / (n_wfs * len(st.wfs_st[0].adcs) * 16.)

    n_zero_triggers = h_trgPerWf.GetBinContent(1)
    independent_trigger_rate = -np.log(n_zero_triggers/n_wfs) / (len(st.wfs_st[0].adcs) * 16.) * 10**9

    # Prints ------------------------------------------------------------------
    print(f"Trigger Rate: {full_trigger_rate} Hz")
    print(f"Zero trigger in {n_zero_triggers}/{n_wfs} wfs -> Rate: {independent_trigger_rate} Hz")

    # Save histograms to ROOT file ---------------------------------------------
    out_file = TFile("ni.root", "RECREATE")
    h_trgDist.Write()
    h_trgPerWf.Write()
    h_1sttrgDist.Write()
    out_file.Close()
