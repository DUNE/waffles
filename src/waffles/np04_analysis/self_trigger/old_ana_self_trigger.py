import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
import waffles.input_output.hdf5_structured as reader
import matplotlib.pyplot as plt
from ROOT import TFile, TF1
import mplhep
import numpy as np

homedir =  "/afs/cern.ch/user/f/fegalizz/"
filename =  "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/files/Run_31990_ChSiPM_11221_ChST_11220_structured.hdf5"


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
    x_model, y_model = st.fit_self_trigger_distribution()
    mplhep.histplot(h_st, label="Self Trigger Distribution", color="blue")
    plt.plot(x_model, y_model, label="Gaus Fit", color="red")
    plt.xlabel("Ticks")
    plt.ylabel("Counts")
    plt.savefig(homedir+"self_trigger_distribution.png")
    
    fig = plt.figure(figsize=(10, 8))
    mplhep.histplot(h_st, label="Self Trigger Distribution", color="blue")
    plt.plot(x_model, y_model, label="Gaus Fit", color="red")
    plt.xlabel("Ticks")
    plt.ylabel("Counts")
    plt.xlim(200, 280)
    plt.savefig(homedir+"self_trigger_distribution_zoom.png")
    
    fig = plt.figure(figsize=(10, 8))
    h_total, h_passed = st.create_efficiency_histos("he_efficiency")
    plt.savefig(homedir+"efficiency.png")
    
    f_sigmoid = TF1("f_sigmoid", "[2]/(1+exp(([0]-x)/[1]))", -2, 7)
    he_efficiency = st.fit_efficiency(f_sigmoid=f_sigmoid)
       

    # Analysis on selected waveforms ---------------------------------------------
    st.select_waveforms()

    h_st = st.create_self_trigger_distribution()
    x_model, y_model = st.fit_self_trigger_distribution()
    
    fig = plt.figure(figsize=(10, 8))
    h_total, h_passed = st.create_efficiency_histos("he_efficiency_select")
    plt.savefig(homedir+"efficiency_select.png")
    
    he_efficiency_select = st.fit_efficiency(f_sigmoid=f_sigmoid)


    # Outlier studies ---------------------------------------------------------
    st.select_outliers(f_sigmoid=f_sigmoid)
    fig = plt.figure(figsize=(10, 8))
    h_st = st.create_self_trigger_distribution()
    mplhep.histplot(h_st, label="Self Trigger Distribution", color="blue")
    plt.xlabel("Ticks")
    plt.ylabel("Counts")
    plt.savefig(homedir+"outliers_self_trigger_distribution.png")

    fig = plt.figure(figsize=(10, 8))
    h, xedges, yedges = self_trigger.persistence_plot(st.wfs_sipm)
    x, y = np.meshgrid(xedges, yedges)
    pcm = plt.pcolormesh(x, y, np.log10(h))
    plt.xlabel("Ticks")
    plt.ylabel("Counts")
    plt.savefig(homedir+"outliers_persistence_plot.png")


    outfile = TFile("st.root", "RECREATE")
    he_efficiency.Write()
    he_efficiency_select.Write()
    outfile.Close()
