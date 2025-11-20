import pickle
import numpy as np
from waffles.np04_analysis.time_resolution.imports import *
from waffles.np04_utils.utils import get_np04_channel_mapping

prepulse_ticks = 50
folder = "/Users/federico/CERN/PDHD/TimeResolution/FromGabriel/"
file_name = folder+"data.pkl"
method = "amplitude"
relative_thr = 0.5

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    with open(file_name, "rb") as file:
        wfsets = pickle.load(file)

    wfset_hv = wfsets[0]
    # hvs = np.arange(len(wfset_hv)).astype(np.int32)
    hv = 0
    mapping_df = get_np04_channel_mapping(version="old")
    daphne_channels = mapping_df['daphne_ch'].values + 100*mapping_df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, mapping_df['offline_ch']))

    tr_hv = []
    iter = 0

    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    run_info_file = config_variables.get("run_info_file")
    ana_folder = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    os.makedirs(raw_ana_folder, exist_ok=True)
    

    for wfset_ch in wfset_hv:
        daphne_ch = int(wfset_ch.waveforms[0].endpoint*100 + wfset_ch.waveforms[0].channel)
        offline_ch = daphne_to_offline[daphne_ch]
        out_root_file_name = raw_ana_folder+f"Run_{hv}_DaphneCh_{daphne_ch}_OfflineCh_{offline_ch}_time_resolution.root"
        root_file = TFile(out_root_file_name, "RECREATE")

        tr_ch = tr.TimeResolution(wf_set=wfset_ch)
        tr_ch.set_analysis_parameters(
            prepulse_ticks=prepulse_ticks,
            postpulse_ticks=100,
            ch=daphne_ch,
            int_low=20,
            int_up=200,
            spe_charge=0.16,
            spe_ampl=4,
            min_pes=1,
            baseline_rms=0.5
        )
        tr_ch.create_wfs()
        tr_ch.select_time_resolution_wfs()
        rel_thr = str(relative_thr).replace(".", "p")
        root_file.mkdir(f"{method}_filt_0_thr_{rel_thr}")
        root_file.cd(f"{method}_filt_0_thr_{rel_thr}")

        t0s, pes, tss = tr_ch.set_wfs_t0(method=method, relative_thr=relative_thr)
        if iter == 0:
            trefs = tss.copy() 
            iter += 1
        else:
            print(type(tss[0]), type(trefs), type(t0s[0]))
            t0s = (tss - trefs).astype(np.int64).astype(np.float64) + np.array(t0s).astype(np.float64)
        print(len(t0s), " selected waveforms for channel ", daphne_ch)

        t = TTree("time_resolution", "time_resolution")
        t0 = np.zeros(1, dtype=np.float64)
        pe = np.zeros(1, dtype=np.float64)
        ts = np.zeros(1, dtype=np.float64)
        t.Branch("t0", t0, "t0/D")
        t.Branch("pe", pe, "pe/D")
        t.Branch("timestamp", ts, "timestamp/D")

        for i in range(len(t0s)):
            t0[0] = t0s[i]
            pe[0] = pes[i]
            ts[0] = tss[i]
            t.Fill()

        t.Write()

        root_file.Close()

        


        # tr_hv.append(tr_ch)

    
