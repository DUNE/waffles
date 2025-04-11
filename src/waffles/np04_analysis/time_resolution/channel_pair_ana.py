# --- IMPORTS -------------------------------------------------------
import yaml
import numpy as np
from ROOT import TH1F, TH2F, TFile
import time_alignment as ta

# --- VARIABLES -----------------------------------------------------
ref_ch = 10401
com_ch = 10403
root_directory_name = "run_34081_half_amplitude_filt_0/thr_0p5/"
h2_nbins = 200

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    output_folder = config_variables.get("output_folder")

    time_alligner = ta.TimeAligner(ref_ch, com_ch)
    time_alligner.set_quantities(output_folder, root_directory_name)
    time_alligner.allign_events()

    t0_diff = time_alligner.ref_ch.t0s - time_alligner.com_ch.t0s
    print("Histogram of t0 differences")


    # --- PLOTTING ---------------------------------------------------
    # t0 differences distribution ------------------------------------
    h_t0_diff = TH1F("h_t0_diff", "Reference-Comparison time difference;t0 [ticks=16ns];Counts",
                     200, np.min(t0_diff), np.max(t0_diff))
    for diff in t0_diff:
        h_t0_diff.Fill(diff)

    # Com vs Ref pes -------------------------------------------------
    counts, xedges, yedges = np.histogram2d(time_alligner.com_ch.pes, time_alligner.ref_ch.pes,
                                            bins=(h2_nbins,h2_nbins),
                                            range=[[500,1000],[500,1000]])
                                            # range=[[np.min(time_alligner.com_ch.pes), np.max(time_alligner.com_ch.pes)],
                                            #        [np.min(time_alligner.ref_ch.pes), np.max(time_alligner.ref_ch.pes)]])
        
    h2_pes = TH2F("h2_pes", "Comparison vs Reference Channel p.e.s;#pe_{ref};#pe_{com}",
                  h2_nbins, 500, 1000, h2_nbins, 500, 1000)
                  # h2_nbins, np.min(time_alligner.ref_ch.pes), np.max(time_alligner.ref_ch.pes),
                  # h2_nbins, np.min(time_alligner.com_ch.pes), np.max(time_alligner.com_ch.pes))

    for i in range(h2_nbins):
        for j in range(h2_nbins):
            h2_pes.SetBinContent(i+1, j+1, counts[i, j])

    # Com vs Ref t0 --------------------------------------------------
    counts, xedges, yedges = np.histogram2d(time_alligner.com_ch.t0s, time_alligner.ref_ch.t0s,
                                            bins=(h2_nbins,h2_nbins),
                                            range=[[np.min(time_alligner.com_ch.t0s), np.max(time_alligner.com_ch.t0s)],
                                                   [np.min(time_alligner.ref_ch.t0s), np.max(time_alligner.ref_ch.t0s)]])

    h2_t0 = TH2F("h2_t0", "Comparison vs Reference Channel t0;t0_{ref} [ticks=16ns];t0_{com} [ticks=16ns]",
                 h2_nbins, np.min(time_alligner.ref_ch.t0s), np.max(time_alligner.ref_ch.t0s),
                 h2_nbins, np.min(time_alligner.com_ch.t0s), np.max(time_alligner.com_ch.t0s))

    for i in range(h2_nbins):
        for j in range(h2_nbins):
            h2_t0.SetBinContent(i+1, j+1, counts[i, j])


    # --- WRITING ----------------------------------------------------
    out_root_file_name = output_folder+f"ch_{ref_ch}_vs_ch_{com_ch}_time_alignment.root"
    out_root_file = TFile(out_root_file_name, "RECREATE")
    h_t0_diff.Write()
    h2_pes.Write()
    h2_t0.Write()
    out_root_file.Close()
