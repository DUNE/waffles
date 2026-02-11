# Open a root file, upload two tgraphs, and cross-correlate them

import ROOT
from ROOT import TFile, TGraph
import numpy as np


if __name__ == "__main__":
    root_file = TFile("./TemplatePersistence.root", "READ")
    g_template = root_file.Get("g_template;1")
    g_wf = root_file.Get("g_single_wfs;1")

    # From tgraphs to numpy arrays
    y_template = g_template.GetY()
    y_wf = g_wf.GetY()

    y_template = np.array(y_template)[130:163].astype(float)
    y_wf = np.array(y_wf)[::2]

    corr = np.convolve(y_wf, y_template, mode='same')

    # Create a new TGraph for the cross-correlation
    x_corr = np.arange(len(corr)).astype(float)
    g_corr = TGraph(len(x_corr), x_corr, corr)
    g_corr.SetName("g_corr")
    g_corr.SetTitle("g_corr")

    out_root_file = TFile("./TemplateWFXcorr.root", "RECREATE")
    out_root_file.cd()
    g_template.Write()
    g_wf.Write()
    g_corr.Write()
    out_root_file.Close()
    root_file.Close()
