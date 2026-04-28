import numpy as np
import pandas as pd
from ROOT import TH1D, TEfficiency, TGraphErrors, TF1
import waffles
import plotly.subplots as psu
from waffles.plotting.plot import plot_CalibrationHistogram


def allow_channel_wfs(waveform: waffles.Waveform, channel: int) -> bool:
    return waveform.endpoint == (channel//100) and waveform.channel == (channel%100)


def get_xmax_in_range(h: TH1D, low: int, up: int) -> int:
    """
    Get the x value of the maximum bin in a given range.
    """
    y_max = -np.inf
    max_bin = -1
    for i in range(low, up+1):
        if h.GetBinContent(i) > y_max:
            y_max = h.GetBinContent(i)
            max_bin = i
    return max_bin
    
def get_efficiency_at(he: TEfficiency, npe: float) -> tuple[float, float, float]:
    """
    Get the efficiency at npe-photoelectrons and its uncertainties.
    """
    bin_x   = he.GetTotalHistogram().FindBin(npe)
    eff     = he.GetEfficiency(bin_x)
    err_up  = he.GetEfficiencyErrorUp(bin_x)
    err_low = he.GetEfficiencyErrorLow(bin_x)

    return eff, err_up, err_low
    
def find_50efficiency(he_STEfficiency: TEfficiency) -> float:
    """

    """
    h_total = he_STEfficiency.GetTotalHistogram()
    effs = np.array([he_STEfficiency.GetEfficiency(i+1) for i in range(h_total.GetNbinsX())])
    n_bins = h_total.GetNbinsX()

    bin_centers = np.array([h_total.GetBinCenter(i + 1) for i in range(n_bins)])

    # Find first window of 5 bins where values are monotonic increasing and cross 0.5
    above_half = effs >= 0.5
    cross_indices = np.flatnonzero(np.diff(above_half.astype(int)) == 1)

    # Check if a clean monotonic region exists
    for i in cross_indices:
        j0 = max(0, i - 2)
        j1 = min(len(effs), i + 3)
        window = effs[j0:j1]
        if len(window) >= 5:
            if np.all(np.diff(window) > 0):
                return float(bin_centers[i + 1])  # use center after crossing

    # Fallback: if no smooth crossing, use first simple crossing
    if cross_indices.size > 0:
        return float(bin_centers[cross_indices[0] + 1])

    return np.nan

def dataframe_columns_to_tgrapherrors(df: pd.DataFrame, x_column: str, y_column: str,
                                      err_x_column: str="", err_y_column: str="", gr_name: str="graph") -> TGraphErrors:
    """
    Convert dataframe columns to a TGraphErrors.
    """
    x = np.array(df[x_column].values).astype(float)
    y = np.array(df[y_column].values).astype(float)
    if err_x_column == "":
        err_x = np.zeros_like(x)
    else:
        err_x = np.array(df[err_x_column].values).astype(float)
    if err_y_column == "":
        err_y = np.zeros_like(y)
    else:
        err_y = np.array(df[err_y_column].values).astype(float)

    print(x,type(x))
    print(y,type(y))
    print(err_x,type(err_x))
    print(err_y,type(err_y))
    gr = TGraphErrors(len(x), x, y, err_x, err_y)
    gr.SetName(gr_name)
    gr.SetTitle(f"{gr_name};{x_column};{y_column}")

    return gr

def fit_thrPE_vs_thrSet(out_df: pd.DataFrame, thrPE_column: str, gr_name: str) -> tuple[TGraphErrors, float, float]:
    """
    Fit the threshold in PE vs the set threshold.
    """
    err_thrPE_column = "Err"+thrPE_column
    # Drop lines where ThresholdFit is NaN
    df = out_df.dropna(subset=[thrPE_column])

    gr = dataframe_columns_to_tgrapherrors(
        df,
        x_column="ThresholdSet",
        y_column=thrPE_column,
        err_y_column=err_thrPE_column,
        gr_name=gr_name
    )

    # Fit with a linear function
    f_lin = TF1("f_lin", "pol1", gr.GetXaxis().GetXmin(), gr.GetXaxis().GetXmax())
    gr.Fit(f_lin, "R")
    offset = f_lin.GetParameter(0)
    slope = f_lin.GetParameter(1)

    return gr, offset, slope

def plot_charge_histo_fit(self_trigger, charge_histo) -> None:
    fig = psu.make_subplots(
        rows=1,
        cols=1,
        subplot_titles="",
        # shared_xaxes=kwargs.pop("shared_xaxes", False),
        # shared_yaxes=kwargs.pop("shared_yaxes", False)
    )
    plot_CalibrationHistogram(
        charge_histo,
        figure=fig,
        row=1, col=1,
        plot_fits=True,
        name=f"run_{self_trigger.run}_led_{self_trigger.led}_ch_{self_trigger.ch_sipm}_charge_histo_fit",
        showfitlabels=False,
    )

    width =1000
    height = 800

    fig.update_layout(title=f"{self_trigger.wafflesSNR}", template="plotly_white",
                      width=width, height=height, showlegend=True)
    fig.update_annotations(
        font=dict(size=14),
        align="center",
    )
    fig.write_image(f"{self_trigger.ana_folder}{self_trigger.fit_type}/run_{self_trigger.run}_led_{self_trigger.led}_ch_{self_trigger.ch_sipm}_charge_histo_fit.png")
