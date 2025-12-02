import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors, TMultiGraph

# channel = 10403
channel = 10441
# channel = 10903
# channel = 10945
# channel = 11121
# channel = 11221
merged = True
if merged:
    merged = "_merged"
else:
    merged = ""
result_file = f"~/CERN/PDHD/Self_trigger/analysis/Ch_{channel}/SelfTrigger_Results_Ch_{channel}{merged}.csv"
out_file_name = f"~/PhD/plotter/projects/NP04_PDS_article/SelfTrigger/SelfTrigger_Results_Graphs_Ch_{channel}{merged}.root"
# result_file   = f"~/CERN/M1/cb_nov_24/cb/Daphne_DAQ/SelfTrigger/Ch_10403/SelfTrigger_Results_Ch_{channel}.csv"
# out_file_name = f"~/CERN/M1/cb_nov_24/cb/Daphne_DAQ/SelfTrigger/SelfTrigger_Results_Graphs_Ch_{channel}{merged}.root"

def column_where_to_array(df, column_name, condition_column, condition_value):
    """
    Extracts a numpy array from a specified column in a DataFrame where another column meets a given condition.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to extract data from.
    condition_column (str): The name of the column to apply the condition on.
    condition_value: The value that the condition_column should match.

    Returns:
    np.ndarray: A numpy array containing the values from column_name where condition_column equals condition_value.
    """
    filtered_values = df[df[condition_column] == condition_value][column_name]
    return np.array(filtered_values).astype(float)

def arrays_to_formatted_graph(x, y, ex=None, ey=None, name_and_title="graph", x_axis_title="", y_axis_title=""):
    """
    Creates a TGraphErrors object from numpy arrays with formatted names and titles.

    Parameters:
    x (np.ndarray): Array of x values.
    y (np.ndarray): Array of y values.
    ex (np.ndarray or None): Array of errors in x values. If None, defaults to zeros.
    ey (np.ndarray or None): Array of errors in y values. If None, defaults to zeros.
    name (str): Base name for the graph.
    title (str): Base title for the graph.

    Returns:
    TGraphErrors: A ROOT TGraphErrors object with the provided data and formatted name/title.
    """
    if ex is None:
        ex = np.zeros_like(x)
    if ey is None:
        ey = np.zeros_like(y)

    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(ex) & ~np.isnan(ey)
    x = x[mask]
    y = y[mask]
    ex = ex[mask]
    ey = ey[mask]

    if len(x) == 0 or len(y) == 0:
        return None

    graph = TGraphErrors(len(x), x, y, ex, ey)
    graph.SetName(name_and_title)
    graph.SetTitle(name_and_title)
    graph.GetXaxis().SetTitle(x_axis_title)
    graph.GetYaxis().SetTitle(y_axis_title)
    return graph

def dataframe_columns_to_tgraph(
            df,
            condition,
            condition_value,
            x_col,
            y_col, 
            ex_col=None,
            ey_col=None,
            name_and_title="graph",
            x_axis_title="",
            y_axis_title=""
):
    """
    Creates a TGraphErrors from specified columns of a DataFrame where a condition is met.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    condition (str): The column name to apply the condition on.
    condition_value: The value that the condition column should match.
    x_col (str): The column name for x values.
    y_col (str): The column name for y values.
    ex_col (str or None): The column name for x errors. If None, defaults to zeros.
    ey_col (str or None): The column name for y errors. If None, defaults to zeros.
    name_and_title (str): Base name and title for the graph.

    Returns:
    TGraphErrors: A ROOT TGraphErrors object with the provided data and formatted name/title.
    """
    x = column_where_to_array(df, x_col, condition, condition_value)
    y = column_where_to_array(df, y_col, condition, condition_value)

    if ex_col is not None:
        ex = column_where_to_array(df, ex_col, condition, condition_value)
    else:
        ex = np.zeros_like(x)

    if ey_col is not None:
        ey = column_where_to_array(df, ey_col, condition, condition_value)
    else:
        ey = np.zeros_like(y)

    graph = arrays_to_formatted_graph(x, y, ex, ey, name_and_title, x_axis_title, y_axis_title)
    return graph

def g_normalize_trigger_rate_2pe(g_trigger_rate):
    """
    
    """
    rate_at_2pe = g_trigger_rate.Eval(2.0)
    for i in range(g_trigger_rate.GetN()):
        y = g_trigger_rate.GetY()[i]
        g_trigger_rate.SetPoint(i, g_trigger_rate.GetX()[i], y / rate_at_2pe)
        g_trigger_rate.SetPointError(i, g_trigger_rate.GetEX()[i], g_trigger_rate.GetEY()[i] / rate_at_2pe)

    return g_trigger_rate


if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    out_root_file = TFile(out_file_name, "RECREATE")
    out_root_file.cd()

    # Convert "LED" column into numpy array of integers
    identifier = "LED"
    its = df_result[identifier].to_numpy(dtype=int)
    its = np.unique(its)
    if merged:
        identifier = "SiPMChannel"
        its = df_result[identifier].to_numpy(dtype=int)
        its = np.unique(its)

    # TMultiGraphs
    gm_ThrFit_ThrCal = TMultiGraph("gm_ThrFit_ThrCal", "Threshold Fit vs Threshold Set;Threshold Set [a.u.];Threshold Fit [p.e.]")
    gm_BkgTrgRate_ThrCal = TMultiGraph("gm_BkgTrgRate_ThrCal", "Background Trigger Rate vs Threshold Set;Threshold Set [a.u.];Background Trigger Rate [Hz]")
    gm_NormBkgTrgRate_ThrCal = TMultiGraph("gm_NormBkgTrgRate_ThrCal", "Normalized Background Trigger Rate vs Threshold Set;Threshold Set [p.e.];Normalized Background Trigger Rate [Hz]")
    gm_TauFit_ThrCal = TMultiGraph("gm_TauFit_ThrCal", "Tau Fit vs Threshold Set;Threshold Set [a.u.];Tau Fit [p.e.]")
    gm_10to90_ThrCal = TMultiGraph("gm_10to90_ThrCal", "10 to 90% Window Upper Edge vs Threshold Set;Threshold Set [a.u.];10 to 90% Window Upper Edge [p.e.]")
    gm_10to90Fit_ThrCal = TMultiGraph("gm_10to90Fit_ThrCal", "10 to 90% Window Upper Edge Fit vs Threshold Set;Threshold Set [a.u.];10 to 90% Window Upper Edge Fit [p.e.]")
    gm_fifty_ThrCal = TMultiGraph("gm_fifty_ThrCal", "Fifty vs Threshold Set;Threshold Set [a.u.];Fifty [p.e.]")
    gm_effAt2pe_ThrCal = TMultiGraph("gm_effAt2pe_ThrCal", "Efficiency at 2 PE vs Threshold Set;Threshold Set [a.u.];Efficiency at 2 PE [%]")
    gm_effAt3pe_ThrCal = TMultiGraph("gm_effAt3pe_ThrCal", "Efficiency at 3 PE vs Threshold Set;Threshold Set [a.u.];Efficiency at 3 PE [%]")
    gm_effAt2peFit_ThrCal = TMultiGraph("gm_effAt2peFit_ThrCal", "Efficiency at 2 PE Fit vs Threshold Set;Threshold Set [a.u.];Efficiency at 2 PE Fit [%]")
    gm_effAt3peFit_ThrCal = TMultiGraph("gm_effAt3peFit_ThrCal", "Efficiency at 3 PE Fit vs Threshold Set;Threshold Set [a.u.];Efficiency at 3 PE Fit [%]")


    for it in its:
        # ThresholdFit vs FiftyCalibrated
        print("Creating ThresholdFit vs FiftyCalibrated graph for it =", it)
        g_ThrFit_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "ThresholdFit", None, "ErrThresholdFit", f"g_ThrFit_ThrCal_it_{it}", "Threshold Set [a.u.]", "Threshold Fit [p.e.]")
        if not g_ThrFit_ThrCal == None:
            g_ThrFit_ThrCal.Write()
            gm_ThrFit_ThrCal.Add(g_ThrFit_ThrCal)

        # BkgTrgRate vs FiftyCalibrated
        g_BkgTrgRate_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "BkgTrgRate", None, "ErrBkgTrgRate", f"g_BkgTrgRate_ThrCal_it_{it}", "Threshold Set [a.u.]", "Background Trigger Rate [Hz]")
        if not g_BkgTrgRate_ThrCal == None:
            g_BkgTrgRate_ThrCal.Write()
            g_NormBkgTrgRate_ThrCal = g_normalize_trigger_rate_2pe(g_BkgTrgRate_ThrCal.Clone(f"g_NormBkgTrgRate_ThrCal_it_{it}"))
            g_NormBkgTrgRate_ThrCal.Write()
            gm_NormBkgTrgRate_ThrCal.Add(g_NormBkgTrgRate_ThrCal)
            gm_BkgTrgRate_ThrCal.Add(g_BkgTrgRate_ThrCal)
        
        # TauFit vs FiftyCalibrated
        g_TauFit_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "TauFit", None, "ErrTauFit", f"g_TauFit_ThrCal_it_{it}", "Threshold Set [a.u.]", "Tau Fit [p.e.]")
        if not g_TauFit_ThrCal == None:
            g_TauFit_ThrCal.Write()
            gm_TauFit_ThrCal.Add(g_TauFit_ThrCal)
        
        # 10to90 vs FiftyCalibrated
        g_10to90_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "10to90", None, None, f"g_10to90_ThrCal_it_{it}", "Threshold Set [a.u.]", "10 to 90% Window Upper Edge [p.e.]")
        if not g_10to90_ThrCal == None:
            g_10to90_ThrCal.Write()
            gm_10to90_ThrCal.Add(g_10to90_ThrCal)

        # 10to90Fit vs FiftyCalibrated
        g_10to90Fit_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "10to90Fit", None, None, f"g_10to90Fit_ThrCal_it_{it}", "Threshold Set [a.u.]", "10 to 90% Window Upper Edge Fit [p.e.]")
        if not g_10to90Fit_ThrCal == None:
            g_10to90Fit_ThrCal.Write()
            gm_10to90Fit_ThrCal.Add(g_10to90Fit_ThrCal)

        # Fifty vs FiftyCalibrated
        g_fifty_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "FiftyEffPoint", None, None, f"g_fifty_ThrCal_it_{it}", "Threshold Set [a.u.]", "Fifty [p.e.]")
        if not g_fifty_ThrCal == None:
            g_fifty_ThrCal.Write()
            gm_fifty_ThrCal.Add(g_fifty_ThrCal)

        # Efficiency at 2 PE vs FiftyCalibrated
        g_effAt2pe_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "EffAt2PE", None, "ErrEffAt2PE", f"g_effAt2pe_ThrCal_it_{it}", "Threshold Set [a.u.]", "Efficiency at 2 PE [%]")
        if not g_effAt2pe_ThrCal == None:
            g_effAt2pe_ThrCal.Write()
            gm_effAt2pe_ThrCal.Add(g_effAt2pe_ThrCal)

        # Efficiency at 3 PE vs FiftyCalibrated
        g_effAt3pe_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "EffAt3PE", None, "ErrEffAt3PE", f"g_effAt3pe_ThrCal_it_{it}", "Threshold Set [a.u.]", "Efficiency at 3 PE [%]")
        if not g_effAt3pe_ThrCal == None:
            g_effAt3pe_ThrCal.Write()
            gm_effAt3pe_ThrCal.Add(g_effAt3pe_ThrCal)

        # Efficiency at 2 PE Fit vs FiftyCalibrated
        g_effAt2peFit_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "EffAt2PEFit", None, "ErrEffAt2PEFit", f"g_effAt2peFit_ThrCal_it_{it}", "Threshold Set [a.u.]", "Efficiency at 2 PE Fit [%]")
        if not g_effAt2peFit_ThrCal == None:
            g_effAt2peFit_ThrCal.Write()
            gm_effAt2peFit_ThrCal.Add(g_effAt2peFit_ThrCal)

        # Efficiency at 3 PE Fit vs FiftyCalibrated
        g_effAt3peFit_ThrCal = dataframe_columns_to_tgraph(df_result, identifier, it, "FiftyCalibrated", "EffAt3PEFit", None, "ErrEffAt3PEFit", f"g_effAt3peFit_ThrCal_it_{it}", "Threshold Set [a.u.]", "Efficiency at 3 PE Fit [%]")
        if not g_effAt3peFit_ThrCal == None:
            g_effAt3peFit_ThrCal.Write()
            gm_effAt3peFit_ThrCal.Add(g_effAt3peFit_ThrCal)


    # Write TMultiGraphs
    if identifier != "SiPMChannel":
        gm_ThrFit_ThrCal.Write()
        gm_BkgTrgRate_ThrCal.Write()
        gm_NormBkgTrgRate_ThrCal.Write()
        gm_TauFit_ThrCal.Write()
        gm_10to90_ThrCal.Write()
        gm_10to90Fit_ThrCal.Write()
        gm_fifty_ThrCal.Write()
        gm_effAt2pe_ThrCal.Write()
        gm_effAt3pe_ThrCal.Write()
        gm_effAt2peFit_ThrCal.Write()
        gm_effAt3peFit_ThrCal.Write()


    print("All graphs written to", out_file_name)
    out_root_file.Close()
