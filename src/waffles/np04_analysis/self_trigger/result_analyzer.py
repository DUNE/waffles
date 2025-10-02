import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors


result_file = "~/CERN/PDHD/Self_trigger/analysis/SelfTrigger_Results_Ch_11121_Selection_True.csv"
out_file_name = "~/PhD/plotter/projects/PosterINSS/SelfTrigger_Results_Ch_11121_Selection_True.root"

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


if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    out_root_file = TFile(out_file_name, "RECREATE")
    out_root_file.cd()

    # Convert "LED" column into numpy array of integers
    leds = df_result["LED"].to_numpy(dtype=int)
    leds = np.unique(leds)

    for led in leds:
        # ThresholdFit vs ThresholdSet
        g_ThrFit_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "ThresholdFit", None, "ErrThresholdFit", f"g_ThrFit_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Threshold Fit [p.e.]")
        g_ThrFit_ThrSet.Write()

        #BkgTrgRate vs ThresholdFit
        g_BkgTrgRate_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "BkgTrgRate", "ErrThresholdFit", None, f"g_BkgTrgRate_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Background Trigger Rate [Hz]")
        g_BkgTrgRate_ThrFit.Write()

        # BkgTrgRate vs ThresholdSet
        g_BkgTrgRate_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "BkgTrgRate", None, None, f"g_BkgTrgRate_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Background Trigger Rate [Hz]")
        g_BkgTrgRate_ThrSet.Write()
        
        #BkgTrgRatePreLED vs ThresholdFit
        g_BkgTrgRatePreLED_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "BkgTrgRatePreLED", "ErrThresholdFit", None, f"g_BkgTrgRatePreLED_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Background Trigger Rate Pre LED [Hz]")
        g_BkgTrgRatePreLED_ThrFit.Write()

        # BkgTrgRatePreLED vs ThresholdSet
        g_BkgTrgRatePreLED_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "BkgTrgRatePreLED", None, None, f"g_BkgTrgRatePreLED_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Background Trigger Rate Pre LED [Hz]")
        g_BkgTrgRatePreLED_ThrSet.Write()

        # MaxAccuracy vs ThresholdFit
        g_MaxAccuracy_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "MaxAccuracy", "ErrThresholdFit", None, f"g_MaxAccuracy_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Max Accuracy [%]")
        g_MaxAccuracy_ThrFit.Write()
        
        # MaxAccuracy vs ThresholdSet
        g_MaxAccuracy_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "MaxAccuracy", None, None, f"g_MaxAccuracy_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Max Accuracy [%]")
        g_MaxAccuracy_ThrSet.Write()

        # FalsePositiveRate vs ThresholdFit
        g_FalsePositiveRate_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "FalsePositiveRate", "ErrThresholdFit", None, f"g_FalsePositiveRate_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "False Positive Rate [%]")
        g_FalsePositiveRate_ThrFit.Write()

        # FalsePositiveRate vs ThresholdSet
        g_FalsePositiveRate_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "FalsePositiveRate", None, None, f"g_FalsePositiveRate_ThrSet_LED_{led}", "Threshold Set [a.u.]", "False Positive Rate [%]")
        g_FalsePositiveRate_ThrSet.Write()

        # TruePositiveRate vs ThresholdFit
        g_TruePositiveRate_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "TruePositiveRate", "ErrThresholdFit", None, f"g_TruePositiveRate_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "True Positive Rate [%]")
        g_TruePositiveRate_ThrFit.Write()
        
        # TruePositiveRate vs ThresholdSet
        g_TruePositiveRate_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "TruePositiveRate", None, None, f"g_TruePositiveRate_ThrSet_LED_{led}", "Threshold Set [a.u.]", "True Positive Rate [%]")
        g_TruePositiveRate_ThrSet.Write()

        # TauFit vs ThresholdFit
        g_TauFit_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "TauFit", "ErrThresholdFit", "ErrTauFit", f"g_TauFit_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Tau Fit [p.e.]")
        g_TauFit_ThrFit.Write()
        
        # TauFit vs ThresholdSet
        g_TauFit_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "TauFit", None, "ErrTauFit", f"g_TauFit_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Tau Fit [p.e.]")
        g_TauFit_ThrSet.Write()

        # Accuracy vs ThresholdFit
        # g_Accuracy_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "Accuracy", "ErrThresholdFit", None, f"g_Accuracy_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Accuracy [%]")
        # g_Accuracy_ThrFit.Write()
        
        # Accuracy vs ThresholdSet
        # g_Accuracy_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "Accuracy", None, None, f"g_Accuracy_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Accuracy [%]")
        # g_Accuracy_ThrSet.Write()
        
        # MaxAccuracy vs ThresholdFit
        g_MaxAccuracy_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "MaxAccuracy", "ErrThresholdFit", None, f"g_MaxAccuracy_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "Max Accuracy [%]")
        g_MaxAccuracy_ThrFit.Write()
    
        # MaxAccuracy vs ThresholdSet
        g_MaxAccuracy_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "MaxAccuracy", None, None, f"g_MaxAccuracy_ThrSet_LED_{led}", "Threshold Set [a.u.]", "Max Accuracy [%]")
        g_MaxAccuracy_ThrSet.Write()

        # FalsePositiveRateAcc vs ThresholdFit
        g_FalsePositiveRateAcc_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "FalsePositiveRateMaxAcc", "ErrThresholdFit", None, f"g_FalsePositiveRateMaxAcc_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "False Positive Rate Acc [%]")
        g_FalsePositiveRateAcc_ThrFit.Write()

        # FalsePositiveRateAcc vs ThresholdSet
        g_FalsePositiveRateAcc_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "FalsePositiveRateMaxAcc", None, None, f"g_FalsePositiveRateMaxAcc_ThrSet_LED_{led}", "Threshold Set [a.u.]", "False Positive Rate Acc [%]")
        g_FalsePositiveRateAcc_ThrSet.Write()

        # TruePositiveRateAcc vs ThresholdFit
        g_TruePositiveRateAcc_ThrFit = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdFit", "TruePositiveRateMaxAcc", "ErrThresholdFit", None, f"g_TruePositiveRateMaxAcc_ThrFit_LED_{led}", "Threshold Fit [p.e.]", "True Positive Rate Acc [%]")
        g_TruePositiveRateAcc_ThrFit.Write()

        # TruePositiveRateAcc vs ThresholdSet
        g_TruePositiveRateAcc_ThrSet = dataframe_columns_to_tgraph(df_result, "LED", led, "ThresholdSet", "TruePositiveRateMaxAcc", None, None, f"g_TruePositiveRateMaxAcc_ThrSet_LED_{led}", "Threshold Set [a.u.]", "True Positive Rate Acc [%]")
        g_TruePositiveRateAcc_ThrSet.Write()


    thresholds = np.array(df_result["ThresholdSet"], dtype=float)
    thresholds = np.unique(thresholds)

    for threshold in thresholds:
        # SigmaTrg vs LED
        g_SigmaTrg_LED = dataframe_columns_to_tgraph(df_result, "ThresholdSet", threshold, "LED", "SigmaTrg", None, "ErrSigmaTrg", f"g_SigmaTrg_LED_Thr_{threshold}", "LED [a.u.]", "Sigma Trigger [ticks]")
        g_SigmaTrg_LED.Write()


    print("All graphs written to", out_file_name)
    out_root_file.Close()
