from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.Exceptions import GenerateExceptionMessage

def subtract_baseline(
        waveform: WaveformAdcs,
        baseline_analysis_label: str
) -> None:
    """This method overwrites the adcs method of the given
    WaveformAdcs object, by subtracting its baseline.

    waveform: WaveformAdcs
        The waveform whose adcs will be modified
    baseline_analysis_label: str
        The baseline to subtract must be available 
        under the 'baseline' key of the result of the analysis
        whose label is given by this parameter, i.e. in
        waveform.analyses[analysis_label].result['baseline']
    """

    try:
        baseline = waveform.analyses[baseline_analysis_label].result['baseline']

    except KeyError:
        raise Exception(
            GenerateExceptionMessage(
                1,
                "subtract_baseline()",
                f"The given waveform does not have the analysis"
                f" '{baseline_analysis_label}' in its analyses "
                "attribute, or it does, but the 'baseline' key "
                "is not present in its result."
            )
        )
    
    waveform._WaveformAdcs__set_adcs(
        waveform.adcs - baseline
    )

    return