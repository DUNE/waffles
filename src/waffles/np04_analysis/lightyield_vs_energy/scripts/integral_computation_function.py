

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.np04_analysis.led_calibration.utils import compute_average_baseline_std #baseline computation
from waffles.utils.baseline.baseline_utils import subtract_baseline # baseline subtraction
from waffles.utils.integral.integral_utils import get_pulse_window_limits
from waffles.utils.baseline.WindowBaseliner import WindowBaseliner
from waffles.data_classes.StoreWfAna import StoreWfAna 
from waffles.data_classes.IPDict import IPDict

from waffles.np04_analysis.lightyield_vs_energy.scripts.MY_WindowIntegrator import MY_WindowIntegrator # ORIGINAL
from waffles.np04_analysis.lightyield_vs_energy.scripts.MY_Integrator_Peak import MY_Integrator_Peak # NEW - With peak finding


def channel_integral_computation(
    ch_wfset: WaveformSet,
    period: str = 'june',
    baseline_limits: list = [0, 50],
    baseliner_std_cut: float = 3.,
    baseliner_type: str = 'mean',
    baseline_analysis_label: str = 'baseliner',
    null_baseline_analysis_label: str = 'null_baseliner',
    deviation_from_baseline: float = 0.3,
    lower_limit_correction: int = 0,
    upper_limit_correction: int = 0,
    integration_analysis_label: str = 'integrator',
    beam_average_timetick = None,
    delta_beam_average_timetick: int = 200
    ): 

    # Check that wfset is associated just to one channel

    # Baseline
    baseliner_input_parameters = IPDict({'baseline_limits': baseline_limits, 'std_cut': baseliner_std_cut, 'type': baseliner_type})    
    checks_kwargs = IPDict({'points_no': ch_wfset.points_per_wf})
    _ = ch_wfset.analyse(baseline_analysis_label, WindowBaseliner, baseliner_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

    # Add a dummy baseline analysis to the merged WaveformSet(we will use this for the integration stage after having - subtracted the actual baseline)
    _ = ch_wfset.analyse(null_baseline_analysis_label, StoreWfAna, {'baseline': 0.}, overwrite=True)

    # Compute average baseline std
    average_baseline_std = compute_average_baseline_std(ch_wfset, baseline_analysis_label)

    # Remove baseline
    ch_wfset.apply(subtract_baseline, baseline_analysis_label, show_progress=False)

    # Compute average  waveform
    mean_wf = ch_wfset.compute_mean_waveform()

    # NEW PART --> bisogna dire che se c'è un valore non None beam_average_timetick allora i limiti li cerca in un certo intervallo
    if beam_average_timetick is not None:
        print('caio')
        adcs_array = mean_wf.adcs[beam_average_timetick-delta_beam_average_timetick: beam_average_timetick+delta_beam_average_timetick]
    else: 
        print('nada')
        adcs_array = mean_wf.adcs

    # Compute integration limits
    limits = get_pulse_window_limits(adcs_array, 0, deviation_from_baseline, lower_limit_correction, upper_limit_correction)
    limits = list(limits)
    print(limits)
    if beam_average_timetick is not None:
        limits[0] = limits[0] + beam_average_timetick-delta_beam_average_timetick
        limits[1] = limits[1] + beam_average_timetick-delta_beam_average_timetick
        print(limits)
    
    print(limits[0])
    print(limits[1])
    # Compute integral + information about spe
    integrator_input_parameters = IPDict({'baseline_analysis': null_baseline_analysis_label, 'inversion': True, 'int_ll': limits[0], 'int_ul': limits[1], 'amp_ll': limits[0], 'amp_ul': limits[1], 'period': period})
    checks_kwargs = IPDict({'points_no': ch_wfset.points_per_wf})
    _ = ch_wfset.analyse(integration_analysis_label, MY_Integrator_Peak, integrator_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

    return ch_wfset