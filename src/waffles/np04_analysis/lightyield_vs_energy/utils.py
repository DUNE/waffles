from waffles.np04_analysis.lightyield_vs_energy.imports import *


# To select beam events for self-trigger channels looking at the time_offset DAQ-PDS (check value!!)    
def beam_self_trigger_filter(waveform : Waveform, timeoffset_min : int = -120, timeoffset_max : int = -90) -> bool:
    daq_pds_timeoffset = np.float32(np.int64(waveform.timestamp)-np.int64(waveform.daq_window_timestamp))
    if (daq_pds_timeoffset < timeoffset_max) and (daq_pds_timeoffset > timeoffset_min) :
        return True
    else:
        return False
    

# To select wf of a given run  
def run_filter(waveform : Waveform, run : int) -> bool:
    if (waveform.run_number == run):
        return True
    else:
        return False

def channel_filter(waveform : Waveform, end : int, ch : int) -> bool:
    if (waveform.channel == ch) and (waveform.endpoint == end) :
        return True
    else:
        return False

################################################################################################################################################    


def which_endpoints_in_the_APA(APA : int):
    endpoint_list = []
    for row in APA_map[APA].data: # cycle on rows
        for ch_info in row: # cycle on columns elements (i.e. channels)
            endpoint_list.append(ch_info.endpoint)
    return list(set(endpoint_list))


def which_APA_for_the_ENDPOINT(endpoint: int):
    apa_endpoints = {1: {104, 105, 107}, 2: {109}, 3: {111}, 4: {112, 113}}
    for apa, endpoints in apa_endpoints.items():
        if endpoint in endpoints:
            return apa
    return None

    

def which_channels_in_the_ENDPOINT(endpoint : int):
    channel_list = []
    for APA, apa_info in APA_map.items():
        for row in apa_info.data: # cycle on rows
            for ch_info in row: # cycle on columns elements (i.e. channels)
                if ch_info.endpoint == endpoint :
                    channel_list.append(ch_info.channel)       
    return channel_list


################################################################################################################################################    


def searching_for_beam_events(wfset : WaveformSet, show : bool = False, save : bool = True, bin : int = 100000, x_min = None, x_max = None, beam_min = None, beam_max = None, output_folder : str = 'output'):
    timeoffset_list_DAQ = []
    for wf in wfset.waveforms: 
        timeoffset_list_DAQ.append(np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp)))
    
    if x_min is None:
        x_min = min(timeoffset_list_DAQ)
    if x_max is None:
        x_max = max(timeoffset_list_DAQ)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=timeoffset_list_DAQ,
        nbinsx=bin,  
        marker=dict(color='blue', line=dict(color='black', width=1))))
    
    if beam_min is not None:
        fig.add_shape(
            type="line",
            x0=beam_min,
            x1=beam_min,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Min \n(x = {beam_min})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Min \n(x = {beam_min})"))

    if beam_max is not None:
        fig.add_shape(
            type="line",
            x0=beam_max,
            x1=beam_max,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Max \n(x = {beam_max})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Max \n(x = {beam_max})"))
    
    fig.update_layout(
        title="Interactive Histogram of Time Offsets",
        xaxis_title="Time Offset",
        yaxis_title="Count",
        xaxis=dict(range=[x_min, x_max]),  
        template="plotly_white",
        bargap=0.1)
    
    if show:
        fig.show()
    
    if save:
        if beam_min is not None and beam_max is not None:
            fig.update_layout(xaxis=dict(range=[beam_min - 200, beam_max + 200]))
        fig.write_image(f"{output_folder}/beam_timeoffset_histogram.png", scale=2)
    
################################################################################################################################################    

def plotting_overlap_wf(wfset, n_wf: int = 50, show : bool = False, save : bool = True, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None, output_folder : str = 'output', deconvolution : bool = False, analysis_label : str = ''):
    fig = go.Figure()

    for i in range(n_wf):
        
        if deconvolution:
            y = wfset.waveforms[i].analyses[analysis_label].result['filtered_deconvolved_wf']
        else:
            y = wfset.waveforms[i].adcs
            
        fig.add_trace(go.Scatter(
            x=np.arange(len(y)) + wfset.waveforms[i].time_offset,
            y=y,
            mode='lines',
            line=dict(width=0.5),
            showlegend=False))

    xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
    yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

    fig.update_layout(
        xaxis_title="Time ticks",
        yaxis_title="Adcs",
        xaxis=xaxis_range,  
        yaxis=yaxis_range,  
        margin=dict(l=50, r=50, t=20, b=50),
        template="plotly_white",
        legend=dict(
            x=1,  
            y=1,  
            xanchor="right",
            yanchor="top",
            orientation="v", 
            bgcolor="rgba(255, 255, 255, 0.8)" ))

    if int_ll is not None:
        fig.add_shape(
            type="line",
            x0=int_ll,
            x1=int_ll,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="coral", width=2, dash="dash"),
            name=f"Lower integral limit \n(x = {int_ll})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="coral", width=2, dash="dash"),
            showlegend=True,
            name=f"Lower integral limit \n(x = {int_ll})"
        ))

    if int_ul is not None:
        fig.add_shape(
            type="line",
            x0=int_ul,
            x1=int_ul,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="chocolate", width=2, dash="dash"),
            name=f"Upper integral limit \n(x = {int_ul})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="chocolate", width=2, dash="dash"),
            showlegend=True,
            name=f"Upper integral limit \n(x = {int_ul})"
        ))

    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=baseline,
            y1=baseline,
            xref="paper",
            yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
            name=f"Baseline \n(y = {baseline})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=1.5, dash="dash"),
            showlegend=True,
            name=f"Baseline \n(y = {baseline})"
        ))

    if save:
        fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
    if show:
        fig.show()
        
        
    
################################################################################################################################################

def LightYield_SelfTrigger_channel_analysis(wfset : WaveformSet, end : int, ch : int, run_set : dict, analysis_label : str, deconvolution: bool, deconvolution_filter: bool = True):
    integral_label_list = ['integral_before']
    if deconvolution:
        integral_label_list.append('integral_deconv')
    if deconvolution_filter:
        integral_label_list.append('integral_deconv_filtered')
    
    info_dic = {}
    
    for integral_label in integral_label_list:
        print(f'\n -- {integral_label} -- \n')
        if len(wfset.runs)>2: 
            ly_data_dic = {key: {} for key in run_set['Runs'].keys()} 
            ly_result_dic = {'x':[], 'y':[], 'e_y':[], 'slope':{}, 'intercept':{}} 
            for energy, run in run_set['Runs'].items():
                try:
                    wfset_run = wfset.from_filtered_WaveformSet(wfset, run_filter, run)
                    ly_data_dic[energy] = charge_study(wfset_run, end, ch, run, energy, analysis_label, integral_label)
                    ly_result_dic['x'].append(int(energy))
                    ly_result_dic['y'].append(ly_data_dic[energy]['gaussian fit']['mean']['value'])
                    ly_result_dic['e_y'].append(ly_data_dic[energy]['gaussian fit']['sigma']['value'])
                    print(f"For energy {energy} GeV --> {ly_data_dic[energy]['gaussian fit']['mean']['value']:.0f} +/- {ly_data_dic[energy]['gaussian fit']['sigma']['value']:.0f}")
                except Exception as e:
                    print(f'For energy {energy} GeV --> no data')
                
            if len(ly_result_dic['x']) > 1:
                popt, pcov = curve_fit(linear_fit, ly_result_dic['x'], ly_result_dic['y'], sigma=ly_result_dic['e_y'], absolute_sigma=True)
                slope, intercept = popt
                slope_err, intercept_err = np.sqrt(np.diag(pcov))
                
                ly_result_dic['slope'] = {'value': slope, 'error': slope_err}
                ly_result_dic['intercept'] = {'value': intercept, 'error': intercept_err}
            else:
                ly_result_dic['slope'] = {'value' : 0, 'error' : 0}
                ly_result_dic['intercept'] = {'value' : 0, 'error' : 0}
            
            info_dic[integral_label] = {'ly_data_dic': ly_data_dic, 'ly_result_dic' : ly_result_dic}
             
        else:
            print('Not enought runs avilable for that channel --> skipped')
            info_dic[integral_label] = {'ly_data_dic': {}, 'ly_result_dic' : {}}   
    
    return info_dic


def charge_study(wfset : WaveformSet, end : int, ch : int, run : int, energy : int, analysis_label : str, integral_label : str):  
    ly_data = {'histogram data': [], 'gaussian fit' : {}}
    charges = []

    for wf in wfset.waveforms:
        charges.append(wf.analyses[analysis_label].result[integral_label])
    ly_data['histogram data'] = charges
    charges = np.array(charges)
    
    try:
        bins = 100
        bin_heights, bin_edges = np.histogram(charges, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
        p0 = [np.mean(charges), np.std(charges), max(bin_heights)]
        popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
        perr = np.sqrt(np.diag(pcov))    
        
        
        ly_data['gaussian fit'] = {'mean': {'value': popt[0], 'error': perr[0]}, 'sigma': {'value': popt[1], 'error': perr[1]}, 'normalized amplitude': {'value': popt[2], 'error': perr[2]}, 'bins' : bins} 
        
        return ly_data
        
    except Exception as e:
        print(f'Fit error: {e} --> skipped')
        return ly_data
    

################################################################################################################################################x    
    
def gaussian(x, mu, sigma, amplitude):
    if not np.isfinite(sigma) or sigma == 0:
        return np.full_like(x, np.nan)  # Ritorna un array di NaN se sigma non è valido
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def linear_fit(x, a, b):
    return a * x + b

def round_to_significant(value, error):
    if not isinstance(error, (int, float, np.number)) or not np.isfinite(error) or error == 0:
        return "N/A", "N/A"  # Restituisce una stringa se l'errore non è valido

    error_order = int(np.log10(error))
    significant_digits = 2  
    rounded_error = round(error, -error_order + (significant_digits - 1))
    rounded_value = round(value, -error_order + (significant_digits - 1))
    return rounded_value, rounded_error

def to_scientific_notation(value, error):
    # Controllo se il valore non è un numero valido
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"  # Se il valore è NaN o infinito, restituisce "N/A"

    if value == 0:
        return "0"  # Caso speciale: zero non ha notazione scientifica

    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0  # Evita log10(0)
    mantissa_value = value / 10**exponent
    mantissa_error = error / 10**exponent if error and np.isfinite(error) else 0  # Evita divisioni errate

    mantissa_value, mantissa_error = round_to_significant(mantissa_value, mantissa_error)
    
    return f"({mantissa_value} ± {mantissa_error}) × 10^{exponent}" if mantissa_value != "N/A" else "N/A"


################################################################################################################################################

def fbk_or_hpk(endpoint: int, channel: int):
    channel_vendor_map = {
    104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
    105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
    107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
          10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
    109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
    113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

    return channel_vendor_map[endpoint][channel]

def spe_charge(df: pd.DataFrame, endpoint: int, channel: int, hpk_ov: float = 3.0, fbk_ov = 4.5):
    if fbk_or_hpk(endpoint, channel) == 'FBK':
        ov_column = 'FBK_OV_V'
        ov = fbk_ov
    else:
        ov_column = 'HPK_OV_V'
        ov = hpk_ov
    result = df[(df['endpoint'] == endpoint) & (df['channel'] == channel) & (df[ov_column] == ov)]['gain']
    return result.iloc[0] if not result.empty else None




#################################################

def daphne_to_offline_channel(apa, endpoint, daq_channel, map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv', maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'):
    df = pd.read_csv(map_path, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))    
    return daphne_to_offline[daq_channel + 100*endpoint]

def offline_to_daphne_channel(offline_ch, map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv', maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'):
    df = pd.read_csv(map_path, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    offline_to_daphne = dict(zip(df['offline_ch'],daphne_channels))  
    daphne_ch = offline_to_daphne[offline_ch]
    end = int(str(daphne_ch)[:3])
    daq_ch = int(str(daphne_ch)[3:])
    return end, daq_ch
