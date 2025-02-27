from waffles.np04_analysis.light_vs_hv.imports import *

def check_endpoint_and_channel(endpoint,channel):
    for APA in range(1,5,1):
        for i in range(10):
            for j in range(4):
                endpoint_now=int(APA_map[APA].data[i][j].endpoint)
                ch_now=int(APA_map[APA].data[i][j].channel)
                
                if endpoint==endpoint_now and channel==ch_now:
                    return True
    return False
    
def get_ordered_timestamps(wfsets,n_channel,n_run):

    timestamps=[ [ [wfsets[i][j].waveforms[k].timestamp for k in range(len(wfsets[i][j].waveforms))] 
              for j in range(n_channel)] for i in range(n_run)]

    min_timestamp = min(min(min(row) for row in layer) for layer in timestamps).astype(np.float64)
    #max_timestamp = max(max(max(row) for row in layer) for layer in timestamps).astype(np.float64)

    timestamps=[ [ [timestamps[i][j][k]-min_timestamp for k in range(len(wfsets[i][j].waveforms))] 
                for j in range(n_channel)] for i in range(n_run)]

    timestamps=[ [ sorted(timestamps[i][j]) for j in range(n_channel)] for i in range(n_run)]

    return timestamps, min_timestamp

def get_all_double_coincidences(timestamps,n_channel,n_run,time_diff):

    coincidences=[[[[] for _ in range(n_channel)] for _ in range(n_channel)] for _ in range(n_run)]

    record_j=0

    for file_index in range(n_run):
        for line_index_i in range(1):#range(n_channel):
            for line_index_j in range(line_index_i+1,n_channel,1):
                record_j=0
                for i in range(len(timestamps[file_index][line_index_i])):
                    taux1 = timestamps[file_index][line_index_i][i].astype(np.float64)
                    for j in range(record_j,len(timestamps[file_index][line_index_j]),1):
                        taux2 = timestamps[file_index][line_index_j][j].astype(np.float64)
                        diff = taux2 - taux1
                        if diff >= 0:
                            record_j=j
                            if diff <= time_diff:
                                coincidences[file_index][line_index_i][line_index_j].append([i,j,diff])
                                break
                            else:
                                break
    return coincidences

def get_all_coincidences(coincidences,timestamps,n_channel,n_run):
    
    coincidences_mult=[[] for _ in range(n_run)]

    for file_index in range(n_run): #varre as runs
        for i in range(len(timestamps[file_index][0])): #varre todos os indices
            chs_aux=[[] for _ in range(3)]
            for j in range(1,n_channel,1): #varre todos os canais targets
                for k in range(len(coincidences[file_index][0][j])):#varre todas as concidencias do canal target
                    if i == coincidences[file_index][0][j][k][0]: #achou uma coincidencia com esse indice
                        if len(chs_aux[0])==0:#se eh o primeiro que acha
                            chs_aux[0].append(0) #salva o canal do canal pai
                            chs_aux[1].append(i) #salva o indice do canal pai
                            chs_aux[2].append(0) #salva o delta_t do canal pai
                        chs_aux[0].append(j) #salva o canal do canal target
                        chs_aux[1].append(coincidences[file_index][0][j][k][1]) #salva o indice do canal target
                        chs_aux[2].append(coincidences[file_index][0][j][k][2]) #salva o delta_t do canal target
                    elif i<coincidences[file_index][0][j][k][0]: #se o indice ja eh maior que o indice da coincidencia buscada
                        break
            if len(chs_aux[0])>0:
                coincidences_mult[file_index].append(chs_aux) 

    return coincidences_mult

def get_level_coincidences(coincidences_mult,n_channel,n_run):
    coincidences_level = [[[] for _ in range(n_channel-1)] for _ in range(n_run) ]

    for file_index in range(n_run):
        for i,value in  enumerate(coincidences_mult[file_index]):
            coincidences_level[file_index][len(value[0])-2].append(value)

    return coincidences_level

def find_true_index(wfs,file_index,channel,timestamps,index,minimo):
    goal=timestamps[file_index][channel][index]+minimo

    N=len(wfs[file_index][channel].waveforms)
    for k in range(N):
        if goal == wfs[file_index][channel].waveforms[k].timestamp:
            return k
    return -1

def filter_not_coindential_wf(wfsets,coincidences_level,timestamps,min_timestamp,n_channel,n_run,coincidence_min):

    true_index_array= [ [set() for _ in range(n_channel)] for _ in range(n_run)]

    for run_index in range(n_run):
        for ch in range(n_channel):
            for coincidence_min_index in range(coincidence_min,coincidence_min+1,1):

                for coincidence_index in range(len(coincidences_level[run_index][coincidence_min_index])):
                    this_ch=  coincidences_level[run_index][coincidence_min_index][coincidence_index][0][ch]  
                    index=coincidences_level[run_index][coincidence_min_index][coincidence_index][1][ch]

                    true_index=find_true_index(wfsets,run_index,this_ch,timestamps,index,min_timestamp)

                    if true_index not in  true_index_array[run_index][this_ch]:
                        true_index_array[run_index][this_ch].add(true_index)


    for run_index in range(n_run):
        for ch in range(n_channel):    
            true_index_array[run_index][ch]=sorted(true_index_array[run_index][ch])

            for n in range(len(wfsets[run_index][ch].waveforms)-1,-1,-1):
                if n not in true_index_array[run_index][ch]:
                    wfsets[run_index][ch].waveforms.pop(n)

    return wfsets

def from_generic(waveform : Waveform, max=None, min=None, analysis_label=None, parameter_label=None) ->bool :

    if parameter_label==None:
        print("didnt sent any parameter to filter")
        return True
    if analysis_label==None:
        print("didnt sent any analysis label to look")
        return True

    if max==None:
        if min==None:
            return True
        elif waveform.analyses[analysis_label].result[parameter_label] < min:
            return False
        else: 
            return True
    elif min == None:
        if waveform.analyses[analysis_label].result[parameter_label] > max:
            return False
        else:
            return True
    else: 
        if waveform.analyses[analysis_label].result[parameter_label] <= max and waveform.analyses[analysis_label].result[parameter_label] >= min:
            return True
        else:
            return False




def generic_plot_ch(wfset,
                 analysis_label,
                 param_label,
                 fig,
                 config_param,
                 ch,
                 endpoint,
                 param_label_y=None):
    
        filter_wfset=WaveformSet.from_filtered_WaveformSet( wfset, comes_from_channel, endpoint, [ch])
        generic_plot(filter_wfset,analysis_label,param_label,fig,config_param,param_label_y)

def generic_plot(wfset,
                 analysis_label,
                 param_label,
                 fig,
                 config_param,
                 param_label_y=None):
    
    #preparing_data
    data=[]
    for n in range(len(wfset.waveforms)):
        data.append(wfset.waveforms[n].analyses[analysis_label].result[param_label])

    try:
        start=config_param["start"]
    except:
        start=np.min(data)

    try:
        end=config_param["end"]
    except:
        end=np.max(data)
    
    try:
        bins=config_param["bins"]
    except:
        bins=100
    
    try:
        color=config_param["color"]
    except:
        color="rgba(0, 0, 255, 0.7)"

    try:
        name=config_param["name"]
    except:
        name=param_label
    
    try:
        title_fig=config_param["title_fig"]
    except:
        title_fig=None

    try: 
        xlabel=config_param["xlabel"]
    except:
        xlabel=None
    try: 
        ylabel=config_param["ylabel"]
    except:
        ylabel=None

    
    try:
        if config_param["norm"]==True:
            histnorm='probability density'
        else:
            histnorm=None
    except:
        histnorm=None

    if param_label_y == None:
        # Adicionar histograma
        fig.add_trace(pgo.Histogram(
        x=data,
        xbins=dict(
            start=start,  # Início do intervalo de bins
            end=end,    # Fim do intervalo de bins
            size=(end-start) / bins  # Tamanho de cada bin
        ),
        marker=dict(color=color),
        name=name,
        histnorm=histnorm
        ))

        fig.update_layout(
            title=title_fig,
            xaxis_title=xlabel,  # Rótulo do eixo x
            yaxis_title=ylabel,  # Rótulo do eixo y
            showlegend=True
        )
    
    else:
        #preparing_data
        data_y=[]
        for n in range(len(wfset.waveforms)):
            data_y.append(wfset.waveforms[n].analyses[analysis_label].result[param_label_y])

        try:
            start_y=config_param["start_y"]
        except:
            start_y=np.min(data_y)

        try:
            end_y=config_param["end_y"]
        except:
            end_y=np.max(data_y)
        
        try:
            bins_y=config_param["bins_y"]
        except:
            bins_y=100

        fig.add_trace(pgo.Histogram2d(
            x=data,
            y=data_y,
            xbins=dict(
                start=start,  # Início do intervalo de bins
                end=end,    # Fim do intervalo de bins
                size=(end-start) / bins  # Tamanho de cada bin
            ),
            ybins=dict(
                start=start_y,  # Início do intervalo de bins
                end=end_y,    # Fim do intervalo de bins
                size=(end_y-start_y) / bins_y  # Tamanho de cada bin
            ),
            colorscale="Viridis",
            coloraxis="coloraxis",  # Associate color scale with the plot
            name=name,
            ))


        fig.update_layout(
            title=title_fig,
            xaxis_title=xlabel,  # Rótulo do eixo x
            yaxis_title=ylabel,  # Rótulo do eixo y
            coloraxis_colorbar_title='Count',
            showlegend=True
        )


def start_plot(APA):
    titles = [f"Endpoint {APA_map[APA].data[i][j].endpoint} - Channel {APA_map[APA].data[i][j].channel}" 
              for i in range(10) for j in range(4)] 
    fig = make_subplots(rows=10, cols=4,subplot_titles=titles)
    return fig


def generic_plot_APA(fig,
                     wfset,
                     APA,
                     analysis_label,
                     param_label,
                     config_param,
                     param_label_y=None):
    try:
        name_label=config_param["name"]
    except:
        name_label=param_label
    

    for i in range(10):
        for j in range(4):
            endpoint=APA_map[APA].data[i][j].endpoint
            ch=APA_map[APA].data[i][j].channel
            
            try:
                #filter_wfset=WaveformSet.from_filtered_WaveformSet( wfset, comes_from_channel, 112, [6])
                
                filter_wfset=WaveformSet.from_filtered_WaveformSet( wfset, comes_from_channel, endpoint, [ch])
                fig_aux = pgo.Figure()
                config_param["name"]=name_label+"_"+str(endpoint)+"-"+str(ch)

                generic_plot(filter_wfset,analysis_label,param_label,fig_aux,config_param,param_label_y)
                for trace in fig_aux.data:
                    fig.add_trace(trace, row=i+1, col=j+1)
            except:
                None
                #print(f"no waveforms found in {endpoint}-{ch}")
    
    try:
        title_fig=config_param["title_fig"]
    except:
        title_fig=param_label


    # Atualizar os eixos x e y para cada subplot
    for i in range(10):
        for j in range(4):
            try:
                fig.update_xaxes(title_text=config_param["xlabel"], row=i+1, col=j+1)
            except:
                pass
            try:
                fig.update_yaxes(title_text=config_param["ylabel"], row=i+1, col=j+1)
            except:
                pass


    fig.update_layout(
        height=2000,  # Ajustar a altura conforme necessário
        width=1200,   # Ajustar a largura conforme necessário
        title_text=title_fig,
        showlegend=False
    )

def my_sin(x,A,T,phi,B,phi2,C):
    return A*np.sin(2*np.pi*x/T+phi)+B*np.sin(2*np.pi*x/(2*T)+phi2)+C

def func_tau(x,tau1,tau2,A1,A2,B):
    return B+A1*np.exp(-x/tau1)+A2*np.exp(-x/tau2)

def calculate_light(params):
    t=np.linspace(0,1023,1024)
    n=len(params)
    integral_conv=[np.sum(func_tau(t,*params[file_index])-params[file_index][4]) for file_index in range(n)]
    return integral_conv/integral_conv[0]

def birks_law(x,B,k):
    y=np.zeros(len(x))
    if x[0]==0:
        y[0]=1
        y[1::]=(1-B/(1+k/x[1::]))
    else:
        y=(1-B/(1+k/x))
    return y
