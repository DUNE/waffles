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

def filter_not_coindential_wf(wfsets,coincidences_level,timestamps,min_timestamp,n_channel,n_run):

    true_index_array= [ [set() for _ in range(n_channel)] for _ in range(n_run)]

    for run_index in range(n_run):
        for ch in range(n_channel):
            for coincidence_index in range(len(coincidences_level[run_index][8])):    
                index=coincidences_level[run_index][8][coincidence_index][1][ch]

               
                true_index=find_true_index(wfsets,run_index,ch,timestamps,index,min_timestamp)
                true_index_array[run_index][ch].add(true_index)


            true_index_array[run_index][ch]=sorted(true_index_array[run_index][ch])

            for n in range(len(wfsets[run_index][ch].waveforms)-1,-1,-1):
                if n not in true_index_array[run_index][ch]:
                    wfsets[run_index][ch].waveforms.pop(n)

    return wfsets

def write_output(self) -> bool:

      
        return True