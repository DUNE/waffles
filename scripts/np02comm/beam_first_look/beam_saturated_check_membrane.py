#!/usr/bin/env python
# coding: utf-8

import waffles
import numpy as np
import json
import shutil 
from tqdm import tqdm
import mplhep
import matplotlib.pyplot as plt
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({'font.size': 20,
                        'grid.linestyle': '--',
                        'axes.grid': True,
                        'figure.autolayout': True,
                        'figure.figsize': [14,6]
                        })

import waffles
import numpy as np
import json
import shutil 
from tqdm import tqdm

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.utils.baseline.baseline import SBaseline
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module, dict_module_to_uniqch, strUch, ordered_channels_membrane, ordered_channels_cathode
from waffles.np02_utils.PlotUtils import np02_gen_grids, plot_grid, plot_detectors, genhist, fithist, runBasicWfAnaNP02, ch_read_calib




def main(run, outputdir):

    dettype = "membrane"


    ## Only change if necessary
    datadir = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/"
    det = "VD_Cathode_PDS" if dettype == "cathode" else "VD_Membrane_PDS"
    endpoint = 106 if dettype == "cathode" else 107

    # Way to low... keep scrollng
    dletter = dettype.upper()[0] # C or M...
    group1 = [ f"{dletter}{detnum}({chnum})" for detnum in range(1, 3) for chnum in range(1,3) ]
    group2 = [ f"{dletter}{detnum}({chnum})" for detnum in range(3, 5) for chnum in range(1,3) ]
    group3 = [ f"{dletter}{detnum}({chnum})" for detnum in range(5, 7) for chnum in range(1,3) ]
    group4 = [ f"{dletter}{detnum}({chnum})" for detnum in range(7, 9) for chnum in range(1,3) ]
    groupall = group1+group2+group3+group4

    calib_values = ch_read_calib()
    list_of_unch = ordered_channels_membrane




    from glob import glob
    import copy
    def open_processed(run, dettype, datadir, channels = None, endpoints=None, nwaveforms=None, mergefiles = False, verbose=True):
        """
        Open the processed waveform set for a given run and detector type.
        """
        try: 
            wfset = load_structured_waveformset(
                f"{datadir}/processed/run{run:0d}_{dettype}/processed_merged_run{run:06d}_structured_{dettype}.hdf5",
                max_to_load=nwaveforms,
                channels_filter=channels,
                endpoint_filter=endpoints
            )
        except:
            files = glob(f"{datadir}/processed/run{run:06d}_{dettype}/processed_*_run{run:06d}_*_{dettype}.hdf5")
            if verbose:
                print("List of files found:")
                print(files)
            if not mergefiles or len(files)==1:
                files = files[0]
                wfset = load_structured_waveformset(files, max_to_load=nwaveforms, channels_filter=channels, endpoint_filter=endpoints, verbose=verbose)
            else: 
                wfset = load_structured_waveformset(files[0], max_to_load=nwaveforms, channels_filter= channels, endpoint_filter=endpoints, verbose=verbose)
                for f in files[1:]:
                    tmpwf = load_structured_waveformset(f, max_to_load=nwaveforms, channels_filter= channels, endpoint_filter=endpoints, verbose=verbose)
                    wfset.merge(copy.deepcopy(tmpwf))
        return wfset


    channels=None

    wfset_full = open_processed(run, dettype, datadir, channels=channels, endpoints = [endpoint], nwaveforms=None, verbose=True, mergefiles=True)


    particle_by_type = {}
    particle_by_type['total'] = 'total'
    particle_by_type['HL']   = 'kCTBBeamChkvHL'
    particle_by_type['HLx']  = 'kCTBBeamChkvHLx'
    particle_by_type['HxLx']  = 'kCTBBeamChkvHxLx'
    particle_by_type['HxL'] = 'kCTBBeamChkvHxL'



    def adjust_offset(waveform:Waveform) -> bool:
        if -5 < waveform.daq_window_timestamp - waveform.timestamp < 25:
            return True
        return False
        
    wfset_triggered_all = WaveformSet.from_filtered_WaveformSet(wfset_full, adjust_offset, show_progress=True)



    from tqdm import tqdm
    triggertypes = {} # just for checking..
    triggergroups = {} # to check if more than one...
    triggertypedaq = {}
    daqtimestamps = {}
    for wf in tqdm(wfset_full.waveforms):
        triggername=wf.trigger_type_names
        firsttime = False
        if wf.daq_window_timestamp not in daqtimestamps:
            firsttime=True
            daqtimestamps[wf.daq_window_timestamp] = 1
            tname = triggername[0]
            triggertypedaq[tname] =  triggertypedaq.get(tname,0) + 1
        else:
            daqtimestamps[wf.daq_window_timestamp] += 1

        if (len(triggername)) > 1:
            raise ValueError("This should never happen!!!!")
        triggergroups[tuple(triggername)] =  triggergroups.get(tuple(triggername), 0) + 1
        for tn in triggername:
            triggertypes[tn] = triggertypes.get(tn, 0) + 1

    for triggername in triggertypedaq.keys():
        print(f"Trigger name {triggername} was done {triggertypedaq[triggername]}")# and have total of {triggergroups[tuple([triggername])]}")
    print(triggergroups)




    triggertypedaq['total'] = np.sum( [ value for key, value in triggertypedaq.items() if key not in ['kCTBOffSpillSnapshot', 'kCTBBeamSpillStart', 'total'] ] ) 


    def filter_offspill(waveform: Waveform, specifictype="") -> bool:
        if 'kCTBOffSpillSnapshot' in waveform.trigger_type_names or 'kCTBBeamSpillStart' in waveform.trigger_type_names:
            return False
        return True
    def filter_types(waveform: Waveform, specifictype) -> bool:
        if specifictype in waveform.trigger_type_names:
            return True
        return False

    def get_particle_wfset(wfset, specifictype):
        try:
            wftmp = WaveformSet.from_filtered_WaveformSet(wfset, filter_types, specifictype, show_progress=False)
            print(specifictype, wftmp)
            return wftmp
        except:
            print(f"{specifictype} not in this run...\n")
            return None
        
    def remove_saturated(waveform:Waveform) -> bool:
        if np.any(waveform.adcs>16000):
            return False
        return True

    wfset_triggered_types = {}
    wfset_triggered_types['total']= WaveformSet.from_filtered_WaveformSet(wfset_triggered_all, filter_offspill, show_progress=True)
    print("total...", wfset_triggered_types['total'])
    wfset_triggered_types['HL']   = get_particle_wfset(wfset_triggered_types['total'], specifictype = particle_by_type['HL'] )
    wfset_triggered_types['HLx']  = get_particle_wfset(wfset_triggered_types['total'], specifictype = particle_by_type['HLx'] )
    wfset_triggered_types['HxLx']  = get_particle_wfset(wfset_triggered_types['total'], specifictype = particle_by_type['HxLx'] )
    wfset_triggered_types['HxL'] = get_particle_wfset(wfset_triggered_types['total'], specifictype = particle_by_type['HxL'] )




    def processes_analysis_by_type(wfset, dkey, daqtriggerdict:dict, output_infos, particle_by_type, verbose=True):
        
        list_of_unch = ordered_channels_membrane
        output_infos[dkey] = {}
        output_infos[dkey]['DAQ_Trigger'] = {}
        output_infos[dkey]['Triggered_Events'] = {}
        output_infos[dkey]['Percent_trigger'] = {}
        output_infos[dkey]['No_saturated'] = {}
        output_infos[dkey]['Percent_nonsat'] = {}
        output_infos[dkey]['Charge'] = {}
        output_infos[dkey]['wfsetnosat'] = {}
        output_infos[dkey]['wfset'] = {}
        for ch in list_of_unch: #inital values all set 
            output_infos[dkey]['DAQ_Trigger'][ch] = 0
            output_infos[dkey]['Triggered_Events'][ch] = 0
            output_infos[dkey]['Percent_trigger'][ch] = 0
            output_infos[dkey]['No_saturated'][ch] = 0
            output_infos[dkey]['Percent_nonsat'][ch] = 0
            output_infos[dkey]['Charge'][ch] = 0
            output_infos[dkey]['wfsetnosat'][ch] = None
            output_infos[dkey]['wfset'][ch] = None
            

        if wfset is None:
            return
        daqtrigger = daqtriggerdict[particle_by_type[dkey]]
        chwfob = ChannelWsGrid.clusterize_waveform_set(wfset)
        if verbose:
            print("\nTriggered channels 'efficiency'")
            print("Module, n records, n DAQ triggers, %")
        nbeamrecors_ch_triggered = {}
        for ep, wfsch in chwfob.items():
            for ch in list_of_unch:
                if ch not in wfsch:
                    nbeamrecors_ch_triggered[ch] = 0
                else:
                    recordschob = len(wfsch[ch].record_numbers[run])
                    nbeamrecors_ch_triggered[ch] = recordschob
                    if len(wfsch[ch].waveforms) != recordschob:
                        raise ValueError("This should never happen...")
                if verbose:
                    print(f"{dict_uniqch_to_module[str(UniqueChannel(ep,ch))]}, {nbeamrecors_ch_triggered[ch]}, {daqtrigger}, {100*nbeamrecors_ch_triggered[ch]/daqtrigger:.2f}%")
                output_infos[dkey]['DAQ_Trigger'][ch] = daqtrigger
                output_infos[dkey]['Triggered_Events'][ch] = nbeamrecors_ch_triggered[ch]
                output_infos[dkey]['Percent_trigger'][ch] = 100*nbeamrecors_ch_triggered[ch]/daqtrigger if daqtrigger != 0 else 0

        runBasicWfAnaNP02(wfset, onlyoptimal=False, baselinefinish=60, int_ll=60, int_ul=600, amp_ll=60, amp_ul=150, configyaml="", show_progress=verbose)
        wfset_onbeam_nonsat = WaveformSet.from_filtered_WaveformSet(wfset, remove_saturated, show_progress=False)

        chwfob_ns = ChannelWsGrid.clusterize_waveform_set(wfset_onbeam_nonsat)
        list_of_devices = sorted([ m for m in dict_module_to_uniqch.keys() if m ])
        list_of_unch = [ dict_module_to_uniqch[m].channel for m in list_of_devices if dict_module_to_uniqch[m].endpoint == endpoint ]
        if verbose:
            print("\nTriggered channels saturation")
            print("Module, n not saturated, n triggers, %")
        nbeamrecors_ch_triggered_nosat = {}
        for ep, wfsch in chwfob_ns.items():
            for ch in list_of_unch:
                meancharge = 0
                if ch not in wfsch:
                    nbeamrecors_ch_triggered_nosat[ch] = 0
                else:
                    recordschob = len(wfsch[ch].record_numbers[run])
                    nbeamrecors_ch_triggered_nosat[ch] = recordschob
                    if len(wfsch[ch].waveforms) != recordschob:
                        raise ValueError("This should never happen...")
                    charges = [ wf.analyses['std'].result['integral'] for wf in chwfob_ns[endpoint][ch].waveforms if wf.analyses['std'].result['integral'] is not np.nan ]
                    meancharge = np.mean(charges)
                    
                nperc = 0 if nbeamrecors_ch_triggered[ch] ==0 else 100*nbeamrecors_ch_triggered_nosat[ch]/nbeamrecors_ch_triggered[ch]
                if verbose:
                    print(f"{dict_uniqch_to_module[str(UniqueChannel(ep,ch))]}, {nbeamrecors_ch_triggered_nosat[ch]}, {nbeamrecors_ch_triggered[ch]}, {nperc:.2f}%")

                output_infos[dkey]['No_saturated'][ch] = nbeamrecors_ch_triggered_nosat[ch]
                output_infos[dkey]['Percent_nonsat'][ch] = nperc
                output_infos[dkey]['Charge'][ch] = meancharge
            output_infos[dkey]['wfsetnosat'] = wfset_onbeam_nonsat
            output_infos[dkey]['wfset'] = wfset
            print("\n")
                
                
    output_infos = {}
    for dkey, wfset in wfset_triggered_types.items():
        processes_analysis_by_type(wfset, dkey, triggertypedaq, output_infos=output_infos, particle_by_type=particle_by_type, verbose=True)


    mnames = [ mname for mname in list(output_infos['total']) if 'wfset' not in mname ]
    print(', '.join(mnames))
    for ttype, outp in output_infos.items():
        for ch in list_of_unch:
            for mname in list(output_infos[ttype])[:-1]:
                if 'wfset' in mname:
                    continue
                if mname != "Charge":
                    if mname.find("Percent"):
                        print(f"{output_infos[ttype][mname][ch]}", end=", ")
                    else:
                        print(f"{output_infos[ttype][mname][ch]:.2f}", end=", ")
                        
                else:
                    print(f"{output_infos[ttype][mname][ch]:.2f}")





    module_ordered = list(dict.fromkeys([ dict_uniqch_to_module[strUch(endpoint, ch)][:2] for ch in ordered_channels_membrane ]))
    print(run)
    print("order...")
    for daqt, out in output_infos.items():
        print(daqt)
    fout = open(f"{outputdir}/pe_info_membrane_run{run:06d}.csv", "w")
    print("\nheader")
    fout.write("Run,Part,")

    for m in module_ordered:
        print(m, end=',')
        fout.write(f"{m},")
    print('SUM')
    fout.write('SUM\n')


    for daqt, out in output_infos.items():
        wftmp:WaveformSet = out['wfset']
        record_pe_total = {}
        module_per_record = {}
        pe_per_module = {}
        if daqt == "HxL":
            continue
        fout.write(f"{run},{daqt},")
        if isinstance(wftmp, dict):
            pe_per_module = { m: [0] for m in module_ordered }
            record_pe_total = { 0 : 0 }
            module_per_record = { 0 : 1 } 
            
        else:
            for wf in wftmp.waveforms:
                chpe = wf.analyses['std'].result['integral']/calib_values[endpoint][wf.channel]['Gain']
                modulefullname = dict_uniqch_to_module[strUch(endpoint, wf.channel)]
                if modulefullname == "M2(1)":
                    continue
                modulename = modulefullname[:2]
                record_pe_total[wf.record_number] = record_pe_total.get(wf.record_number, 0) + chpe
                module_per_record[wf.record_number] =  module_per_record.get(wf.record_number,0) + 1
                pe_per_module[modulename] = pe_per_module.get(modulename, []) + [chpe]
        for m in module_ordered:
            if m in pe_per_module:
                pe = np.mean(pe_per_module[m])*2
            else:
                pe = 0
            print(f"{pe:.2f}", end=',')
            fout.write(f"{pe:.2f},")
        totalpe = []
        for rc, pe in record_pe_total.items():
            totalpe += [pe*16/module_per_record[rc]]
        print(f"{np.mean(totalpe):.2f}")
        fout.write(f"{np.mean(totalpe):.2f}\n")

    fout.close()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True, help="Run number to process")
    parser.add_argument("--output-dir", type=str, default="/afs/cern.ch/work/h/hvieirad/public/np02_light_response_data/", help="Output folder to save results")
    args = parser.parse_args()
    main(args.run, args.output_folder)

