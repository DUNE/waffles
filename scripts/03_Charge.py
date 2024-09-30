import click, pickle, json, inquirer
from rich import print
import plotly.subplots as psu
import plotly.graph_objects as pgo

from waffles.utils.utils import print_colored
from waffles.plotting.plot_utils import save_plot

@click.command(help=f"\033[34mSave the WaveformSet object in a pickle file for easier loading.\n\033[0m")
@click.option("--run",   default = None, help="Run number to process", type=str)
@click.option("--chs",   default = None, help="Channel number to process (i.e. 44)", type=int)
@click.option("--eps",   default = None, help="Endpoint number to process (i.e. 109)", type=int) #TODO: allow simulation channels map?
@click.option("--debug", default = False, help="Debug flag")
def main(run,chs, eps, debug):
    '''
    Script to process peak/pedestal variables and save the WaveformSet object + unidimensional variables in a pickle file.

    Args:
        - run (int): Run number to be analysed. I can also be a list of runs separated by commas.
        - chs (int): Channel number to be analysed. I can also be a list of channels separated by commas.
        - eps (int): Endpoint number to be analysed.
        - debug (bool): Debug flag.
    Example: python 03Charge.py --run 28602 --chs 31 --eps 109 --debug True
    '''
    
    if run is None: 
        q = [ inquirer.Text("run", message="Please provide the run(s) number(s) to be analysed, separated by commas:)")]
        run_list = inquirer.prompt(q)["run"].split(",")
    else:
        run_list = [run]
    
    if chs is None: 
        q = [ inquirer.Text("chs", message="Please provide the channel(s) number(s) to be analysed, separated by commas:)")]
        chs_list = inquirer.prompt(q)["chs"].split(",")
    else:
        chs_list = [chs]
    
    if eps is None:
        q = [ inquirer.Text("eps", message="Please provide the endpoint number to be analysed, separated by commas:)")]
        eps = int(inquirer.prompt(q)["eps"].split(",")[0])
    
    
    for r in run_list:
        json_file_path = f"../conf/{str(r).zfill(6)}.json"
        try:
            with open(json_file_path, 'r') as json_file:
                analysis_conf = json.load(json_file)
        except FileNotFoundError:
            print_colored(f"\nFile {json_file_path} not found.", color="ERROR")
            continue
        
        analysis_labels = list(analysis_conf.keys())
        if len(analysis_labels) == 1:
            analysis_label = analysis_labels[0]
        else:
            question = [inquirer.List('analysis_label', message="Choose the analysis label", choices=analysis_labels)]
            answer = inquirer.prompt(question)
            analysis_label = answer['analysis_label']
        
        for ch in chs_list:
            ch = int(ch)
            try:
                with open(f"../data/{str(r).zfill(6)}_wfset_ana_ep{eps}_ch{ch}.pkl", 'rb') as file:
                    wfset = pickle.load(file)
            except FileNotFoundError:
                print_colored(f"\nFile {str(r).zfill(6)}_full_wfset_raw.pkl not found. Please run 01Process.py first.", color="ERROR")
                return
            
            results = wfset.waveforms[0].analyses[analysis_label].result.keys()
            choose_result = [ inquirer.Checkbox("result", message="Choose the result(s) to be extracted", choices=results) ]
            chosen_results = inquirer.prompt(choose_result)["result"]
            for my_result in chosen_results:
                print_colored(f"Extracting {my_result} from {analysis_label} analysis...", color="INFO")
                result_values = [wf.analyses[analysis_label].result[my_result] for wf in wfset.waveforms]
                with open(f"../data/{my_result}_run{r}_ep{eps}_ch{ch}.pkl", "wb") as f:
                    pickle.dump(result_values, f)
                figure = psu.make_subplots( rows = 1, cols = 1 )
                histogram = pgo.Histogram(x=result_values, nbinsx=500, name=f'{my_result}')
                figure.add_trace(histogram, row=1, col=1)
                
                figure.update_layout(template="presentation", title = "", xaxis_title="Charge [ADC*ticks]", yaxis_title="Counts")
                save_plot(figure, f"../data/{my_result}_dist_ep{eps}_ch{ch}.png")
                figure.show()
            

if __name__ == "__main__":
    main()