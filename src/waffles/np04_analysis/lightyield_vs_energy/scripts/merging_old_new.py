# merge_waveformsets.py

import pickle
import click
from waffles.data_classes.WaveformSet import WaveformSet

@click.command(help="Merge due WaveformSet pickles e salva in un nuovo file.")
@click.option("--old-file", default='/eos/user/a/anbalbon/set_A/set_A_self_beam_ALL_beam_pkl.pkl', type=click.Path(exists=True), help="Pickle file esistente.")
@click.option("--new-file", default='//afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/set_A_beam_ALL.pkl', type=click.Path(exists=True), help="Nuovo pickle file da unire.")
@click.option("--output-name", required=True, type=str, help="Nome aggiuntivo da usare per il file di output.")
@click.option("--output-folder", default='/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data', type=str, help="Output folder per il file unito.")
@click.option("--set-name", default='A', type=str, help="Set che si sta analizzando")

def main(old_file, new_file, output_name, output_folder, set_name):
    # Leggi primo pickle
    with open(old_file, "rb") as f1:
        wfset1 = pickle.load(f1)
    click.echo(f"✅ Caricato {old_file} con {len(wfset1.waveforms)} waveforms")

    # Leggi secondo pickle
    with open(new_file, "rb") as f2:
        wfset2 = pickle.load(f2)
    click.echo(f"✅ Caricato {new_file} con {len(wfset2.waveforms)} waveforms")

    # Merge
    wfset1.merge(wfset2)
    click.echo(f"🔗 Merge completato. Totale waveforms = {len(wfset1.waveforms)}")

    # Salva
    output_file = f"{output_folder}/set_{set_name}/merged_data_{output_name}.pkl"
    with open(output_file, "wb") as fout:
        pickle.dump(wfset1, fout)
    click.echo(f"💾 Salvato file merge: {output_file}")


if __name__ == "__main__":
    main()
