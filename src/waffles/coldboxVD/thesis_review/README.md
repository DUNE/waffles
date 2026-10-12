# VGAIN scan di novembre 2025

Questa analisi legge le waveform dello spy buffer con waffles, seleziona gli
eventi, sceglie la finestra di integrazione e calibra lo spettro di carica con
il fit multigaussiano di waffles. Ogni file `channel_N.dat` è analizzato in
modo indipendente: i parametri ricavati da un VGAIN, un OV o un modulo non
vengono riutilizzati per gli altri punti.

Eseguire dalla radice della repository, nell'ambiente Python di waffles. Per
esempio, dopo che il download di **tutti** i file interessati è terminato,
per analizzare il canale 18 di M1 a DVbias 42V85 su tutti i VGAIN:

```bash
python src/waffles/coldboxVD/thesis_review/select_waveforms.py \
  --input src/waffles/coldboxVD/thesis_review/data/M1/vgain_scan_M1_DVbias_42V85 \
  --recursive --channels 18 --expected-waveforms 15000 \
  --output src/waffles/coldboxVD/thesis_review/output/M1_42V85_ch18 \
  --report-pdf src/waffles/coldboxVD/thesis_review/output/pdf/M1_42V85_ch18.pdf \
  --debug-pdf
```

`--report-pdf` richiede un solo canale in `--channels`. Per una prova breve si
possono aggiungere, per esempio, `--vgains 1000 1500`; per un solo file o una
sola cartella `vgain_1000` si può passarne direttamente il percorso a
`--input`, senza `--recursive`. La stessa struttura di input funziona su
lxplus passando il percorso assoluto in `/eos/.../VGAIN_SCAN/M1/...`. Lo script
legge al massimo 40000 waveform per file (`--max-waveforms`) e assume 1024
campioni per waveform (`--length`). Per questo scan ciascun file scaricato
contiene 15000 waveform; `--expected-waveforms` intercetta anche un download
interrotto dopo un numero intero di record. Cambiare o omettere questa opzione
per acquisizioni con una lunghezza diversa. I dati in `data/` e i risultati in
`output/` sono esclusi da Git.

## Come vengono scelti i parametri

1. Il reader spy buffer di waffles crea un `WaveformSet` per ogni file. Un
   profilo robusto delle waveform individua il fronte del LED e la polarità;
   la finestra di baseline finisce 12 campioni prima del fronte. È possibile
   fissare manualmente `--baseline-stop` e forzare la polarità negativa con
   `--invert`.
2. La baseline di ogni waveform è la mediana dei campioni precedenti il LED.
   `--baseline-estimator mean` permette il confronto con il programma C++
   del collega, che usa la media del pretrigger.
   La selezione richiede un RMS preimpulso accettabile, nessuna grande
   escursione preimpulso e nessun campione ai rail ADC nella regione del LED.
   Le soglie di rumore sono stimate dallo stesso file e registrate in
   `summary.json`.
3. Se non vengono impostati `--int-low` e `--int-up`, lo script stima l'inizio
   dell'impulso da un profilo medio robusto e prova diversi estremi superiori
   della finestra. Fra i fit validi della metà delle waveform selezionate,
   sceglie la finestra più corta con SNR compatibile entro un'incertezza con
   il massimo; poi verifica il risultato sull'altra metà.
   Se non c'è un fit di training affidabile, usa una finestra basata sulla
   forma dell'impulso e segnala il problema: il punto non va considerato una
   misura validata di SNR. Gli estremi possono essere fissati insieme con
   `--int-low` e `--int-up` per una verifica manuale.
4. Waffles `WindowIntegrator` integra le waveform rispetto alla baseline senza
   modificare gli ADC originali. Gli estremi della finestra sono **inclusivi**
   e la carica è espressa in **ADC ns**. Lo spettro delle waveform selezionate
   viene adattato con `CalibrationHistogram` e `multigauss_iminuit` di waffles.
   Sono salvati guadagno, larghezza del piedistallo, SNR (guadagno diviso
   larghezza del piedistallo), incertezze del fit e qualità dell'adattamento.
   Un fit dell'insieme originale, prima della selezione, è salvato
   separatamente per confronto.
5. Per i fit accettati, i candidati 1 PE sono identificati usando la posizione
   e la larghezza del picco nel fit. La loro waveform media dà l'ampiezza SPE;
   una correzione per la contaminazione 0/2 PE stimata dalle gaussiane porta
   al dynamic range `2**14 / ampiezza_SPE` in PE, come nell'analisi del collega.
   `spe_metrics.json` conserva i conteggi, la correzione e i controlli di
   qualità. Le incertezze SPE/DR riportate sono solo statistiche: non includono
   l'incertezza del fit né le differenze di metodo.

I criteri sono automatici, ma un minimo numerico non dimostra da solo che i
picchi fotoelettronici siano identificati correttamente. Controllare sempre
lo stato del fit (`ok`, `review_required` o un motivo di fallimento), i grafici
e la validazione prima di usare i numeri nella tesi. La selezione di rumore
può restringere il piedistallo e quindi influenzare l'SNR; il confronto con
il fit dell'insieme originale è un controllo utile. Il controllo ai rail ADC non
esclude una saturazione analogica.

## Risultati

Con il comando di esempio, ogni punto ha una cartella propria, per esempio
`output/M1_42V85_ch18/vgain_1000/channel_18/`. Vi si trovano:

- `waveform_metrics.csv`: baseline, rumore, esito dei tagli e carica di ogni
  waveform caricata, comprese quelle scartate;
- `summary.json`, `fit_result.json`, `unselected_fit.json` e
  `integration_scan.json`: parametri, conteggi, risultati dei fit e scansione
  della finestra;
- PNG `00_original`–`04_rejected`: waveform e profili prima e dopo ogni
  selezione; `05_noise_diagnostics` e `06_selection_counts`: soglie e conteggi;
  `07_charge_histogram`, `08_charge_zoom`, `10_multigaussian_fit`: spettri,
  fit e residui; `09_pulse_zoom`, `11_integration_scan`, `12_led_discovery`:
  posizione dell'impulso e scelta della finestra. Alcuni grafici possono
  mancare se il punto fallisce prima di quella fase;
- `debug_report.pdf`, se è specificato `--debug-pdf`: raccoglie i grafici
  diagnostici del singolo canale, VGAIN e OV. I PNG restano disponibili per
  un'ispezione dettagliata.
- `spe_metrics.json` e, se disponibile, `13_spe_waveform.png`: candidati
  1 PE, ampiezza, correzione e dynamic range.

Nella radice di `--output`, `scan_summary.csv` riassume tutti i punti e
`summary_all_channels.json` conserva i risultati completi. Se un punto non
può essere analizzato, la sua cartella contiene `failure.json` e la riga
corrispondente riporta `analysis_failed`. Nel CSV le colonne `gain_adc_ns` e
`snr` sono compilate solo se il fit ha stato `ok`; anche `spe_amplitude_adc`
e `dynamic_range_pe` richiedono un'estrazione SPE con stato `ok`. Gli eventuali valori da
esaminare si trovano in `candidate_gain_adc_ns` e `candidate_snr`. Il PDF
passato a `--report-pdf` contiene **una pagina per VGAIN** del canale scelto:
istogramma della carica
selezionata, fit, residui, SNR prima e dopo la selezione, parametri e stato
della validazione. È il report compatto per confrontare lo scan; i PDF di
debug servono a controllare come sono stati ottenuti i numeri di ciascun punto.

La figura riassuntiva delle curve si crea dal CSV, per un solo bias o per più
CSV dello stesso modulo e canale:

```bash
python src/waffles/coldboxVD/thesis_review/plot_scan_summary.py \
  src/waffles/coldboxVD/thesis_review/output/M1_42V85_ch18/scan_summary.csv \
  --module M1 --channel 18 --bias 42V85 \
  --output-prefix src/waffles/coldboxVD/thesis_review/output/M1_42V85_ch18/summary_curves
```

Questo produce PNG e PDF con SNR–VGAIN e, dove il dynamic range è valido,
SNR–dynamic range. Le curve usano solo i fit `ok`; la legenda indica il DVbias
nominale. Per chiamare l'asse “OV” serve prima la conversione calibrata fra
setpoint DAPHNE e bias effettivo.

## Confronto con il programma C++ di novembre 2025

La presentazione `Argon_property.pdf` mostra per M1 canale 18 le curve
SNR–VGAIN, SNR–dynamic range e SNR–ampiezza SPE (fra le altre figure). Per il
confronto diretto con la configurazione M1 del collega, che integra i campioni
380–395 e usa la media del pretrigger 0–378, si può eseguire:

```bash
python src/waffles/coldboxVD/thesis_review/select_waveforms.py \
  --input src/waffles/coldboxVD/thesis_review/data/M1/vgain_scan_M1_DVbias_42V85 \
  --recursive --channels 18 --expected-waveforms 15000 \
  --baseline-stop 379 --baseline-estimator mean \
  --int-low 380 --int-up 395 --excursion-sigma 3 \
  --output src/waffles/coldboxVD/thesis_review/output/M1_42V85_ch18_cpp_compare \
  --report-pdf src/waffles/coldboxVD/thesis_review/output/pdf/M1_42V85_ch18_cpp_compare.pdf
```

Questa è una prova di confronto, non la scelta automatica delle finestre. Il
programma C++ ricava il livello del taglio pretrigger da un'acquisizione di
rumore separata; qui la scala RMS è stimata dai dati dello stesso VGAIN, così
non servono run precedenti. Inoltre il C++ somma i campioni in ADC tick, mentre
Waffles integra con la regola del trapezio in ADC ns: confrontare direttamente
SNR, ampiezza e dynamic range, non i gain numerici senza conversione. I punti
con fit o correzione SPE dubbi sono marcati `review_required` o esclusi.

Le figure di persistenza mostrano un campione riproducibile di al più 1000
waveform (`--plot-waveforms`), mentre il profilo medio usa tutte le waveform
della fase. I limiti grafici basati sui quantili possono nascondere code
estreme; il CSV conserva le cariche di tutte le waveform caricate. Il C++ del
collega usa una somma di campioni: dividere una carica in ADC ns per 16 ns la
porta in unità ADC campione, ma la regola trapezoidale pesa diversamente gli
estremi. Per confronti numerici esatti occorre allineare anche integrazione
e definizione della baseline.
