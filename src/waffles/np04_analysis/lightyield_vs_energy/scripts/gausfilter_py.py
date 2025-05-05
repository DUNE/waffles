import numpy as np
import matplotlib.pyplot as plt
import argparse

def gauss_filter(x, sigma):
    return np.exp(-0.5 * (x/sigma)**2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian filter')
    parser.add_argument('-fc','--cutoff', type=float, help='Cutoff frequencty', default=8.)
    parms = vars(parser.parse_args())

    fc = parms['cutoff']

    npoints = 1024//2 # The fft has half the length of the waveforms
    FFTFreq = 32.5 # The frequency of the fft in MHz

    binwidth = FFTFreq/npoints # Each bin of the fft has a width of binwidth in MHz

    x_freq = np.linspace(0, FFTFreq, npoints, endpoint=False)
    x_tick = np.linspace(0, npoints, npoints, endpoint=False)

    # ---- If working with frequency ----
    gaussian_stddev = fc/np.sqrt(np.log(2))
    
    # In frequency, you can compute like this
    y_freq = [ gauss_filter(i, gaussian_stddev) for i in x_freq ]
    # # Another possibility
    y_freq = [ gauss_filter(i*binwidth, gaussian_stddev) for i in x_tick ]

    # ---- If working with ticks ----

    fc_in_ticks = fc/binwidth
    gaussian_stddev_in_ticks = fc_in_ticks/np.sqrt(np.log(2))
    # In ticks, you can compute like this
    y_ticks = [ gauss_filter(i, gaussian_stddev_in_ticks) for i in x_tick ]


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x_freq, y_freq)
    ax[0].vlines(fc, 0, 1/np.sqrt(2), color='r', linestyle='--')
    ax[0].vlines(gaussian_stddev*2.355/2, 0, 0.5, color='b', linestyle='--')
    ax[0].hlines(1/np.sqrt(2), 0, fc, color='r', linestyle='--')
    ax[0].hlines(0.5, 0, gaussian_stddev*2.355/2, color='b', linestyle='--')

    ax[0].set_title(f'Filter in Frequency:\ncutoff = {fc} MHz\nsigma = {gaussian_stddev:.2f}')
    ax[0].set_xlabel('Frequency (MHz)')
    ax[0].set_ylabel('Gain')

    ax[1].plot(x_tick, y_ticks)
    ax[1].set_title(f'Filter in Ticks:\ncutoff = {fc_in_ticks:.2f} ticks\nsigma = {gaussian_stddev_in_ticks:.2f}')
    ax[1].set_xlabel('Ticks')
    ax[1].set_ylabel('Gain')
    ax[1].vlines(fc_in_ticks, 0, 1/np.sqrt(2), color='r', linestyle='--')
    ax[1].vlines(gaussian_stddev_in_ticks*2.355/2, 0, 0.5, color='b', linestyle='--')
    ax[1].hlines(1/np.sqrt(2), 0, fc_in_ticks, color='r', linestyle='--')
    ax[1].hlines(0.5, 0, gaussian_stddev_in_ticks*2.355/2, color='b', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/scripts/prova.jpg')

    



