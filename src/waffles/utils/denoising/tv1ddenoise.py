import numpy as np
import ctypes
from tqdm import tqdm

from typing import Tuple, List, Dict, Callable, Optional

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
import os



class Denoise:
    def __init__(self, npts: Optional[int] = 0):
        """This class applied denosing filtering from C++ code. 
        The code is provided here: https://lcondat.github.io/index.html

        Methods
        ----------
        apply_denoise
            Apply denoising on Waveform.adcs or numpy array objects. Returns
            a numpy array
        create_filtered_waveforms
            Apply denoising over all waveforms of a WaveformSet. Adds it to the
            object as `filtered`
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = f"{dir_path}/tv1d_cpp/"

        self.load_filter(dir_path)

        self.__hasnptsset = False
        self.__npts = npts
        if npts > 0:
            self.__setupreturn()

        
    def apply_denoise(self, raw: Waveform.adcs, filter: float = 0) -> np.ndarray:
        """Apply denoising on Waveform.adcs or numpy array objects.

        Parameters
        ----------
        raw: Waveform.adcs (or numpy array)
            Raw waveform in which you wish to apply the filtering
        filter: float
            Filtering level.
            ATTENTION: the filtering level depends on the length of the
            waveform. Therefore, for different lengths, one need to test what
            is the appropriate filter value.

        Returns
        ----------
        output: np.ndarray
            filtered waveform as numpy array (float32)
        """
        if not self.__hasnptsset:
            self.__npts = len(raw)
            self.__setupreturn()
        return self.tv1filter.TV1D_denoise(raw.astype(np.float32), self.__npts, filter)

    def apply_denoise_inplace(self, raw: Waveform.adcs, filter: float = 0) -> np.ndarray:
        """Apply denoising on Waveform.adcs or numpy array objects.

        Parameters
        ----------
        raw: Waveform.adcs (or numpy array)
            Raw waveform in which you wish to apply the filtering
        filter: float
            Filtering level.
            ATTENTION: the filtering level depends on the length of the
            waveform. Therefore, for different lengths, one need to test what
            is the appropriate filter value.

        """
        if not self.__hasnptsset:
            self.__npts = len(raw)
            self.__setupreturn()
            self.output = np.zeros(self.__npts, dtype=np.float32)
        self.tv1filter.TV1D_denoise_inplace(raw.astype(np.float32), self.__npts, filter, self.output)
        return self.output

    def create_filtered_waveforms(self, wfset:WaveformSet, filter: float = 0, show_progress: bool = False):
        """Apply denoising on all waveforms of a WaveformSet. Saved the filtered waveforms in each `waveforms` object as `filtered`

        Parameters
        ----------
        wfset: WaveformSet
            Well...
        filter: float
            Filtering level.
            ATTENTION: the filtering level depends on the length of the
            waveform. Therefore, for different lengths, one need to test what
            is the appropriate filter value.
        show_progress: bool
            If True, will show tqdm progress bar
        """
        for i in tqdm(range(len(wfset.waveforms)), disable=not show_progress):
            wfset.waveforms[i].filtered = self.apply_denoise(wfset.waveforms[i].adcs, filter)


    def load_filter(self, dir_path:str):
        
        if not os.path.isfile(f'{dir_path}/tv1ddenoise.so'):
            if os.path.isfile(f'{dir_path}/tv1ddenoise.c'):
                print("Installing denoise...")
                os.system(f'g++ -O3 -fPIC -shared {dir_path}/tv1ddenoise.c -o {dir_path}/tv1ddenoise.so')
            else:
                raise Exception(f"No tv1ddenoise.c found at {dir_path}")
                return

        self.tv1filter = ctypes.cdll.LoadLibrary(f"{dir_path}/tv1ddenoise.so")
        self.tv1filter.TV1D_denoise.argtypes = [ np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int , ctypes.c_double ]
        try:
            self.tv1filter.TV1D_denoise_inplace.argtypes = [ np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int , ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.float32) ]
        except Exception as e:
            print(f"Error setting up TV1D_denoise_inplace: {e}")
            raise


    def __setupreturn(self):
        self.__hasnptsset = True
        self.tv1filter.TV1D_denoise.restype = np.ctypeslib.ndpointer(dtype=np.float32, shape=(self.__npts,))
        self.tv1filter.TV1D_denoise_inplace.restype = None
        

        











