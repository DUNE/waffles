import numpy as np
from typing import List, Union

import waffles.utils.numerical_utils as wun

from waffles.Exceptions import GenerateExceptionMessage


class TrackedHistogram:

    """
    This class implements a histogram which keeps
    track of which samples contribute to which bin,
    by keeping its indices with respect to some
    assumed ordering.

    Attributes
    ----------
    BinsNumber : int
        Number of bins in the histogram. It must
        be greater than 1.
    Edges : unidimensional numpy array of floats
        Its length must match BinsNumber + 1. The
        i-th bin, with i = 0, ..., BinsNumber - 1,
        contains the number of occurrences between
        Edges[i] and Edges[i+1].
    MeanBinWidth : float
        The mean difference between two consecutive
        edges. It is computed as
        (Edges[BinsNumber] - Edges[0]) / BinsNumber.
        For histograms with an uniform binning, this
        value matches (Edges[i+1] - Edges[i]) for
        whichever i.
    Counts : unidimensional numpy array of integers
        Its length must match BinsNumber. Counts[i]
        gives the number of occurrences in the i-th
        bin, with i = 0, ..., BinsNumber - 1.
    Indices : list of lists of integers
        Its length must match BinsNumber. Indices[i]
        gives the list of indices, with respect to
        some ordering, of the samples which contributed
        to the i-th bin. Note that the length of
        Indices[i] must match Counts[i], for
        i = 0, ..., BinsNumber - 1.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
            self, bins_number: int,
            edges: np.ndarray,
            counts: np.ndarray,
            indices: List[List[int]]):
        """
        TrackedHistogram class initializer. It is the
        caller's responsibility to check the types of the
        input parameters. No type checks are perfomed here.

        Parameters
        ----------
        bins_number : int
        edges : unidimensional numpy array of floats
        counts : unidimensional numpy array of integers
        indices : list of lists of integers
        """

        if bins_number < 2:
            raise Exception(GenerateExceptionMessage(
                1,
                'TrackedHistogram.__init__()',
                f"The given bins number ({bins_number})"
                " must be greater than 1."))
        if len(edges) != bins_number + 1:
            raise Exception(GenerateExceptionMessage(
                2,
                'TrackedHistogram.__init__()',
                f"The length of the 'edges' parameter ({len(edges)})"
                f" must match 'bins_number + 1' ({bins_number + 1})."))
        if len(counts) != bins_number:
            raise Exception(GenerateExceptionMessage(
                3,
                'TrackedHistogram.__init__()',
                f"The length of the 'counts' parameter ({len(counts)})"
                f" must match 'bins_number' ({bins_number})."))
        if len(indices) != bins_number:
            raise Exception(GenerateExceptionMessage(
                4,
                'TrackedHistogram.__init__()',
                f"The length of the 'indices' parameter ({len(indices)})"
                f" must match 'bins_number' ({bins_number})."))
        for i in range(bins_number):
            if len(indices[i]) != counts[i]:
                raise Exception(GenerateExceptionMessage(
                    5,
                    'TrackedHistogram.__init__()',
                    f"The length of 'indices[{i}]' parameter"
                    f" ({len(indices[i])}) must match 'counts[{i}]'"
                    f" ({counts[i]})."))
        self.__bins_number = bins_number
        self.__edges = edges
        self.__mean_bin_width = (
            self.__edges[
                self.__bins_number] - self.__edges[0]) / self.__bins_number
        self.__counts = counts
        self.__indices = indices

    # Getters
    @property
    def BinsNumber(self):
        return self.__bins_number

    @property
    def Edges(self):
        return self.__edges

    @property
    def MeanBinWidth(self):
        return self.__mean_bin_width

    @property
    def Counts(self):
        return self.__counts

    @property
    def Indices(self):
        return self.__indices

    @classmethod
    def from_samples(
            cls, samples: List[Union[int, float]],
            bins_number: int,
            domain: np.ndarray) -> 'TrackedHistogram':
        """
        Alternative initializer for the TrackedHistogram class.
        It creates a tracked histogram from a list of samples.

        Parameters
        ----------
        samples : list of int or float
            The samples to add to the tracked histogram
        bins_number : int
            The number of bins for the created histogram.
            It must be greater than 1.
        domain : np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the created
            histogram. Any sample which falls outside
            this range is ignored.

        Returns
        ----------
        output : TrackedHistogram
            The created tracked histogram
        """

        if bins_number < 2:
            raise Exception(GenerateExceptionMessage(
                1,
                'TrackedHistogram.from_samples()',
                f"The given bins number ({bins_number})"
                " must be greater than 1."))
        if np.ndim(domain) != 1 or len(domain) != 2:
            raise Exception(GenerateExceptionMessage(
                2,
                'TrackedHistogram.from_samples()',
                "The 'domain' parameter must be a "
                "2x1 numpy array."))
        edges = np.linspace(domain[0],
                            domain[1],
                            num=bins_number + 1,
                            endpoint=True)

        counts, indices = wun.histogram1d(
            np.array(samples),
            bins_number,
            domain,
            keep_track_of_idcs=True)
        return cls(
            bins_number,
            edges,
            counts,
            indices)
