import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.Exceptions as we


class RandomWindowPedestalBaseliner(WfAna):
    """Baseline (pedestal) estimator robust to dark-noise pulses by sampling
    multiple random windows in the pretrigger region and taking the median
    of the window means (MATLAB calculateMedianPedestal equivalent).

    Concept
    -------
    - Choose `number_of_regions` random window starts within the pretrigger
      range [0, pedestal_limit)
    - Compute the mean ADC value in each window of length `average_width`
    - Use the median of these means as the pedestal (robust to outliers)
    - Also return 25% and 75% quantiles of the window means

    Input parameters (IPDict)
    -------------------------
    pedestal_limit : int
        Upper bound (exclusive) of the pretrigger region, expressed in the
        same "time coordinate" convention used elsewhere in waffles:
        i.e., it is compared against (sample_index + waveform.time_offset).
        Internally, local index is `pedestal_limit - waveform.time_offset`.
    average_width : int
        Number of samples per window (exactly this many samples).
    number_of_regions : int
        Number of random windows to sample.
    seed : int
        RNG seed for reproducibility.
    store_window_means : bool, optional (default False)
        If True, stores the array of window means in the result (can be large).

    Result (WfAnaResult)
    --------------------
    baseline : float
        Pedestal estimate (median of window means).
    baseline_std : float
        Standard deviation of the window means.
    baseline_25 : float
        25% quantile of window means.
    baseline_75 : float
        75% quantile of window means.
    rng_state : dict
        RNG internal state after initialization (for debugging/repro).
    window_means : np.ndarray (optional)
        Only if store_window_means=True.
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        self.__pedestal_limit = int(input_parameters["pedestal_limit"])
        self.__average_width = int(input_parameters["average_width"])
        self.__number_of_regions = int(input_parameters["number_of_regions"])
        self.__seed = int(input_parameters["seed"])
        self.__store_window_means = bool(input_parameters.get("store_window_means", False))
        super().__init__(input_parameters)

    # Getters
    @property
    def pedestal_limit(self) -> int:
        return self.__pedestal_limit

    @property
    def average_width(self) -> int:
        return self.__average_width

    @property
    def number_of_regions(self) -> int:
        return self.__number_of_regions

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def store_window_means(self) -> bool:
        return self.__store_window_means

    def analyse(self, waveform: WaveformAdcs) -> None:
        # Convert pedestal_limit from "global time coordinate" to local ADC index
        ped_lim_local = self.__pedestal_limit - waveform.time_offset

        # Pretrigger region is waveform.adcs[0:ped_lim_local] (exclusive upper bound)
        # Window start must satisfy start + average_width <= ped_lim_local
        start_max = ped_lim_local - self.__average_width

        rng = np.random.default_rng(self.__seed)
        rng_state = rng.bit_generator.state

        # Sample random window starts uniformly (inclusive 0 .. start_max)
        starts = rng.integers(low=0, high=start_max + 1, size=self.__number_of_regions, endpoint=False)

        # Compute means per window
        adcs = waveform.adcs
        window_means = np.empty(self.__number_of_regions, dtype=float)
        for i, s in enumerate(starts):
            window_means[i] = float(np.mean(adcs[s : s + self.__average_width]))

        q25, q50, q75 = np.quantile(window_means, [0.25, 0.50, 0.75])

        # RMS of the full waveform after baseline subtraction
        # (note: this includes any real pulses; it's a "waveform RMS" metric)
        waveform_rms = float(np.sqrt(np.mean((adcs) ** 2)))

        result_kwargs = dict(
            baseline=float(q50),
            baseline_std=float(np.std(window_means)),
            waveform_rms=waveform_rms,
            baseline_25=float(q25),
            baseline_75=float(q75),
            rng_state=rng_state,
        )
        if self.__store_window_means:
            result_kwargs["window_means"] = window_means

        self._WfAna__result = WfAnaResult(**result_kwargs)
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(input_parameters: IPDict, points_no: int) -> None:
        required = ["pedestal_limit", "average_width", "number_of_regions", "seed"]
        for k in required:
            if k not in input_parameters:
                raise Exception(we.GenerateExceptionMessage(
                    1,
                    "RandomWindowPedestalBaseliner.check_input_parameters()",
                    f"Missing required input parameter '{k}'."
                ))

        pedestal_limit = int(input_parameters["pedestal_limit"])
        average_width = int(input_parameters["average_width"])
        number_of_regions = int(input_parameters["number_of_regions"])

        if pedestal_limit < 1 or pedestal_limit > points_no:
            raise Exception(we.GenerateExceptionMessage(
                2,
                "RandomWindowPedestalBaseliner.check_input_parameters()",
                f"pedestal_limit={pedestal_limit} must be in [1, {points_no}]."
            ))

        if average_width < 1:
            raise Exception(we.GenerateExceptionMessage(
                3,
                "RandomWindowPedestalBaseliner.check_input_parameters()",
                f"average_width={average_width} must be >= 1."
            ))

        # Need at least one valid start index in [0, pedestal_limit-average_width]
        if average_width > pedestal_limit:
            raise Exception(we.GenerateExceptionMessage(
                4,
                "RandomWindowPedestalBaseliner.check_input_parameters()",
                f"average_width={average_width} is too long for pedestal_limit={pedestal_limit}."
            ))

        if number_of_regions < 1:
            raise Exception(we.GenerateExceptionMessage(
                5,
                "RandomWindowPedestalBaseliner.check_input_parameters()",
                f"number_of_regions={number_of_regions} must be >= 1."
            ))