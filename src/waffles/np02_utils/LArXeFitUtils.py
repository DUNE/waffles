from dataclasses import dataclass, fields
from typing import Optional

@dataclass
class FitParameter:
    value: float
    error: float

@dataclass
class FitResults:
    A:     Optional[FitParameter] = None
    fp:    Optional[FitParameter] = None
    t1:    Optional[FitParameter] = None
    t3:    Optional[FitParameter] = None
    fs:    Optional[FitParameter] = None
    td:    Optional[FitParameter] = None
    t0:    Optional[FitParameter] = None
    sigma: Optional[FitParameter] = None

    def __getitem__(self, key: str) -> FitParameter:
        return getattr(self, key)

    def __setitem__(self, key: str, value: FitParameter):
        if key in {f.name for f in fields(self)}:
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid fit parameter.")

    def __contains__(self, key: str) -> bool:
        return key in {f.name for f in fields(self)} and getattr(self, key) is not None


@dataclass
class FitInitParams:
    """
    Initial values and limits for the scintillation fit minimizer.
    Create one per scinttype, or override per-channel via yaml.
    """
    # --- Initial values ---
    A: float = 10e3
    fp: float = 0.2
    t1: float = 25.0
    t3: float = 600.0
    td: float = 3200.0
    fs_frac: float = 0.3 / (1 - 0.2)   # fs / (1 - fp)
    sigma: float = 20.0
    t0: float = 0.0          # will typically be overridden dynamically
    initialized: bool = False  # flag to indicate if initial parameters have been set (either via classmethod or yaml)

    # --- Limits ---
    A_limits:      tuple = (0, None)
    fp_limits:     tuple = (0, 1)
    t1_limits:     tuple = (2, 50)
    t3_limits:     tuple = (10, 5000)
    td_limits:     tuple = (10, 5000)
    fs_frac_limits:tuple = (0, 1)
    sigma_limits:  tuple = (5, 100)
    t0_limits:     tuple = (None, None)  # set dynamically in minimizer

    @classmethod
    def for_lar(cls) -> "FitInitParams":
        return cls(fp=0.3, t1=25.0, t3=1400.0, sigma=40.0, initialized=True)

    @classmethod
    def for_larxe(cls) -> "FitInitParams":
        return cls(fp=0.2, t1=25.0,
                   fs_frac=0.7 / (1 - 0.2), initialized=True)

    @classmethod
    def for_lar_oneexp(cls) -> "FitInitParams":
        return cls(fp=0.95, t1=25.0, t3=35.0,
                   t3_limits=(0, 100), initialized=True)

    def update_from_dict(self, d: dict) -> None:
        """Patch any field from a yaml-loaded dict, warn on unknown keys."""
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: '{k}' is not a valid FitInitParams field, ignoring.")
        self.initialized = True

