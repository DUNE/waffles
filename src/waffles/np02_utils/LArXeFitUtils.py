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
