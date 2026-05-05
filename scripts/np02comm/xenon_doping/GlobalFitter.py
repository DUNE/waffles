import numpy as np
from scipy.special import erfc
from dataclasses import dataclass, fields
from typing import Optional
from iminuit import Minuit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from waffles.np02_utils.AutoMap import getModuleName


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class FitParameter:
    value: float
    error: float = 0.0

    def __repr__(self):
        return f"{self.value:.4g} ± {self.error:.4g}"


@dataclass
class PerModuleResults:
    """Fit results for a single module (per-dataset parameters)."""
    # A:  Optional[FitParameter] = None
    B:  Optional[FitParameter] = None
    C:  Optional[FitParameter] = None
    offset: Optional[FitParameter] = None

    def __getitem__(self, key):           return getattr(self, key)
    def __setitem__(self, key, value):    setattr(self, key, value)
    def __contains__(self, key):
        return key in {f.name for f in fields(self)} and getattr(self, key) is not None

    def to_dict(self):
        """Returns {param_label: value} for use in model(), ignoring errors."""
        return {f.name: getattr(self, f.name).value
                for f in fields(self) if getattr(self, f.name) is not None}


@dataclass
class GlobalFitResults:
    """Fit results for global (shared) parameters + one PerModuleResults per key."""
    # tf:    Optional[FitParameter] = None
    # sigma: Optional[FitParameter] = None
    tta:   Optional[FitParameter] = None
    ttx:   Optional[FitParameter] = None

    # per-module results, keyed by dataset key
    modules: dict = None

    def __post_init__(self):
        if self.modules is None:
            self.modules = {}

    def __getitem__(self, key):        return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __contains__(self, key):
        return key in {f.name for f in fields(self)} and getattr(self, key) is not None

    def as_params_dict(self, vtype='value'):
        """
        Reconstruct a flat {minuit_param_name: 'value' or 'error } dict from stored results.
        Useful for plotting after loading results from file, bypassing Minuit entirely.
        """
        params = {}
        for f in fields(self):
            if f.name == "modules":
                continue
            fp = getattr(self, f.name)
            if fp is not None:
                params[f.name] = getattr(fp,vtype)
        for key, mod in self.modules.items():
            safe = GlobalFitter._safe_key(key)
            for label, pname in [#("A",  f"A_{safe}"),
                                 ("B",  f"B_{safe}"),
                                 ("C",  f"C_{safe}"),
                                 ("offset", f"offset_{safe}")]:
                if mod[label] is not None:
                    params[pname] = getattr(mod[label], vtype)
        return params

    def print(self):
        print("=== Global parameters ===")
        for f in fields(self):
            if f.name == "modules": continue
            fp = getattr(self, f.name)
            if fp is not None:
                print(f"  {f.name}: {fp}")
        print("=== Per-module parameters ===")
        for key, mod in self.modules.items():
            print(f"  {key}:")
            for f in fields(mod):
                fp = getattr(mod, f.name)
                if fp is not None:
                    print(f"    {f.name}: {fp}")


# ------------------------------------------------------------------
# Fitter
# ------------------------------------------------------------------
class GlobalFitter:

    # _PER_MODULE_INIT  = {"A": 100., "B": 8000., "C": 12000., "t0": 900.0}
    # _GLOBAL_INIT      = {"tf": 20.0, "tta": 900., "ttx": 2000., "sigma": 50.0}
    _PER_MODULE_INIT  = {"B": 8000., "C": 12000., "offset": 0.0}
    _GLOBAL_INIT      = {"tta": 700., "ttx": 2000.}

    def __init__(self,
                 datasets: dict[tuple[int, int], np.ndarray] = None,
                 offset_t0: float = 500.0,
                 penalty_strength: float = 300.0,
                 penalty_scale: float = 50.0,
                 error: float = 0.5,
                 prominence: float = 0.5
                 ):

        self.datasets     = datasets or {}
        self.param_names  = self._build_param_names()
        self.x            = np.linspace(0, 1024 * 16, 1024, endpoint=False)
        self.fit_results: Optional[GlobalFitResults] = None
        self.minuit:      Optional[Minuit]           = None
        self.error = error
        self.startfit = {} 
        self.tzero = {}
        self.x_keys = {}
        self.mask_key = {}
        self.penalty_strength = penalty_strength
        self.penalty_scale = penalty_scale
        self.prominence = prominence 
        self.offset_t0 = offset_t0
        self.debug = False

    # ------------------------------------------------------------------
    # Param name helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_key(key:tuple[int,int]) -> str:
        return f"{key[0]}_{key[1]}"

    def _build_param_names(self):
        return {
            key: {
                # "A":  f"A_{self._safe_key(key)}",
                "B":  f"B_{self._safe_key(key)}",
                "C":  f"C_{self._safe_key(key)}",
                "offset": f"offset_{self._safe_key(key)}",
            }
            for key in self.datasets
        }

    def _initial_params(self):
        params = dict(self._GLOBAL_INIT)
        for names in self.param_names.values():
            for label, pname in names.items():
                params[pname] = self._PER_MODULE_INIT[label]
        return params

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def expo(self, x, tau, t0):
        if tau==0:
            return np.zeros_like(x)
        return np.exp(-(x - t0) / tau)

    def _components(self, x, params, key):
        tta, ttx  = params["tta"],  params["ttx"]
        t0 = self.tzero[key]
        names     = self.param_names[key]
        B, C   = params[names["B"]], params[names["C"]]

        term_ta = self.expo(x, tta, t0) 
        term_tx = self.expo(x, ttx, t0)

        return B * term_ta / tta,  C * (term_ta - term_tx) / (tta - ttx)

    def model(self, x, params, key):
        names = self.param_names[key]
        offset = params[names["offset"]]
        return sum(self._components(x, params, key)) + offset

    # ------------------------------------------------------------------
    # Cost & minimisation
    # ------------------------------------------------------------------
    def _cost(self, datasets, **params):
        tta = params["tta"]
        ttx = params["ttx"]

        chi2 = sum(
            np.sum((y - self.model(self.x_keys[key], params, key))**2/self.error**2)
            for key, y in datasets.items()
        )
        penalty = 0.0
        order_penalty = 0.0
        if self.apply_penalty:
            delta_tau = ttx - tta # we know ttx > tta, so this is always positive
            order_penalty = 1e12 * max(0, -delta_tau)**2  # large penalty if order is wrong
            penalty   = self.penalty_strength * np.exp(-delta_tau / self.penalty_scale) if delta_tau >= 0 else 0.0

        if self.debug:
            # print(f"chi2={chi2:.2f}  penalty={penalty:.2f} order_penalty={order_penalty:.2e}  tta={tta:.1f} ttx={ttx:.1f}")
            self.values_chi2.append(chi2)
            self.values_penalty.append(penalty)
            self.values_order_penalty.append(order_penalty)

            


        return chi2 + penalty + order_penalty


    def debug_peaks(self, key=None, prominence=None):
        if key is not None:
            keys = [key]
        else:
            keys = self.datasets.keys()
        fig, axs = plt.subplots(len(keys)//2, 2, figsize=(12, 4 * (len(keys)//2)), squeeze=False)

        if prominence is None:
            prominence = self.prominence
        
        for ax, key in zip(axs.flatten(), keys):
            y = self.datasets[key]
            peaks, _ = find_peaks(y, prominence=prominence)
            prominences = peak_prominences(y, peaks)[0]

            ax.plot(self.x, y, label='Data')
            ax.plot(self.x[peaks], y[peaks], 'x', label='Peaks')
            ax.vlines(self.x[peaks], ymin=y[peaks] - prominences, ymax=y[peaks], color='C1', label='Prominence')
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Amplitude [a.u.]")
            ax.legend(title=f"{getModuleName(key[0], key[1])}: {key}")

        return fig, axs

    

    def minimize(self, fit_limit_ns=None, oneexp=False):
        if not self.datasets:
            raise ValueError("No datasets loaded.")
           
        init = self._initial_params()
        param_names = list(init.keys())


        x = np.linspace(0, 1024 * 16, 1024, endpoint=False)
        datasets = self.datasets  # no copy needed yet, slicing creates new arrays

        for key, y in datasets.items():
            # Needed to search peak because at 10 ppm M7 has a bump
            peaks, _ = find_peaks(y, prominence=self.prominence)

            self.tzero[key] = self.x[peaks[0]] 
            self.startfit[key] = self.tzero[key] + self.offset_t0
            # print(f"Module {key}: t0 estimate = {self.startfit[key]:.1f} ns, module: {getModuleName(key[0], key[1])}")
            mask = np.ones(len(self.x), dtype=bool)
            mask &= x >= self.startfit[key]
            if fit_limit_ns is not None:
                mask &= x <= fit_limit_ns
            self.x_keys[key] =  x[mask]
            self.mask_key[key] = mask

        datasets = {key: y[self.mask_key[key]] for key, y in self.datasets.items()}

        self.values_chi2 = []
        self.values_penalty = []
        self.values_order_penalty = []

        # Wrap _cost to accept positional args since Minuit calls it that way
        def cost_wrapper(*args):
            params = dict(zip(param_names, args))
            return self._cost(datasets, **params)

        cost_wrapper.ndata = sum(len(y) for y in datasets.values()) # for proper error scaling in Minuit

        m = Minuit(cost_wrapper, *init.values(), name=param_names)
        # Global params
        # m.limits["tf"]    = (0, None)
        m.limits["tta"]   = (200, None)
        m.limits["ttx"]   = (0, None)
        # m.limits["sigma"] = (0, None)

        if oneexp:
            m.values["ttx"] = 0
            m.fixed["ttx"] = True
            self.apply_penalty = False
        else:
            self.apply_penalty = True

        # Per-module params
        for names in self.param_names.values():
            # m.limits[names["A"]]  = (0, None)
            m.limits[names["B"]]  = (0, None)
            m.limits[names["C"]]  = (0, None)
            # m.limits[names["offset"]]  = (None, 0)
            if oneexp:
                m.values[names["C"]] = 0.0
                m.fixed[names["C"]] = True 
                # m.fixed[names["t0"]] = True

        m.errordef = Minuit.LEAST_SQUARES

        m.migrad()
        m.migrad()
        m.hesse()
        self.minuit = m
        self.fit_results = self._store_results(m)
        return self.fit_results

    def _store_results(self, m: Minuit) -> GlobalFitResults:
        results = GlobalFitResults(
            # tf    = FitParameter(m.values["tf"],    m.errors["tf"]),
            # sigma = FitParameter(m.values["sigma"], m.errors["sigma"]),
            tta   = FitParameter(m.values["tta"],   m.errors["tta"]),
            ttx   = FitParameter(m.values["ttx"],   m.errors["ttx"]),
        )
        for key, names in self.param_names.items():
            results.modules[key] = PerModuleResults(
                # A  = FitParameter(m.values[names["A"]],  m.errors[names["A"]]),
                B  = FitParameter(m.values[names["B"]],  m.errors[names["B"]]),
                C  = FitParameter(m.values[names["C"]],  m.errors[names["C"]]),
                offset = FitParameter(m.values[names["offset"]], m.errors[names["offset"]]),
            )
        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _resolve_params(self, params_override=None):
        """
        Returns a flat param dict to use for plotting.
        Priority: params_override > fit_results > error.
        params_override can be:
          - a flat dict  {minuit_param_name: value}
          - a GlobalFitResults instance
        """
        if params_override is not None:
            if isinstance(params_override, GlobalFitResults):
                return params_override.as_params_dict()
            return params_override  # assume flat dict

        if self.fit_results is not None:
            return self.fit_results.as_params_dict()

        raise RuntimeError("No fit results available. Run minimize() or pass params_override.")

    def plot_results(self, keys = [], ncols=2, params_override=None, logscale=False):

        params   = self._resolve_params(params_override)
        if params_override is None and self.fit_results is not None:
            params_err = self.fit_results.as_params_dict(vtype='error')
        else:
            params_err = {}


        datasets = {key: y for key, y in self.datasets.items() if not keys or key in keys}
        nkeys    = len(datasets)
        nrows    = int(np.ceil(nkeys / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for ax, (key, y) in zip(axs.flatten(), datasets.items()):
            x = self.x[self.mask_key[key]]
            # comp_A, comp_B, comp_C = self._components(x, params, key)
            # y_fit = comp_A + comp_B + comp_C
            comp_B, comp_C = self._components(x, params, key)
            y_fit = comp_B + comp_C + params[self.param_names[key]["offset"]]

            ax.plot(self.x, y,     color="black", lw=1,   alpha=0.7, label="Data")
            ax.plot(x, y_fit, color="red",   lw=2,   ls="--",   label="Total fit")
            # ax.plot(x, comp_A, lw=1.5, ls=":", label=r"$A \cdot f(\tau_f)$")
            ax.plot(x, comp_B, lw=1.5, ls=":", label=r"$\frac{B}{\tau_{TA}} \cdot e^{-t/\tau_{TA}}$")
            ax.plot(x, comp_C, lw=1.5, ls=":", label=r"$\frac{C}{\tau_{TA} - \tau_{TX}} \cdot [e^{-t/\tau_{TA}}-e^{-t/\tau_{TX}}]$")

            ax.legend(loc="upper right", fontsize=12)
            ep, ch = key[0], key[1]
            ax.set_title(f"{getModuleName(ep, ch)}: {ep}-{ch}", fontsize=12)
            ax.set_xlabel("Time [ns]", fontsize=12)
            ax.set_ylabel("Amplitude [a.u.]", fontsize=12) 

            # We change the fontsize of minor ticks label 
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

            if logscale:
                ax.set_yscale("log")


            # Parameter text box
            names = self.param_names[key]
            params_text = [
                rf"$B$ = {params[names['B']]:.1f}",
                rf"$C$ = {params[names['C']]:.1f}",
                rf"$offset$ = {params[names['offset']]:.3f}",
                rf"$\tau_{{ta}}$ = {params['tta']:.1f}",
                rf"$\tau_{{tx}}$ = {params['ttx']:.1f}"
            ]
            if params_err:
                for i, label in enumerate(["B", "C"]):
                    err = params_err.get(names[label], 0)
                    params_text[i] += rf" $\pm$ {err:.1f}"
                params_text[2] += rf" $\pm$ {params_err.get(names['offset'], 0):.3f}"
                for i, label in enumerate(["tta", "ttx"]):
                    err = params_err.get(label, 0)
                    params_text[i+3] += rf" $\pm$ {err:.1f}"
            lines = "\n".join(params_text)
            ax.text(0.60, 0.55, lines, transform=ax.transAxes, fontsize=10,
                    va="top", ha="left")

        for ax in axs.flatten()[nkeys:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig, axs
