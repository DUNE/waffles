#!/usr/bin/env python3
"""Studia la risposta luminosa di APA 1 rispetto all'energia cinetica media del fascio.

INPUT: apa1_population_fit_results.csv prodotto da plot_apa12_trigger_distributions.py;
       data/np04_beam_particle_content.csv con i tassi delle specie in Hz.
       I tassi sono previsioni simulate per H4-VLE (Tabella IV di
       Charitonidis ed Efthymiopoulos, PRAB 20, 111001, 2017), non conteggi
       misurati in questa analisi. Il 5% sul momento è un'ipotesi conservativa.
OUTPUT: apa1_linearity.png, apa1_linearity_points.csv, apa1_linearity_fits.csv,
        apa1_linearity_report.txt nella stessa cartella dei fit di popolazione.
ESECUZIONE (da scripts/review):
    python plot_apa1_linearity.py \
      --fit-results ../../output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv \
      --composition ../../data/np04_beam_particle_content.csv \
      --relative-momentum-error 0.05

La barra x rappresenta la dispersione efficace in energia cinetica del campione:
la dispersione tra specie e il contributo del 5% sul momento sono combinati
in quadratura. Il contributo del momento e' propagato tramite la derivata
della media K.
Il PDF descrive +/-5% come accettanza di momento, non come errore gaussiano
1-sigma: qui si usa conservativamente come incertezza efficace 1-sigma.
L'incertezza dei pesi simulati non e' fornita e non viene inventata. Le
incertezze dei fit PE sono statistiche locali e condizionate.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
from scipy.stats import chi2


MASS_GEV = {"e": 0.00051099895, "k": 0.493677, "p": 0.938272088,
            "pi": 0.13957039}
MOMENTA = (1, 2, 3, 5, 7)


def weighted_kinetic_energy(momentum, rates, relative_p_error):
    """Restituisce K_eff, contributo dal momento, RMS tra specie e totale."""
    species = tuple(MASS_GEV)
    values = np.array([float(rates[s]) for s in species])
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError(f"Tassi non validi a {momentum} GeV/c")
    if relative_p_error < 0:
        raise ValueError("Incertezza relativa non negativa richiesta")
    weights = values / values.sum()
    masses = np.array([MASS_GEV[s] for s in species])
    kinetic = np.sqrt(momentum**2 + masses**2) - masses
    mean = float(weights @ kinetic)
    mixture_var = float(weights @ (kinetic - mean)**2)
    derivative = float(weights @ (momentum / np.sqrt(momentum**2 + masses**2)))
    momentum_var = (derivative * momentum * relative_p_error)**2
    momentum_error = math.sqrt(momentum_var)
    mixture_rms = math.sqrt(mixture_var)
    effective_spread = math.hypot(momentum_error, mixture_rms)
    return mean, momentum_error, mixture_rms, effective_spread


def read_points(args):
    with args.composition.open(newline="") as handle:
        composition = {int(float(row["Momentum [GeV/c]"])): row
                       for row in csv.DictReader(handle)}
    with args.fit_results.open(newline="") as handle:
        fit_rows = {int(float(row["momentum_GeV_c"])): row
                    for row in csv.DictReader(handle)}
    points = []
    for momentum in MOMENTA:
        if momentum not in composition or momentum not in fit_rows:
            raise ValueError(f"Mancano composizione o fit per {momentum} GeV/c")
        row = fit_rows[momentum]
        expected_model = "langauss" if momentum == 1 else "langauss_plus_gaussian"
        if row["status"] != "success" or row["model"] != expected_model:
            raise ValueError(f"Fit non valido o modello inatteso a {momentum} GeV/c: "
                             f"{row['status']}, {row['model']}")
        rates = {species: composition[momentum][f"{species} [Hz]"]
                 for species in MASS_GEV}
        x, momentum_error, mixture_rms, effective_spread = weighted_kinetic_energy(
            momentum, rates, args.relative_momentum_error)
        y_column = "langauss_peak" if momentum == 1 else "gaussian_mean"
        y = float(row[y_column])
        sy = float(row[y_column + "_error"])
        if not all(map(math.isfinite,
                       (x, momentum_error, mixture_rms, effective_spread, y, sy))):
            raise ValueError(f"Valori non finiti a {momentum} GeV/c")
        if effective_spread <= 0 or sy <= 0:
            raise ValueError(f"Incertezze non valide a {momentum} GeV/c")
        points.append(dict(momentum_GeV_c=momentum, kinetic_mean_GeV=x,
                           momentum_error_GeV=momentum_error,
                           mixture_rms_GeV=mixture_rms,
                           effective_kinetic_energy_spread_GeV=effective_spread,
                           response_PE=y,
                           response_error_PE=sy, response_estimator=y_column))
    return points


def fit_line(points, label):
    x = np.array([p["kinetic_mean_GeV"] for p in points])
    sx = np.array([p["effective_kinetic_energy_spread_GeV"] for p in points])
    y = np.array([p["response_PE"] for p in points])
    sy = np.array([p["response_error_PE"] for p in points])
    initial = np.polyfit(x, y, 1)
    model = odr.Model(lambda beta, xx: beta[0] * xx + beta[1])
    result = odr.ODR(odr.RealData(x, y, sx=sx, sy=sy), model,
                     beta0=initial, maxit=1000).run()
    if result.info not in (1, 2, 3, 4):
        raise RuntimeError(f"ODR non convergente ({label}): {result.stopreason}")
    slope, intercept = map(float, result.beta)
    # Il rapporto x/y di ODR viene trattato come noto; la matrice non scalata
    # evita di nascondere una cattiva aderenza gonfiando gli errori parametrici.
    errors = np.sqrt(np.diag(result.cov_beta))
    predicted = slope * x + intercept
    chi_square = float(np.sum((y - predicted)**2 / (sy**2 + (slope * sx)**2)))
    ndf = len(points) - 2
    r2 = 1 - float(np.sum((y - predicted)**2) / np.sum((y - y.mean())**2))
    return dict(model=label, momenta=";".join(str(p["momentum_GeV_c"]) for p in points),
                slope_PE_per_GeV=slope, slope_error_PE_per_GeV=float(errors[0]),
                intercept_PE=intercept, intercept_error_PE=float(errors[1]),
                chi2=chi_square, ndf=ndf, chi2_per_ndf=chi_square / ndf,
                p_value=float(chi2.sf(chi_square, ndf)), r_squared=r2)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(points, fit_all, fit_four, destination):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.linewidth": 1.0, "savefig.dpi": 300})
    fig = plt.figure(figsize=(10.8, 7.5))
    layout = fig.add_gridspec(2, 1, height_ratios=[3.6, 1], hspace=0.08)
    axis = fig.add_subplot(layout[0])
    residual_axis = fig.add_subplot(layout[1], sharex=axis)
    x = np.array([p["kinetic_mean_GeV"] for p in points])
    sx = np.array([p["effective_kinetic_energy_spread_GeV"] for p in points])
    y = np.array([p["response_PE"] for p in points])
    sy = np.array([p["response_error_PE"] for p in points])
    grid = np.linspace(max(0, x.min() - 0.35), x.max() + 0.35, 300)
    line_all = fit_all["slope_PE_per_GeV"] * grid + fit_all["intercept_PE"]
    line_four = fit_four["slope_PE_per_GeV"] * grid + fit_four["intercept_PE"]
    axis.plot(grid, line_all, color="#D55E00", linewidth=2.1,
              label="Linear fit: 1–7 GeV/c")
    axis.plot(grid, line_four, color="#173F6B", linewidth=2.2,
              linestyle="--", label="Linear fit: 2–7 GeV/c")
    axis.errorbar(x[1:], y[1:], xerr=sx[1:], yerr=sy[1:], fmt="o",
                  markersize=7, capsize=3, color="#0072B2", ecolor="#0072B2",
                  label="Gaussian mean (2–7 GeV/c)", zorder=4)
    axis.errorbar(x[:1], y[:1], xerr=sx[:1], yerr=sy[:1], fmt="D",
                  markersize=7, capsize=3, color="#C33232", ecolor="#C33232",
                  label="Langauss peak (1 GeV/c)", zorder=5)
    axis.text(0.55, 0.97, r"$\bf{ProtoDUNE\!-\!HD}$ Work in Progress",
              transform=axis.transAxes, ha="left", va="top", fontsize=11,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    axis.set_ylabel(r"$\langle N_{\mathrm{PE}}\rangle_{\mathrm{APA\,1}}$")
    axis.legend(loc="upper left", facecolor="white", framealpha=1,
                edgecolor="0.7", fontsize=9)
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(0, max(y) * 1.13)
    def fit_text(fit):
        return (
            rf"$m = ({fit['slope_PE_per_GeV']:.1f} \pm "
            rf"{fit['slope_error_PE_per_GeV']:.1f})$ PE/GeV" "\n"
            rf"$q = ({fit['intercept_PE']:.1f} \pm "
            rf"{fit['intercept_error_PE']:.1f})$ PE" "\n"
            rf"$\chi^2/\mathrm{{ndf}} = "
            rf"{fit['chi2']:.2f}/{fit['ndf']} = "
            rf"{fit['chi2_per_ndf']:.2f}$" "\n"
            rf"$R^2 = {fit['r_squared']:.3f}$"
        )
    fit_summary = (
        "Fit including 1 GeV/c\n" + fit_text(fit_all) +
        "\n\nFit excluding 1 GeV/c\n" + fit_text(fit_four)
    )
    axis.text(0.98, 0.04, fit_summary, transform=axis.transAxes,
              ha="right", va="bottom", fontsize=9.5, linespacing=1.25,
              bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.98,
                    "boxstyle": "square,pad=0.55"}, zorder=6)
    residual_all = y - (fit_all["slope_PE_per_GeV"] * x + fit_all["intercept_PE"])
    residual_four = y[1:] - (fit_four["slope_PE_per_GeV"] * x[1:] + fit_four["intercept_PE"])
    residual_axis.axhline(0, color="0.3", linewidth=1)
    residual_axis.errorbar(x, residual_all, xerr=sx, yerr=sy, fmt="o", capsize=2,
                           color="#D55E00", label="1–7 GeV/c")
    residual_axis.errorbar(x[1:], residual_four, xerr=sx[1:], yerr=sy[1:],
                           fmt="s", markerfacecolor="white", capsize=2,
                           color="#173F6B", label="2–7 GeV/c")
    residual_axis.set_ylabel("Residual [PE]")
    residual_axis.set_xlabel(r"$K_{\mathrm{eff}}$ [GeV]")
    residual_axis.legend(loc="upper left", frameon=True, facecolor="white",
                         framealpha=1, fontsize=8)
    for panel in (axis, residual_axis):
        panel.tick_params(direction="in", top=True, right=True)
        panel.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
    axis.tick_params(labelbottom=False)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    analysis = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-results", type=Path, default=analysis / "output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv")
    parser.add_argument("--composition", type=Path, default=analysis / "data/np04_beam_particle_content.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--relative-momentum-error", type=float, default=0.05,
                        help="Incertezza relativa efficace sul momento; default 0.05")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.fit_results.parent
    points = read_points(args)
    fit_all = fit_line(points, "including_1_GeV_c")
    fit_four = fit_line(points[1:], "excluding_1_GeV_c")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "apa1_linearity_points.csv", points)
    write_csv(args.output_dir / "apa1_linearity_fits.csv", [fit_all, fit_four])
    draw(points, fit_all, fit_four, args.output_dir / "apa1_linearity.png")
    with (args.output_dir / "apa1_linearity_report.txt").open("w") as handle:
        handle.write("APA 1: risposta luminosa vs energia cinetica media attesa del fascio\n")
        handle.write(f"Composizione: {args.composition}\nFit: {args.fit_results}\n")
        handle.write(f"Incertezza relativa efficace sul momento: "
                     f"{args.relative_momentum_error:g}\n")
        handle.write("I tassi sono previsioni simulate H4-VLE, Tabella IV, non "
                     "conteggi di eventi: nessuna incertezza Poisson viene "
                     "attribuita ai pesi. Le barre x sono la dispersione "
                     "efficace sqrt(sigma_mixture^2 + sigma_Keff,p^2): la "
                     "prima componente e' la RMS tra specie, la seconda "
                     "propaga il 5% sul momento comune. L'origine del 5% nel "
                     "paper e' un'accettanza +/-5%, qui usata come incertezza "
                     "efficace conservativa 1-sigma.\n")
        handle.write("Il punto a 1 GeV/c è il massimo Langauss, gli altri sono medie "
                     "gaussiane: il fit su cinque punti è un controllo di sensibilità.\n")
        handle.write("La composizione dell'intero fascio può differire da quella degli "
                     "eventi nella componente gaussiana. La media PE è per canale "
                     "contribuente, non la somma di APA 1.\n")
        for fit in (fit_all, fit_four):
            handle.write(f"{fit['model']}: m={fit['slope_PE_per_GeV']:.5g} +/- "
                         f"{fit['slope_error_PE_per_GeV']:.3g} PE/GeV; "
                         f"q={fit['intercept_PE']:.5g} +/- "
                         f"{fit['intercept_error_PE']:.3g} PE; "
                         f"chi2/ndf={fit['chi2']:.4g}/{fit['ndf']}; "
                         f"p={fit['p_value']:.3g}; R2={fit['r_squared']:.4g}\n")
    print(args.output_dir / "apa1_linearity.png")


if __name__ == "__main__":
    main()
