import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def add_period(ax, start, end, y, color):
    ax.barh(y, end - start, left=start, height=0.3, color=color)

def parse_datetime(s):
    """
    Tries to parse a datetime string in multiple formats:
    Accepts both '.' or '-' as separators and year as YY or YYYY.
    Supported:
    - dd.mm.yy HH:MM
    - dd.mm.yyyy HH:MM
    - dd-mm-yy HH:MM
    - dd-mm-yyyy HH:MM
    """
    s = s.strip()
    formats = [
        "%d.%m.%y %H:%M", "%d.%m.%Y %H:%M",
        "%d-%m-%y %H:%M", "%d-%m-%Y %H:%M"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {s}")

def add_run_period(ax, run_name, start_str, end_str, color="tab:red", alpha=0.2):
    """
    Adds a vertical shaded band and dashed boundary lines for a given run.
    """
    start = parse_datetime(start_str)
    end = parse_datetime(end_str)

    # Shaded region
    ax.axvspan(start, end, color=color, alpha=alpha, label=f"Run {run_name}")
    # Dashed vertical lines
    ax.axvline(start, color=color, linestyle="--", linewidth=1)
    ax.axvline(end, color=color, linestyle="--", linewidth=1)

# --- Setup grafico base ---
fig, ax = plt.subplots(figsize=(13, 3.8))

# --- XCET-H (High Pressure, 14 bar) ---
add_period(ax, datetime(2024,6,20,1,0), datetime(2024,6,20,14,40), "XCET-H", "tab:blue")
add_period(ax, datetime(2024,7,5,12,30), datetime(2024,7,10,8,0), "XCET-H", "tab:blue")
add_period(ax, datetime(2024,7,15,11,37), datetime(2024,8,14,10,37), "XCET-H", "tab:blue")
add_period(ax, datetime(2024,8,14,18,47), datetime(2024,8,26,12,0), "XCET-H", "tab:blue")
add_period(ax, datetime(2024,8,26,13,0), datetime(2024,8,27,12,0), "XCET-H", "tab:blue")
add_period(ax, datetime(2024,8,27,13,0), datetime(2024,9,18,8,22), "XCET-H", "tab:blue")

# --- XCET-L (Low Pressure) ---
add_period(ax, datetime(2024,6,1), datetime(2024,6,20,0,0), "XCET-L", "tab:green")
add_period(ax, datetime(2024,7,5,11,32), datetime(2024,7,11,10,7), "XCET-L", "tab:green") # 5 bar
add_period(ax, datetime(2024,6,25,10,0), datetime(2024,7,5,11,32), "XCET-L", "tab:orange") # 3.5 bar
add_period(ax, datetime(2024,8,14,18,30), datetime(2024,8,26,13,0), "XCET-L", "tab:purple")    # 3 bar
add_period(ax, datetime(2024,8,27,10,0), datetime(2024,9,18,8,22), "XCET-L", "tab:purple")   # 3 bar


# ONLY CHERENKOV STATUS
name = "cherenkov_status"

# ENERGY SCAN JUNE
# name = "energy_scan"
# add_run_period(ax, run_name="27343 (1 GeV)", start_str="21.06.24 15:40", end_str="21.06.24 15:40", color="tab:red")
# add_run_period(ax, run_name="27355 (2 GeV)", start_str="21.06.24 22:47", end_str="21.06.24 23:23", color="tab:cyan")
# add_run_period(ax, run_name="27361(3 GeV)", start_str="22.06.24 02:43", end_str="22.06.24 03:14", color="tab:green")
# add_run_period(ax, run_name="27367 (5 GeV)", start_str="22.06.24 05:06", end_str="22.06.24 05:36", color="tab:pink")
# add_run_period(ax, run_name="27374 (7 GeV)", start_str="22.06.24 09:20", end_str="22.06.24 09:50", color="tab:orange")

# OTHER JUNE RUNS
# name = "other_june_runs"
# add_run_period(ax, run_name="27404 (5 GeV)", start_str="24.06.24 12:04", end_str="24.06.24 15:35", color="tab:orange")
# add_run_period(ax, run_name="27408 (7 GeV)", start_str="24.06.24 17:06", end_str="25.06.24 01:29", color="tab:pink")
# add_run_period(ax, run_name="27410 (7 GeV)", start_str="25.06.24 03:06", end_str="25.06.24 05:38", color="tab:green")
# add_run_period(ax, run_name="27412 (7 GeV)", start_str="25.06.24 05:53", end_str="25.06.24 09:28", color="tab:blue")

# July runs
# name = "7GeVruns"
# add_run_period(ax, run_name="27971 (7 GeV)", start_str="10-07-2024 20:16", end_str="11-07-2024 00:26", color="tab:orange")
# add_run_period(ax, run_name="27973 (7 GeV)", start_str="11-07-2024 00:45", end_str="11-07-2024 09:15", color="tab:red")

# Other 1 GeV 
# name = "1GeV_runs"
# # July
# add_run_period(ax, run_name="From 28005 to 28008 (1 GeV)", start_str="12-07-2024 20:25", end_str="14-07-2024 03:07", color="tab:orange")
# add_run_period(ax, run_name="28010 (1 GeV)", start_str="14-07-2024 04:06", end_str="15-07-2024 02:17", color="tab:red")
# add_run_period(ax, run_name="28012 (1 GeV)", start_str="15-07-2024 01:20", end_str="15-07-2024 10:47", color="tab:blue")
# add_run_period(ax, run_name="28051-52 (1 GeV)", start_str="18-07-2024 09:30", end_str="18-07-2024 11:49", color="tab:cyan")
# add_run_period(ax, run_name="28054 (1 GeV)", start_str="18-07-2024 12:19", end_str="18-07-2024 14:42", color="tab:pink")
# add_run_period(ax, run_name="From 28059 to 28085 (1 GeV)", start_str="18-07-2024 16:06", end_str="21-07-2024 08:02", color="tab:purple")
# # August
# add_run_period(ax, run_name="From 28538 to 28548 (1 GeV)", start_str="02-08-2024 15:32", end_str="05-08-2024 11:34", color="tab:green")
# add_run_period(ax, run_name="From 28586 to 28595 (1 GeV)", start_str="08-08-2024 16:05", end_str="09-08-2024 14:38", color="tab:grey")
# add_run_period(ax, run_name="From 28674 to 28677 (1 GeV)", start_str="14-08-2024 03:59", end_str="14-08-2024 08:18", color="tab:brown")
# # ne mancano alcune a meta agosto
# # September
# add_run_period(ax, run_name="From 29068 to 29072 (1 GeV)", start_str="03-09-2024 13:31", end_str="04-09-2024 09:04", color="tab:olive")
# add_run_period(ax, run_name="29091-92 (1 GeV)", start_str="05-09-2024 16:32", end_str="06-09-2024 09:58", color="lime")
# add_run_period(ax, run_name="29097 (1 GeV)", start_str="06-09-2024 12:09", end_str="06-09-2024 18:18", color="coral")

# Other 5GeV
# name = "5GeV_runs"
# add_run_period(ax, run_name="From 28693 to 28707 (Kaons, 5 GeV)", start_str="14-08-2024 10:06", end_str="16-08-2024 07:57", color="tab:orange")
# add_run_period(ax, run_name="From 28711 to 28736 (5 GeV)", start_str="16-08-2024 10:34", end_str="16-08-2024 16:11", color="tab:blue")
# add_run_period(ax, run_name="From 28751 to 28755 (5 GeV)", start_str="16-08-2024 23:01", end_str="17-08-2024 12:35", color="tab:pink")
# add_run_period(ax, run_name="From 28773 to 28778 (Kaons, 5 GeV)", start_str="19-08-2024 15:29", end_str="20-08-2024 16:03", color="tab:red")

# --- Formatting ---
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax.set_xlim(datetime(2024,6,1), datetime(2024,9,30))
ax.set_xlabel("Date (CERN time)")
ax.set_title("Cherenkov Detectors status - NP04 Summer 2024")

# Griglia fine
ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

plt.xticks(fontsize=6)
plt.yticks(fontsize=9)

# Legenda: detector + run
handles = [
    plt.Rectangle((0,0),1,1,color='tab:blue', label='XCET-H on (14 bar)'),
    plt.Rectangle((0,0),1,1,color='tab:green', label='XCET-L on (5 bar)'),
    plt.Rectangle((0,0),1,1,color='tab:orange', label='XCET-L on (3.5 bar)'),
    plt.Rectangle((0,0),1,1,color='tab:purple', label='XCET-L on (3 bar)'),
]
# aggiunge anche i run dinamicamente
for h in ax.patches:
    if hasattr(h, 'get_label') and 'Run' in h.get_label():
        handles.append(h)
ax.legend(handles=handles, loc='lower left', fontsize=6)

plt.tight_layout()


plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/scripts/cherenkov_state/{name}.png', dpi=300)

