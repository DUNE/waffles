#!/usr/bin/env python3
import csv
import requests
import sys
import urllib3

# Disable HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_ifbeam(var, event, t0, t1):
    """Fetch IFBeam variable as CSV"""
    url = (
        f"https://dbdata3vm.fnal.gov:9443/ifbeam/data/data"
        f"?e={event}&v={var}&t0={t0}&t1={t1}&f=csv"
    )
    r = requests.get(url, verify=False)
    r.raise_for_status()
    return r.text.splitlines()

def parse_clock(lines):
    """Extract the clock values from CSV lines"""
    clocks = []
    for row in lines[1:]:  # skip header
        parts = row.strip().split(",")
        if len(parts) < 4:
            continue
        clocks.append(parts[3])
    return clocks

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 get_ifbeam_clock.py <event> <t0> <t1> <output_csv>")
        sys.exit(1)

    event, t0_str, t1_str, output_csv = sys.argv[1:]

    # IFBeam variable names
    seconds_var = "dip/acc/NORTH/NP02/BI/XTOF/XBTF022638A:seconds[]"
    coarse_var  = "dip/acc/NORTH/NP02/BI/XTOF/XBTF022638A:coarse[]"
    frac_var    = "dip/acc/NORTH/NP02/BI/XTOF/XBTF022638A:frac[]"

    # Fetch clock values for each variable
    seconds_clock = parse_clock(fetch_ifbeam(seconds_var, event, t0_str, t1_str))
    coarse_clock  = parse_clock(fetch_ifbeam(coarse_var, event, t0_str, t1_str))
    frac_clock    = parse_clock(fetch_ifbeam(frac_var, event, t0_str, t1_str))

    # Write clocks to CSV
    with open(output_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["seconds_clock", "coarse_clock", "frac_clock"])
        n = max(len(seconds_clock), len(coarse_clock), len(frac_clock))
        for i in range(n):
            writer.writerow([
                seconds_clock[i] if i < len(seconds_clock) else "",
                coarse_clock[i]  if i < len(coarse_clock)  else "",
                frac_clock[i]    if i < len(frac_clock)    else ""
            ])

    print(f"Clock values saved to {output_csv} ({n} rows).")

if __name__ == "__main__":
    main()