import os
import subprocess
import getpass
import click
from collections import defaultdict
from urllib.parse import urlparse



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Directory where the per‐run text files with PFNs will be written
SAVING_PATH = ""

# Possible dataset prefixes (try them in order until one yields files)
REPLICA_PREFIXES = [
    "vd-protodune",  # Vertical drift
    "hd-protodune",  # Horizontal drift
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _choose_realm(pfn_lines: list[str]) -> tuple[str, list[str]]:
    """
    Decide which group of PFNs to keep, in the following order:

        1. DUNE_CERN_EOS   (host contains "eospublic.cern.ch")
        2. FNAL_DCACHE     (host contains "fndca1.fnal.gov")
        3. Any other realm – except eosctapublic.cern.ch
           (if *only* eosctapublic is present, raise RuntimeError)

    Returns
    -------
    chosen_realm : str
        The host[:port] that won the selection.
    realm_lines  : list[str]
        PFNs that belong to that realm.
    """
    # Group PFNs by realm
    realm_to_lines: dict[str, list[str]] = defaultdict(list)
    for ln in pfn_lines:
        if "://" not in ln:
            continue
        host_port = urlparse(ln).netloc
        realm_to_lines[host_port].append(ln)

    # Preferred realms (sub-string match makes the port irrelevant)
    priorities = ["eospublic.cern.ch", "fndca1.fnal.gov"]

    # 1-2.  Try the preferred ones in order
    for dom in priorities:
        for realm, lines in realm_to_lines.items():
            if dom in realm:
                return realm, lines

    # 3.  If the only realm is eosctapublic → abort
    non_cta = [(r, l) for r, l in realm_to_lines.items()
               if "eosctapublic.cern.ch" not in r]
    if not non_cta:
        raise RuntimeError(
            "Only CASTOR replicas (eosctapublic.cern.ch) were found – aborting."
        )

    # Otherwise just pick the first acceptable realm
    return non_cta[0]


def is_rucio_active() -> bool:
    """Return ``True`` if the user's Rucio environment is already usable."""
    try:
        subprocess.run(
            "rucio whoami",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("\033[92mRucio is already set up. Skipping environment setup.\033[0m")
        return True
    except subprocess.CalledProcessError:
        print("\033[93mRucio is not set up. Initialising the environment…\033[0m")
        return False


def setup_rucio_environment() -> None:
    """Perform Kerberos + Rucio initialisation unless it is already active."""
    if is_rucio_active():
        return  # nothing to do

    print("\033[94mSetting up the Rucio environment…\033[0m")

    username = input("Enter your @FNAL.GOV username: ")
    password = getpass.getpass("Please enter your password: ")

    setup_cmd = f"""
    source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh && \
    spack load r-m-dd-config experiment=dune && \
    spack load kx509 && \
    echo "{password}" | kinit {username}@FNAL.GOV && \
    kx509 && \
    export RUCIO_ACCOUNT={username} && \
    voms-proxy-init -rfc -noregen -voms=dune:/dune/Role=Analysis -valid 120:00 && \
    export UPS_OVERRIDE="-H Linux64bit+3.10-2.17" && \
    setup ifdhc && \
    rucio whoami
    """

    result = subprocess.run(f"bash -c \"{setup_cmd}\"", shell=True, executable="/bin/bash")
    if result.returncode == 0:
        print("\033[92mRucio authentication successful!\033[0m")
    else:
        print("\033[31mRucio setup failed. Please check your environment settings.\033[0m")
        raise SystemExit(1)


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def fetch_rucio_replicas(run_number: str, max_files: int) -> None:
    """Try all dataset prefixes until files are found, then write them to disk.

    Parameters
    ----------
    run_number : str
        The run number (e.g. "28676"). **No zero‑padding needed here**.
    max_files : int
        Maximum number of PFNs to write to the output file.
    """

    run_str = str(run_number).zfill(6)  # for nice, fixed‑width filenames only
    output_file = f"{SAVING_PATH}{run_str}.txt"

    all_lines = []
    used_prefix = None

    for prefix in REPLICA_PREFIXES:
        dataset = f"{prefix}:{prefix}_{run_number}"
        rucio_cmd = f"rucio replica list file --pfns {dataset}"
        print(f"\033[94mTrying prefix '{prefix}' for run {run_number}…\033[0m")

        try:
            result = subprocess.run(
                rucio_cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=True,
                executable="/bin/bash",
            )
            lines = [ln for ln in result.stdout.strip().split("\n") if ln]
            if lines:
                all_lines = lines
                used_prefix = prefix
                break  # success — no need to try further prefixes
        except subprocess.CalledProcessError:
            # This prefix did not work — try the next one
            continue

    if not all_lines:
        print(
            f"\033[31mNo file replicas found for run {run_number} using any of the prefixes {REPLICA_PREFIXES}.\033[0m"
        )
        return

    # ------------------------------------------------------------------
    # Keep only PFNs from the *first* realm (protocol+domain)
    # ------------------------------------------------------------------
    try:
        chosen_realm, same_realm_lines = _choose_realm(all_lines)
    except RuntimeError as err:
        print(f"\033[31m{err}\033[0m")
        return

    selected_lines = same_realm_lines[:max_files]

    with open(output_file, "w") as fh:
        fh.write("\n".join(selected_lines) + "\n")

    print(
        f"\033[92mSaved {len(selected_lines)} PFNs "
        f"(realm {chosen_realm}, prefix {used_prefix}) "
        f"to {output_file}\033[0m"
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

@click.command()
@click.option("--runs", required=True, help="Comma‐separated run numbers, e.g. 28676,28677")
@click.option("--max-files", required=True, type=int, help="Maximum number of files per run")
def main(runs: str, max_files: int) -> None:
    """Entry point when the script is executed via the command line."""
    setup_rucio_environment()

    for run in [r.strip() for r in runs.split(",") if r.strip()]:
        fetch_rucio_replicas(run, max_files)

    # Clean‑up Kerberos tickets
    subprocess.run("kdestroy", shell=True)
    print("\033[92mSession complete. Kerberos credentials destroyed.\033[0m")


if __name__ == "__main__":
    main()