import os
import shutil
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional


class RucioHandler:
    """
    Manage fetching replica lists and downloading data from Rucio with reliable logging.

    Attributes
    ----------
    data_folder : Path
        Destination folder for downloads.
    txt_folder : Path
        Folder where the replica list txt file is written/read.
    max_files : int
        Max files to request from the fetch script.
    waffles_scripts_dir : Path
        Directory containing setup_rucio_a9.sh and fetch_rucio_replicas.py.
    log_dir : Path
        Where logs are written (defaults to data_folder).
    rucio_cmd : str
        Executable for rucio (defaults to "rucio").

    Methods
    -------
    setup_rucio(script_name="setup_rucio_a9.sh") -> dict
        Sources the env and stores it for subsequent commands.
    download_data_from_rucio(run_number: int) -> List[str]
        Fetches the list and downloads each DID with Rucio.
    """

    def __init__(
        self,
        data_folder: str,
        txt_folder: str,
        max_files: int,
        *,
        waffles_scripts_dir: str = "/home/ecristal/software/DUNE/waffles/scripts",
        log_dir: Optional[str] = None,
        rucio_cmd: str = "rucio",
        replicas_filename_template: str = "0{run}.txt",
    ):
        self.data_folder = Path(data_folder)
        self.txt_folder = Path(txt_folder)
        self.max_files = int(max_files)

        self.waffles_scripts_dir = Path(waffles_scripts_dir)
        self.log_dir = Path(log_dir) if log_dir else self.data_folder
        self.rucio_cmd = rucio_cmd
        self.replicas_filename_template = replicas_filename_template

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.txt_folder.mkdir(parents=True, exist_ok=True)

        self._env: Optional[dict] = None  # populated by setup_rucio()

    # ---------- public API ----------

    def setup_rucio_1(self, script_name: str = "setup_rucio_a9.sh") -> None:
        """
        Run the Rucio setup script interactively (no env capture).
        This shows all prompts (OIDC/2FA, username/password, etc.) and lets the
        user type into the terminal. Leaves the parent process env unchanged.
        """
        script = self.waffles_scripts_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"Rucio setup script not found: {script}")

        #Use an interactive login shell so 'source' works and prompts are visible.
        subprocess.run(
            f"bash -i -c 'source {shlex.quote(str(script))}'",
            shell=True,
            check=True,
        )


    def setup_rucio_2(self, script_name: str = "setup_rucio_a9.sh") -> dict:
        """
        Run the Rucio setup script interactively (user sees prompts, incl. 2FA),
        then capture the resulting environment to reuse in later subprocess calls.

        After capturing, we:
          - Prefer a *storage* SciToken (e.g. /run/user/$UID/davs.token) over any
            API token (bt_u...), and propagate it via BEARER_TOKEN_FILE /
            RUCIO_AUTH_TOKEN_FILE (and BEARER_TOKEN for stacks that honor inline).
          - Unset X509_USER_PROXY so it doesn't shadow tokens at DAVS endpoints.
          - Set CA bundle defaults to quiet TLS warnings and ensure certs are found.
        """
        script = self.waffles_scripts_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"Rucio setup script not found: {script}")

        # One interactive shell: source the script (prompts shown), then dump env.
        cmd = (
            "bash -i -c '"
            f"source {shlex.quote(str(script))}; env -0'"
        )        
        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=False,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit stderr so prompts and messages are visible
        )
        out, _ = proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        # Parse `env -0` output into a dict
        new_env: dict[str, str] = {}
        for entry in out.split(b"\x00"):
            if not entry:
                continue
            k, _, v = entry.partition(b"=")
            new_env[k.decode()] = v.decode()

        # Start from current env, then overlay what the setup script exported
        env = os.environ.copy()
        env.update(new_env)

        # ---- Prefer a *storage* token over an API token (bt_u...) ----------------
        uid = os.getuid()

        # Priority:
        #   1) explicit override via WAFFLES_STORAGE_TOKEN
        #   2) /run/user/$UID/davs.token (common path we minted earlier)
        #   3) whatever BEARER_TOKEN_FILE points to already
        candidates = [
            env.get("WAFFLES_STORAGE_TOKEN") or os.environ.get("WAFFLES_STORAGE_TOKEN"),
            f"/run/user/{uid}/davs.token",
            env.get("BEARER_TOKEN_FILE"),
        ]
        chosen = next((p for p in candidates if p and os.path.isfile(p)), None)
        if chosen:
            env["BEARER_TOKEN_FILE"] = chosen
            env["RUCIO_AUTH_TOKEN_FILE"] = chosen
            env["RUCIO_STORAGE_TOKEN_FILE"] = chosen
            # Some stacks also honor an inline token
            try:
                with open(chosen, "r") as fh:
                    env["BEARER_TOKEN"] = fh.read().strip()
                    env["RUCIO_STORAGE_TOKEN"] = env["BEARER_TOKEN"]
            except Exception:
                pass
            # Ensure a proxy (if any) doesn't override bearer tokens at DAVS
            env.pop("X509_USER_PROXY", None)

        # ---- Reasonable defaults for trust anchors / TLS -------------------------
        env.setdefault("REQUESTS_CA_BUNDLE", "/etc/pki/tls/certs/ca-bundle.crt")
        env.setdefault("X509_CERT_DIR", "/etc/grid-security/certificates")

        self._env = env
        return env

    def _davix_head_with_token(self, url: str, env: dict) -> int:
        """
        Do a HEAD to `url` with the bearer token from env.
        Return the process returncode (0 = success).
        """
        # Prefer inline; else read from file.
        tok = env.get("BEARER_TOKEN")
        if not tok:
            f = env.get("BEARER_TOKEN_FILE") or env.get("RUCIO_STORAGE_TOKEN_FILE")
            if f and os.path.isfile(f):
                with open(f, "r") as fh:
                    tok = fh.read().strip()
        if not tok:
            print("[preflight] No bearer token visible in env")
            return 1

        cmd = [
            "davix-http", "--head",
            "-H", f"Authorization: Bearer {tok}",
            url,
        ]
        # Be verbose if you like:
        # cmd += ["-v", "3"]

        try:
            return subprocess.call(cmd, env=env)
        except FileNotFoundError:
            print("[preflight] davix-http not found; skipping HEAD test")
            return 0


    def download_data_from_rucio(self, run_number: int) -> List[str]:
        if self._env is None:
            self.setup_rucio_2()

        fetch_log = self.log_dir / f"run_{run_number}_fetch.log"
        fetch_script = self.waffles_scripts_dir / "fetch_rucio_replicas.py"
        if not fetch_script.exists():
            raise FileNotFoundError(f"Fetch script not found: {fetch_script}")

        # 1. produce the replicas list
        #self._stream_run(
        #    ["python", "-u", str(fetch_script), "--runs", str(run_number),
        #     "--max-files", str(self.max_files)],
        #    cwd=self.txt_folder,
        #    log_file=fetch_log,
        #    env=self._env,
        #)

        # 2. parse the list and pick PFNs
        replica_file = self.txt_folder / self.replicas_filename_template.format(run=run_number)
        if not replica_file.exists():
            alt = self.txt_folder / f"{run_number}.txt"
            if alt.exists():
                replica_file = alt
            else:
                raise FileNotFoundError(f"Replica list not found: {replica_file} or {alt}")

        pfn_list = []
        with open(replica_file) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("root://"):           # prefer xrootd
                    pfn_list.append(line)

        if not pfn_list:
            raise RuntimeError(f"No xrootd PFNs found in {replica_file}")
        
        dest_dir = self.data_folder / "hd-protodune"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 3. copy each file with xrdcp
        downloaded = []
        for pfn in pfn_list:
            dest = dest_dir / Path(pfn).name
            print(f"Copying {pfn} → {dest}")
            self._stream_run(
                ["xrdcp", pfn, str(dest)],
                env=self._env,
            )
            downloaded.append(str(dest))

        return downloaded


    # ---------- internals ----------

    def _stream_run(
        self,
        cmd,
        *,
        cwd: Optional[Path] = None,
        log_file: Optional[Path] = None,
        env: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Run a command, stream stdout line-by-line to console and (optionally) a log file,
        and raise subprocess.CalledProcessError on non-zero exit.
        """
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)

        log_fh = open(log_file, "a", buffering=1) if log_file else None
        try:
            with subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    print(line)
                    if log_fh:
                        log_fh.write(line + "\n")

                ret = proc.wait(timeout=timeout)
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
        finally:
            if log_fh:
                log_fh.close()

    def _parse_replicas_file(self, path: Path) -> List[str]:
        """
        Read lines, keep basename, and prepend the DID scope.
        """
        dids: List[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                name = Path(line).name
                dids.append(f"hd-protodune:{name}")
        return dids

    def clean_downloads(self) -> None:
        """
        Remove the local folder where Rucio downloads its files:
        <data_folder>/hd-protodune

        If the folder does not exist, do nothing.
        """
        target = self.data_folder / "hd-protodune"
        if target.exists():
            print(f"Removing downloaded files under: {target}")
            shutil.rmtree(target)
        else:
            print(f"No hd-protodune folder found in {self.data_folder}, nothing to clean.")
