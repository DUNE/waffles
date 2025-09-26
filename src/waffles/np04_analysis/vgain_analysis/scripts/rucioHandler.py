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
        waffles_scripts_dir: str = "/afs/cern.ch/work/e/ecristal/daphne/waffles/scripts",
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
        Run the Rucio setup script interactively.
        This will show all prompts (user/pass, etc.) and let the user type into the terminal.
        """
        script = self.waffles_scripts_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"Rucio setup script not found: {script}")

        # Use subprocess.run with shell=True so `source` works inside bash.
        # stdin/stdout/stderr are connected to the parent process (interactive).
        subprocess.run(
            f"bash -i -c 'source {shlex.quote(str(script))}'",
            shell=True,
            check=True
        )


    def setup_rucio_2(self, script_name: str = "setup_rucio_a9.sh") -> dict:
        """
        Run the Rucio setup script interactively (user can type credentials),
        then capture the resulting environment for later subprocess calls.
        """
        script = self.waffles_scripts_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"Rucio setup script not found: {script}")

        # Run one interactive shell:
        #  1. source the script (interactive prompts shown)
        #  2. after setup completes, dump environment with env -0
        cmd = f"bash -i -c 'source {shlex.quote(str(script))}; env -0'"

        proc = subprocess.Popen(
            cmd,
            shell=True,
            text=False,
            stdout=subprocess.PIPE,
            stderr=None  # let stderr go to your terminal for prompts
        )

        out, _ = proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        # Parse env -0 output
        new_env = {}
        for entry in out.split(b"\x00"):
            if not entry:
                continue
            k, _, v = entry.partition(b"=")
            new_env[k.decode()] = v.decode()

        env = os.environ.copy()
        env.update(new_env)
        self._env = env
        return env


    def download_data_from_rucio(self, run_number: int) -> List[str]:
        """
        1) Runs fetch_rucio_replicas.py to produce the replica list.
        2) Parses the list, building DIDs as 'hd-protodune:<basename>'.
        3) Downloads each DID with `rucio download`.

        Returns the list of DIDs attempted. Raises on any failing step.
        """
        if self._env is None:
            #Auto-setup if the caller forgot; comment this out if you prefer explicit setup.
            self.setup_rucio()

        fetch_log = self.log_dir / f"run_{run_number}_fetch.log"
        dl_log = self.log_dir / f"run_{run_number}_download.log"

        # 1) fetch list (use unbuffered python so prints are live)
        fetch_script = self.waffles_scripts_dir / "fetch_rucio_replicas.py"
        if not fetch_script.exists():
            raise FileNotFoundError(f"Fetch script not found: {fetch_script}")

        self._stream_run(
            ["python", "-u", str(fetch_script), "--runs", str(run_number), "--max-files", str(self.max_files)],
            cwd=self.txt_folder,
            log_file=fetch_log,
            env=self._env,
        )

        # 2) parse produced file (support template and a fallback without leading 0)
        produced = self.txt_folder / self.replicas_filename_template.format(run=run_number)
        if not produced.exists():
            alt = self.txt_folder / f"{run_number}.txt"
            if alt.exists():
                produced = alt
            else:
                raise FileNotFoundError(f"Replica list not found: {produced} (or fallback {alt})")

        rucio_dids = self._parse_replicas_file(produced)
        if not rucio_dids:
            raise RuntimeError(f"No entries in {produced} for run {run_number}.")

        # 3) download each DID
        local_dids: List[str] = []
        for did in rucio_dids:
            did_name = did.split(':')
            local_dids.append(f'{self.data_folder}/hd-protodune/{did_name[-1]}')
            self._stream_run(
                [self.rucio_cmd, "download", did],
                cwd=self.data_folder,
                log_file=dl_log,
                env=self._env,
            )
        
        return local_dids

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