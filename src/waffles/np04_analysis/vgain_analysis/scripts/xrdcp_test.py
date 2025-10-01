import os, subprocess

env = os.environ.copy()
env["BEARER_TOKEN_FILE"] = "/run/user/1000/davs.token"
env["X509_CERT_DIR"]     = "/etc/grid-security/certificates"

cmd = [
          "xrdcp",
            "root://ccxrootdegee.in2p3.fr:1094/pnfs/in2p3.fr/data/dune/disk/hd-protodune/43/0b/np04hd_raw_run029876_0000_dataflow0_datawriter_0_20241011T071028.hdf5",
              "./test_from_python.hdf5"
              ]

subprocess.run(cmd, check=True)

