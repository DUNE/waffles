import os, subprocess

env = os.environ.copy()
#env["BEARER_TOKEN_FILE"] = "/run/user/1000/davs.token"
#env["X509_CERT_DIR"]     = "/etc/grid-security/certificates"

cmd = [
          "xrdcp",
            "root://eospublic.cern.ch:1094//eos/experiment/neutplatform/protodune/dune/np_buffer/vd-protodune/7a/e0/np02vd_raw_run043191_0000_df-s03-d1_dw_0_20260309T112125.hdf5",
              "./test_from_python.hdf5"
              ]

subprocess.run(cmd, check=True)

