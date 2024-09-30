import os, socket

class SetServer:
    """
    Set up the Dash app to run in Jupyter or as a standalone server.
    """

    def __init__(self):
        """
        Initialize the SetServer class.
        """
        self.jupyter_mode = self.check_environment()

    def check_environment(self):
        """
        Check if the script is running in a Jupyter notebook or as a standalone script.
        """
        if 'ipykernel' in os.sys.modules:
            from IPython import get_ipython
            if get_ipython() is not None: return "inline"
        return "external"

    def get_config(self):
        """
        Get the appropriate configuration for a Dash application based on the environment.
        """
        if self.jupyter_mode == "inline":
            # Return the configuration for running in a Jupyter notebook
            hostname = socket.gethostname()
            print(f"Working on this {hostname}")
            return {"service_prefix": "/display/",
                    "server_url": f"http://{hostname}",
                    "port": self.find_free_port(),
                    "jupyter_mode": "inline"}
        else:
            # Return the configuration for running as a standalone script
            return {"service_prefix": "/display/",
                    "server_url": "http://localhost",
                    "port": self.find_free_port(),
                    "jupyter_mode": None}

    def find_free_port(self, start_port=8050):
        """
        Find a free port to run the Dash app on.
        """
        port = start_port
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("localhost", port))
                sock.close()
                return port
            except OSError:
                port += 1
                if port - start_port > 1000:  # Just to avoid an infinite loop
                    raise OSError("No free ports available within 1000 ports of start_port")