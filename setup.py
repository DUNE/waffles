from setuptools import find_packages, setup

setup(
    name='waffles',
    version='0.1.0',
    description='PDS Waveform Analysis Framework For Light Emission Studies',
    author='Many',
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    package_data={
        # Include the .c file
        'waffles.utils.denoising.tv1d_cpp': ['tv1ddenoise.c']
    },
    include_package_data=True,
    # ,
    # install_requires=['numpy','dash','dash_bootstrap_components','graphviz','h5py','inquirer','jupyter','matplotlib','rich','scipy','pandas','plotly','uproot','kaleido','numba']
)
