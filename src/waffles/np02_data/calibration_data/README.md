### Files available and description:

Files are created following the tag created in the DUNE/PDS repository. For instance, the file [np02-config-v3.0.0.csv](https://github.com/DUNE/PDS/releases/tag/np02-config-v3.0.0) has the calibration data for the PDS with the configuration as set on the file `details.json`. 

The available files are:

- np02-config-v4.0.0.csv : Optimal calibration created after trim scan for the membranes. It was used during TPC+PDS run on beam period and aftwards

- np02-config-v3.0.0.csv : Calibration created before trim scan. It was used during standalone PDS run on beam period.

### How to load calibration data

```python
from waffles.np02_utils.load_utils import ch_read_calib
calib_data = ch_read_calib(filename='np02-config-v3.0.0.csv')
```

### Structure of the calibration data

The variable `calib_data` is a nested dictionary containing calibration information for each `endpoint` and `channel`. The structure of the dictionary is as follows:

```python
{
    'endpoint': {
        'channel_1': {
            'Gain': value,
            'SpeAmpl': value,
        },
        'channel_2': {
            'Gain': value,
            'SpeAmpl': value,
            ...
        },
        ...
    },
    'endpoint_2': {
        ...
    }
}
```




