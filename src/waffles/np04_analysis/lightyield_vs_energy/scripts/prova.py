import pandas as pd

def spe_charge(df: pd.DataFrame, endpoint: int, channel: int, hpk_ov: float = 3.0, fbk_ov = 4.5):
    if fbk_or_hpk(endpoint, channel) == 'FBK':
        ov_column = 'FBK_OV_V'
        ov = fbk_ov
    else:
        ov_column = 'HPK_OV_V'
        ov = hpk_ov
    result = df[(df['endpoint'] == endpoint) & (df['channel'] == channel) & (df[ov_column] == ov)]['gain']
    return result.iloc[0] if not result.empty else None

def fbk_or_hpk(endpoint: int, channel: int):
    channel_vendor_map = {
    104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
    105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
    107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
          10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
    109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
    113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

    return channel_vendor_map[endpoint][channel]

print('\nReading led calibration info...')
LED_calibration_info = pd.read_pickle(f"/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/batch_1_LED_calibration_data.pkl")
print(spe_charge(LED_calibration_info,112,0))        

