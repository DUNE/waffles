import os,sys
import traceback
#path = 'C:\\Users\\e_cri\\OneDrive\\Documents\\INFN\\Software\\waffles\\src\\'
#os.chdir(path)
#print(os.getcwd())
#sys.path.append(path)  # Add this line
#from waffles.np04_analysis.led_calibration.configs.calibration_batches.LED_configuration_to_channel import config_to_channels as ch_conf
from waffles.np04_analysis.vgain_analysis.configs.LED_map import config_to_channels as ch_conf
import pandas as pd

def getOvValues(PDE):
    if(PDE == '40%'):
        return [2.0, 3.5]
    elif(PDE == '45%'):
        return [2.5, 4.5]
    elif(PDE == '50%'):
        return [3.0, 7.0]
    else:
        raise RuntimeError('config file is missing host address')

def getPDEValues(PDE):
    if(PDE == '40%'):
        return 0.4
    elif(PDE == '45%'):
        return 0.45
    elif(PDE == '50%'):
        return 0.5
    else:
        raise RuntimeError('config file is missing host address')

def processCableMapSwap(map_dictionary):
    vgain_dict = {}
    channel_cable_swap_list = []
    endpoint_104_swap_dict = {
        6:6,
        4:4,
        3:3,
        1:1,
        0:0,
        2:0,
        5:5,
        7:7,
        16:36,
        14:34,
        13:33,
        11:31,
        10:30,
        12:32,
        15:35,
        17:37
    }
    endpoint_105_swap_dict = {
        6:16,
        4:14,
        3:13,
        1:11,
        0:10,
        2:12,
        5:15,
        7:17,
        17:27,
        15:25,
        12:22,
        10:20,
        21:40,
        23:42,
        24:45,
        26:47
    }
    endpoint_107_swap_dict = {
        7:26,
        5:24,
        2:23,
        0:21,
        10:41,
        12:43,
        15:44,
        17:46
    }
    for key in map_dictionary.keys():
        if(key == 104):
            channel_list = map_dictionary[key]
            for channel in channel_list:
                channel_cable_swap_list.append(endpoint_104_swap_dict[channel])
        elif(key == 105):
            channel_list = map_dictionary[key]
            for channel in channel_list:
                channel_cable_swap_list.append(endpoint_105_swap_dict[channel])
        elif(key == 107):
            channel_list = map_dictionary[key]
            for channel in channel_list:
                channel_cable_swap_list.append(endpoint_107_swap_dict[channel])
        else:
            vgain_dict[key] = map_dictionary[key]
    if(len(channel_cable_swap_list) != 0):
        vgain_dict[104] = channel_cable_swap_list
    return vgain_dict



def process_csv(file_path):
    # Cargar el CSV
    df = pd.read_csv(file_path, header=None, names=["run_number", "date", "endpoints", "comments_1", "comments_2"])
    
    # Función para dividir "nombre: valor" con separador ";"
    def split_comments_1(comment):
        if pd.isna(comment):  # Manejar valores NaN
            return {}
        items = comment.split("; ")
        result = {}
        for item in items:
            parts = item.split(":", 1)  # Asegurar que solo se divide en 2 partes máximo
            if len(parts) == 2:
                result[parts[0].strip()] = parts[1].strip()
        return result

    # Función para dividir "nombre: valor" con separador "-"
    def split_comments_2(comment):
        if pd.isna(comment):  # Manejar valores NaN
            return {}
        items = comment.split(" - ")
        result = {}
        for item in items:
            parts = item.split(": ", 1)  # Asegurar que solo se divide en 2 partes máximo
            if len(parts) == 2:
                result[parts[0].strip()] = parts[1].strip()
        return result

    # Aplicar la separación
    df_comments_1 = df["comments_1"].apply(split_comments_1).apply(pd.Series)
    df_comments_2 = df["comments_2"].apply(split_comments_2).apply(pd.Series)

    # Concatenar los nuevos DataFrames con el original
    df_final = pd.concat([df[["run_number", "date", "endpoints"]], df_comments_1, df_comments_2], axis=1)

    return df_final

# Ejemplo de uso
csv_file = 'vgain_scans.csv'  # Reemplázalo con el nombre de tu archivo
df_result = process_csv(csv_file)
print(df_result.keys())
data = []
run_channels = []
columns_top_csv = ['date','run','ov','vgain','integrator','mask','width','intensity']
columns_run_csv = ['run','channels']

ep2apa = {
  104: 1,
  109: 2,
  111: 3,
  112: 4,
  113: 4
}

for i in range(len(df_result["run_number"])):
    data.append([df_result['date'][i],
                 df_result['run_number'][i],
                 getOvValues(df_result['PDE'][i]),
                 df_result['VGAIN'][i],
                 0,
                 df_result['channel_mask'][i],
                 df_result['pulse1_width_ticks'][i],
                 df_result['Pulse_bias_percent_270nm'][i]
                ])
    s = df_result['endpoints'][i]
    lst = [int(x.strip()) for x in s.split(",")]
    channels_list = []
    #for batch_number in range(1,3):
    for endpoint in lst:
        print(df_result['endpoints'][i])
        apa = ep2apa[endpoint]
        pde = getPDEValues(df_result['PDE'][i])
        try:
            #LED_conf = (int(df_result['channel_mask'][i]), int(df_result['pulse1_width_ticks'][i]), int(df_result['Pulse_bias_percent_270nm'][i]))
            #vgain_dict = processCableMapSwap(ch_conf[batch_number][apa][pde][LED_conf])
            #for key in vgain_dict.keys():
            LED_intensity = int(df_result['Pulse_bias_percent_270nm'][i])
            ep_ch_list = ch_conf[endpoint][LED_intensity]
            for ch in ep_ch_list:
                channels_list.append(ch + endpoint*100)
            #print(vgain_dict)

        except Exception as e:
            # Print the line number where the exception occurred
            tb = traceback.extract_tb(e.__traceback__)
            print(f"Exception caught at line {tb[-1].lineno}: {type(e).__name__}: {e}")
            print(f"Run number: {df_result['run_number'][i]}.")
            with open("error.log", "a") as f:
                f.write(f"Exception caught at line {tb[-1].lineno}: {type(e).__name__}: {e}\n")
                f.write(f"Run number: {df_result['run_number'][i]}.\n")
    run_channels.append([df_result['run_number'][i],channels_list])
df_ = pd.DataFrame(data, columns=columns_top_csv)
df_ch = pd.DataFrame(run_channels, columns=columns_run_csv)
df_.to_csv('vgain_top_level.csv', index=False)
df_ch.to_csv('vgain_channels.csv', index=False)
print(type(df_ch["run"]))
print(df_)
print(df_ch)