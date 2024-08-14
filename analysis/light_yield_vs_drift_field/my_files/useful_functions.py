import re

def extract_run_number(text):
    # Use regular expression to find the pattern 'run_' followed by digits
    match = re.search(r'run_(\d+)', text)
    if match:
        # Return the matched digits
        return match.group(1)
    else:
        return None

#retornar posicoes da matriz para um determinado canal e APA
def return_index(channel, endpoint):
    for k in range(1,5):    
        for i in range (10):
            for j in range(4):
                if (APA_map[k].Data[i][j].Endpoint == endpoint and APA_map[k].Data[i][j].Channel == channel):
                    return k,i,j
    return -1,-1,-1