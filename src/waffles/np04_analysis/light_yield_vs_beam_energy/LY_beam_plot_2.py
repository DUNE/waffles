import plotly.graph_objects as go
from useful_data import *

'''for ch, data in CH_data.items():
    x = list(data.keys())
    y = [data[key]['mean'] for key in x]
    y_err = [data[key]['e_mean'] for key in x]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type='data',
                array=y_err,
                visible=True
            ),
            mode='markers+lines',
            marker=dict(color='blue', size=8),
            line=dict(color='blue', width=2),
            name='Mean with error'
        )
    )


    fig.update_layout(
        title=f'LY vs Energy beam - Endpoint 109 Channel {ch:.0f}',
        xaxis_title='Energy beam (GeV)',
        yaxis_title='Mean integral',
    )


    fig.write_image(f"LY_vs_E_ch_{ch:.0f}.png")


    fig.show()'''
    
    
    

for e, e_data in ch_35.items():
    fig = go.Figure()

    start = 0
    stop = 2000000.0
    n = 100.0
    bin = (stop - start ) / n
    # Aggiunta del tracciato istogramma
    fig = go.Figure(data=[go.Histogram(x=e_data['int_list'], xbins=dict(start=0, end=2000000, size = bin))])


    # Impostazioni del layout
    fig.update_layout(
        title='Integrated charge distribution',
        xaxis_title='Integrated charge',
        yaxis_title='Counts',
        bargap=0.1
    )

    # Mostra il grafico
    fig.write_image(f"Charge_hist_ch35_e{e}.png")
    fig.show()