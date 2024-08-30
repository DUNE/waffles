import plotly.graph_objects as go

end109_ch35 = {'+1' : { 'time_offset' : {'start': 15400, 'stop': 15600 },
                        'charge_integral' : {'start': 55, 'stop': 110, 'mean' : 133503, 'std' : 54078}
                      },
               '+2' : { 'time_offset' : {'start': 15400, 'stop': 15600},
                        'charge_integral' : {'start': 55, 'stop': 120, 'mean' : 235853, 'std' : 103976}
                      },
               '+3' : { 'time_offset' : {'start': 15480, 'stop': 15600},
                        'charge_integral' : {'start': 55, 'stop': 120, 'mean' : 368207, 'std' : 284250}
                      },
               '+5' : { 'time_offset' : {'start': 15500, 'stop': 15600},
                        'charge_integral' : {'start': 60, 'stop': 120, 'mean' : 807833, 'std' : 821107}
                      },
               '+7' : { 'time_offset' : {'start': 15480, 'stop': 15570},
                        'charge_integral' : {'start': 60, 'stop': 120, 'mean' : 1093522, 'std' : 575593}
                      }
              }

# Estrai i dati per il grafico
x = list(end109_ch35.keys())
y = [end109_ch35[key]['charge_integral']['mean'] for key in x]
y_err = [end109_ch35[key]['charge_integral']['std'] for key in x]

# Crea il grafico
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
        name='Mean with Std Dev'
    )
)

# Aggiungi layout
fig.update_layout(
    title='LY vs Energy beam',
    xaxis_title='Energy beam (GeV)',
    yaxis_title='Mean integral',
)

# Salva il grafico come immagine PNG
fig.write_image("ly_vs_energy_beam.png")

# Mostra il grafico (opzionale)
fig.show()

