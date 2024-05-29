import plotly
import plotly.graph_objects as go


def custom_legend_name(fig,new_names):
    '''
    Custom legend name for plotly figures

    Args:
        fig    (px.Figure): plotly figure
        new_names (list):      list of new names for the legend

    Returns:
        fig (px.Figure): plotly figure with the custom legend names
    '''
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
        # fig.data[i].hovertemplate = new_name
    # fig.for_each_trace(lambda t: t.update(name = dict[t.name],legendgroup = dict[t.name],hovertemplate = t.hovertemplate.replace(t.name, dict[t.name])))
    return fig

def custom_plotly_layout(fig,xaxis_title ="",yaxis_title="",title="",legend_pos=["out","h"],log_menu=True,fontsize=16,figsize=None,ranges=(None,None),matches=("x","y"),
    tickformat=('.s','.s'),barmode="stack",margin={"auto":True}):
    '''
    Custom layout for plotly figures

    Args:
        fig      (Figure):       plotly figure
        xaxis_title (str):       x axis title (default: "")
        yaxis_title (str):       y axis title (default: "")
        title       (str):       title (default: "")
        legend_pos  (list):      legend position (default: ["out","h"])
        log_menu    (bool):      log menu (default: True)
        fontsize    (int):       font size (default: 16)
        figsize     (tuple):     figure size (default: None)
        ranges      (tuple):     axis ranges (default: (None,None))
        matches     (tuple):     axis matches (default: ("x","y"))
        tickformat  (tuple):     axis tick format (default: ('.s','.s'))
        barmode     (str):       barmode (default: "stack")
        margin      (dict):      figure margin (default: {"auto":True,"color":"white","margin":(0,0,0,0)})
    
    Returns:
        fig (Figure): plotly figure with the custom layout
    '''
    default_margin = {"color":"white","margin":(0,0,0,0)}
    if margin != None:
            for key in default_margin.keys():
                if key not in margin.keys():
                    margin[key] = default_margin[key]

    legend_opt = {"in": dict(yanchor="top", xanchor="right", x=0.99),"out": dict(yanchor="bottom", xanchor="right", x=1.2,y=0.2) }
    if log_menu: fig.update_layout( updatemenus=[ dict( buttons=list([ dict(args=[{"xaxis.type": "linear", "yaxis.type": "linear"}], label="LinearXY", method="relayout"),
                                                             dict(args=[{"xaxis.type": "log", "yaxis.type": "log"}],       label="LogXY",    method="relayout"),
                                                             dict(args=[{"xaxis.type": "linear", "yaxis.type": "log"}],    label="LogY",     method="relayout"),
                                                             dict(args=[{"xaxis.type": "log", "yaxis.type": "linear"}],    label="LogX",     method="relayout") ]),
                          direction="down", pad={"r": 10, "t": 10}, showactive=True, x=-0.1, xanchor="left", y=1.5, yanchor="top" ) ] )
    fig.update_layout( template="presentation", title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, barmode=barmode,
                          font=dict(size=fontsize,family="serif"), legend_title_text='', legend = legend_opt[legend_pos[0]],legend_orientation=legend_pos[1], showlegend=True, paper_bgcolor=margin["color"])
    fig.update_xaxes(matches=matches[0],showline=True,mirror="ticks",zeroline=False,showgrid=True,minor_ticks="inside",tickformat=tickformat[0],range=ranges[0])
    fig.update_yaxes(matches=matches[1],showline=True,mirror="ticks",zeroline=False,showgrid=True,minor_ticks="inside",tickformat=tickformat[1],range=ranges[1])

    if figsize != None: fig.update_layout(width=figsize[0],height=figsize[1])
    if margin["auto"] == False: fig.update_layout(margin=dict(l=margin["margin"][0], r=margin["margin"][1], t=margin["margin"][2], b=margin["margin"][3]))

    return fig

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        if intermed <= cutoff:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )