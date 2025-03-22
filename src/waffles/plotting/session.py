# plotting/session.py
from plotly import graph_objects as go
import plotly.io as pio

class PlotSession:
    def __init__(self,
                 mode='html',
                 template='plotly_white',
                 line_color='black',
                 width=800,
                 height=600):
        self.fig = go.Figure()
        self.mode = mode
        self.template = template
        self.line_color = line_color
        self.width = width
        self.height = height
        self.fig.update_layout(template=self.template)

    def add_trace(self, trace, row=None, col=None):
        if row is not None and col is not None:
            self.fig.add_trace(trace, row=row, col=col)
        else:
            self.fig.add_trace(trace)

    def set_labels(self, xlabel, ylabel):
        self.fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)

    def set_title(self, title: str):
        self.fig.update_layout(title=title)

    def update_axes(self, x_range=None, y_range=None):
        if x_range is not None:
            self.fig.update_xaxes(range=x_range)
        if y_range is not None:
            self.fig.update_yaxes(range=y_range)

    def write(self, filename='plot.html'):
        if self.mode == 'html':
            pio.write_html(self.fig, file=filename, auto_open=True)
        elif self.mode == 'png':
            pio.write_image(self.fig, file=filename.replace('.html', '.png'),
                            format='png', width=self.width, height=self.height)
        else:
            raise ValueError(f"Unknown plotting mode '{self.mode}'")