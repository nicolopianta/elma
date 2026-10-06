"""
Zoom/pan toolbar for the GUI's matplotlib canvases.

`canvas_with_toolbar(canvas)` returns a widget holding the standard
matplotlib navigation toolbar (zoom-to-rectangle, pan, back/forward, save
image) above the canvas, plus the toolbar itself so callers can reset its
history after they rebuild the figure's axes.

For plots that grow while data come in (live potential/current, PEIS/GEIS
points), the Home button re-enables autoscaling instead of restoring the
first view it saw: zooming or panning switches autoscaling off for that
axes (so live updates don't fight the zoom), and Home is the way back to
following the data.
"""
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class ZoomToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None, autoscale_on_home=True):
        super().__init__(canvas, parent)
        self._autoscale_on_home = autoscale_on_home

    def home(self, *args):
        if not self._autoscale_on_home:
            super().home(*args)
            return
        for ax in self.canvas.figure.axes:
            ax.autoscale(True)
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()


def canvas_with_toolbar(canvas, autoscale_on_home=True):
    """Returns (widget, toolbar): the toolbar stacked above `canvas`."""
    toolbar = ZoomToolbar(canvas, autoscale_on_home=autoscale_on_home)
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(toolbar)
    layout.addWidget(canvas)
    return widget, toolbar
