"""
Launch the elma GUI from Spyder.

Two ways to use this, depending on whether you want the console usable while the window is open:

1) Blocking (simplest): just hit F5 in Spyder. The console will be busy until you close the window.

2) Non-blocking: run `%gui qt5` once in the console first, *then* run this script (F5 again). That
   integrates Qt's event loop into IPython's, so the window stays open and responsive while you keep
   using the console. This script detects that automatically and skips the blocking call.

From a terminal use the `elma-gui` command instead (pip install elma[gui]).
"""
from PyQt5.QtWidgets import QApplication

from elma.gui.main_window import MainWindow

try:
    from IPython import get_ipython
    _ip = get_ipython()
    _qt_loop_active = _ip is not None and getattr(_ip, "active_eventloop", None) in ("qt", "qt5")
except ImportError:
    _qt_loop_active = False

app = QApplication.instance() or QApplication([])

window = MainWindow()
window.show()

if not _qt_loop_active:
    app.exec_()
