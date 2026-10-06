from PyQt5.QtWidgets import QMainWindow, QTabWidget

from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
from elma.gui.tabs.inspection_tab import InspectionTab
from elma.gui.tabs.multisine_tab import MultisineDesignerTab


class MainWindow(QMainWindow):
    """
    Top-level window for elma. Each tab is a thin front-end over a
    module of the underlying library, not a place for new business logic.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("elma - dynamic EIS")
        self.resize(1400, 850)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(MultisineDesignerTab(), "Multisine Designer")
        self.tabs.addTab(ExperimentBuilderTab(), "Experiment Builder")
        self.tabs.addTab(InspectionTab(), "Inspection")
