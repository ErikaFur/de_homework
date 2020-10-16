import sys
from PyQt5.QtWidgets import (QApplication, QComboBox, QInputDialog, QWidget,
                             QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QCheckBox, QMenu, QMenuBar, QMainWindow, QPushButton, QSpinBox, QTextEdit,
                             QVBoxLayout, QLineEdit, QTabWidget,QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QCoreApplication, Qt, QRect, pyqtSlot
from numpy import double

import de

class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.centralWidget = QWidget()
        self.lay = QVBoxLayout(self.centralWidget)
        self.x_init = 1
        self.y_init = -2
        self.x_end = 10
        self.n = 9
        self.N = 30
        self.step = 1
        self.check = "1111"

        self.initUI()
        self.setCentralWidget(self.centralWidget)
        self.move(300, 200)
        self.setGeometry(500, 400, 700, 600)
        self.setWindowTitle('DE')
        self.show()

    def initUI(self):
        self.tabs = QTabWidget()
        self.lay.addWidget(self.tabs)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab6 = QWidget()
        self.tabs.addTab(self.tab1, "main")
        self.tabs.addTab(self.tab2, "func_plot")
        self.tabs.addTab(self.tab3, "GTE")
        self.tabs.addTab(self.tab4, "LTE")
        self.tabs.addTab(self.tab5, "max_GTE")
        self.tabs.addTab(self.tab6, "max_LTE")
        self.tabs.resize(300, 200)

        self.tab1.layout = QVBoxLayout(self)
        self.EulerCheckbox = self.checkboxes("Euler method", 0, 400, 410)
        self.IEulerCheckbox = self.checkboxes("Improved Euler\n method", 1, 400, 440)
        self.RKCheckbox = self.checkboxes("Runte-Kutta", 2, 400, 470)
        self.ExactCheckbox = self.checkboxes("Exact", 3, 400, 500)
        self.a_param = self.textboxes(130, 412)
        self.y_param = self.textboxes(130, 442)
        self.b_param = self.textboxes(130, 472)
        self.n_param = self.textboxes(130, 502)
        self.N_param = self.textboxes(130, 532)
        self.step_param = self.textboxes(130, 562)
        self.b_exe = self.button_execute()
        self.b_exit = self.button_exit()

        self.tab1_1 = QWidget()
        self.tab1_1.layout = QVBoxLayout(self)
        self.tab1_1.layout.addWidget(self.EulerCheckbox)
        self.tab1_1.layout.addWidget(self.IEulerCheckbox)
        self.tab1_1.layout.addWidget(self.RKCheckbox)
        self.tab1_1.layout.addWidget(self.ExactCheckbox)
        self.tab1.layout.addLayout(self.tab1_1.layout)

        self.tab1_2 = QWidget()
        self.tab1_2.layout = QHBoxLayout(self)
        self.tab1_2_1 = QWidget()
        self.tab1_2_1.layout = QVBoxLayout(self)
        self.tab1_2_1.layout.addWidget(QLabel("x init (a) =",self))
        self.tab1_2_1.layout.addWidget(QLabel("y init =",self))
        self.tab1_2_1.layout.addWidget(QLabel("x end (b) =",self))
        self.tab1_2_1.layout.addWidget(QLabel("n =",self))
        self.tab1_2_1.layout.addWidget(QLabel("N =",self))
        self.tab1_2_1.layout.addWidget(QLabel("step =",self))
        self.tab1_2_2 = QWidget()
        self.tab1_2_2.layout = QVBoxLayout(self)
        self.tab1_2_2.layout.addWidget(self.a_param)
        self.tab1_2_2.layout.addWidget(self.y_param)
        self.tab1_2_2.layout.addWidget(self.b_param)
        self.tab1_2_2.layout.addWidget(self.n_param)
        self.tab1_2_2.layout.addWidget(self.N_param)
        self.tab1_2_2.layout.addWidget(self.step_param)
        self.tab1_2.layout.addLayout(self.tab1_2_1.layout)
        self.tab1_2.layout.addLayout(self.tab1_2_2.layout)
        self.tab1.layout.addLayout(self.tab1_2.layout)

        self.tab1_3 = QWidget()
        self.tab1_3.layout = QVBoxLayout(self)
        self.tab1_3.layout.addWidget(self.b_exe)
        self.tab1_3.layout.addWidget(self.b_exit)
        self.tab1.layout.addLayout(self.tab1_3.layout)


        self.tab2.layout = QVBoxLayout(self)
        self.label2 = QLabel(self)
        pixmap = QPixmap('plot_functions.png')
        self.label2.setPixmap(pixmap)
        self.tab2.layout.addWidget(self.label2)

        self.tab3.layout = QVBoxLayout(self)
        self.label3 = QLabel(self)
        pixmap = QPixmap('plot_GTE.png')
        self.label3.setPixmap(pixmap)
        self.tab3.layout.addWidget(self.label3)

        self.tab4.layout = QVBoxLayout(self)
        self.label4 = QLabel(self)
        pixmap = QPixmap('plot_LTE.png')
        self.label4.setPixmap(pixmap)
        self.tab4.layout.addWidget(self.label4)

        self.tab5.layout = QVBoxLayout(self)
        self.label5 = QLabel(self)
        pixmap = QPixmap('plot_max_GTE.png')
        self.label5.setPixmap(pixmap)
        self.tab5.layout.addWidget(self.label5)

        self.tab6.layout = QVBoxLayout(self)
        self.label6 = QLabel(self)
        pixmap = QPixmap('plot_max_LTE.png')
        self.label6.setPixmap(pixmap)
        self.tab6.layout.addWidget(self.label6)

        self.tab1.setLayout(self.tab1.layout)
        self.tab2.setLayout(self.tab2.layout)
        self.tab3.setLayout(self.tab3.layout)
        self.tab4.setLayout(self.tab4.layout)
        self.tab5.setLayout(self.tab5.layout)
        self.tab6.setLayout(self.tab6.layout)
        #print(self.label6)

    def button_exit(self):
        btn_exit = QPushButton('Exit', self)
        btn_exit.clicked.connect(QCoreApplication.instance().quit)
        btn_exit.resize(60, 50)
        btn_exit.move(440, 550)
        return btn_exit
    def button_execute(self):
        self.button = QPushButton('Solve this!', self)
        self.button.resize(180, 50)
        self.button.move(130, 550)
        self.button.clicked.connect(self.buttonClicked_functions)
        return self.button

    def checkboxes(self,name: str, pos, x , y):
        cb = QCheckBox(name, self)
        cb.move(x , y)
        cb.toggle()
        cb.stateChanged.connect(
            lambda state = cb.isChecked(), pas = pos: self.changeTitle(state, pas))
        return cb
    def textboxes(self, x , y):
        self.textbox = QLineEdit(self)
        self.textbox.move(x, y)
        self.textbox.resize(100, 20)
        return self.textbox
    def set_text(self,x,y, name:str):
        self.lbl = QLabel(self)
        self.lbl.setText(name + " =")
        self.lbl.move(x - 40, y - 5)

    def changeTitle(self, state, pos):
        if state == Qt.Checked:
            self.check = self.check[:pos] + "1" + self.check[1+pos:]
        else:
            self.check = self.check[:pos] + "0" + self.check[1+pos:]

    def buttonClicked_functions(self):
        self.x_init = self.x_init if self.a_param.text() == "" else self.a_param.text()
        self.y_init = self.y_init if self.y_param.text() == "" else self.y_param.text()
        self.x_end = self.x_end if self.b_param.text() == "" else self.b_param.text()
        self.n = self.n if self.n_param.text() == "" else self.n_param.text()
        self.N = self.N if self.N_param.text() == "" else self.N_param.text()
        self.step = self.step if self.step_param.text() == "" else self.step_param.text()
        output = de.Grid(x_init = double(self.x_init), y_init=double(self.y_init),
                X= double(self.x_end), n = double(self.n),check=self.check)
        output.plot_functions()

        output.plot_GTE()

        output.plot_LTE()

        output.plot_max_GTE(N=self.N,step=self.step)

        output.plot_max_LTE(N=self.N,step=self.step)

        self.tab2.layout.removeWidget(self.label2)
        self.label2.setPixmap(QPixmap('plot_functions.png'))
        self.tab2.layout.addWidget(self.label2)

        self.tab3.layout.removeWidget(self.label3)
        self.label3.setPixmap(QPixmap('plot_GTE.png'))
        self.tab3.layout.addWidget(self.label3)

        self.tab4.layout.removeWidget(self.label4)
        self.label4.setPixmap(QPixmap('plot_LTE.png'))
        self.tab4.layout.addWidget(self.label4)

        self.tab5.layout.removeWidget(self.label5)
        self.label5.setPixmap(QPixmap('plot_max_GTE.png'))
        self.tab5.layout.addWidget(self.label5)

        self.tab6.layout.removeWidget(self.label6)
        self.label6.setPixmap(QPixmap('plot_max_LTE.png'))
        self.tab6.layout.addWidget(self.label6)

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())