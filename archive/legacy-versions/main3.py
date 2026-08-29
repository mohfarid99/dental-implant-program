# -*- coding: utf-8 -*-
"""
main3.py
- Integrates your existing logic (browse/convert, open IAN tool, open recommendation+Napari)
- Matches the new Designer layout (extra tab "3. Set Dental Implant Angle")
- Safe to run directly (no need to regenerate every time)
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess, os, sys, importlib.util

# ----------------------------
# (optional) quick environment diag
# ----------------------------
def _diagnose_env(parent=None, title="Python Env"):
    spec = importlib.util.find_spec("napari")
    QtWidgets.QMessageBox.information(
        parent, title,
        f"PYTHON: {sys.executable}\n"
        f"napari found: {spec is not None}\n"
        f"napari spec: {spec}\n"
    )

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 520)

        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 20, 860, 480))
        self.tabWidget.setObjectName("tabWidget")

        # ----------------------------
        # TAB 1: Load Patient Folder
        # ----------------------------
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab)
        self.tabWidget_2.setGeometry(QtCore.QRect(10, 20, 600, 360))
        self.tabWidget_2.setObjectName("tabWidget_2")

        # subtab: Convert to DICOMDIR
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.pushButton = QtWidgets.QPushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(300, 20, 200, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 70, 120, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit.setGeometry(QtCore.QRect(20, 20, 260, 32))
        self.lineEdit.setObjectName("lineEdit")
        self.tabWidget_2.addTab(self.tab_3, "")

        # subtab: Load DICOMDIR Files
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_4)
        self.lineEdit_2.setGeometry(QtCore.QRect(20, 30, 260, 32))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_3.setGeometry(QtCore.QRect(300, 30, 200, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget_2.addTab(self.tab_4, "")

        self.tabWidget.addTab(self.tab, "")

        # ----------------------------
        # TAB 2: Annotate The Jaw Curve
        # ----------------------------
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 40, 250, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.tabWidget.addTab(self.tab_2, "")

        # ----------------------------
        # TAB 3: Set Dental Implant Angle (placeholder for now)
        # ----------------------------
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        # you can add angle controls later (sliders/spinboxes/preview)
        self.tabWidget.addTab(self.tab_7, "")

        # ----------------------------
        # TAB 4: Dental Implant Recommendation
        # ----------------------------
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_3.setGeometry(QtCore.QRect(30, 50, 260, 32))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_5.setGeometry(QtCore.QRect(310, 50, 200, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_6.setGeometry(QtCore.QRect(30, 100, 260, 32))
        self.pushButton_6.setObjectName("pushButton_6")
        self.tabWidget.addTab(self.tab_5, "")

        # ----------------------------
        # TAB 5: Dental Implant Database
        # ----------------------------
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.tableView = QtWidgets.QTableView(self.tab_6)
        self.tableView.setGeometry(QtCore.QRect(40, 160, 760, 230))
        self.tableView.setObjectName("tableView")
        self.textEdit = QtWidgets.QTextEdit(self.tab_6)
        self.textEdit.setGeometry(QtCore.QRect(180, 20, 180, 28))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab_6)
        self.textEdit_2.setGeometry(QtCore.QRect(180, 60, 180, 28))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.tab_6)
        self.textEdit_3.setGeometry(QtCore.QRect(180, 100, 180, 28))
        self.textEdit_3.setObjectName("textEdit_3")

        self.label = QtWidgets.QLabel(self.tab_6)
        self.label.setGeometry(QtCore.QRect(50, 24, 120, 20))
        font = QtGui.QFont(); font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.tab_6)
        self.label_2.setGeometry(QtCore.QRect(50, 64, 120, 20))
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.tab_6)
        self.label_3.setGeometry(QtCore.QRect(50, 104, 120, 20))
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.pushButton_7 = QtWidgets.QPushButton(self.tab_6)
        self.pushButton_7.setGeometry(QtCore.QRect(380, 100, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.tab_6)
        self.pushButton_8.setGeometry(QtCore.QRect(480, 100, 93, 28))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.tab_6)
        self.pushButton_9.setGeometry(QtCore.QRect(580, 100, 93, 28))
        self.pushButton_9.setObjectName("pushButton_9")

        self.tabWidget.addTab(self.tab_6, "")

        # ----------------------------
        # SIGNALS (wire your old logic)
        # ----------------------------
        self.pushButton.clicked.connect(self.browse_directory)       # Load Patient Folder (tab1/subtab1)
        self.pushButton_2.clicked.connect(self.run_convert)          # Convert (tab1/subtab1)
        self.pushButton_3.clicked.connect(self.browse_dicomdir)      # Load DICOMDIR Files (tab1/subtab2)
        self.pushButton_4.clicked.connect(self.open_ian_tool)        # Annotate The Jaw Curve (tab2)
        self.pushButton_5.clicked.connect(self.load_jaw_curve)       # Load The Jaw Curve (tab4)
        self.pushButton_6.clicked.connect(self.show_recommendation)  # Get Dental Implant Recommendation (tab4)

        # Optional CRUD (tab5) — placeholders
        self.pushButton_7.clicked.connect(lambda: QtWidgets.QMessageBox.information(None, "Insert", "Insert implant (TODO)"))
        self.pushButton_8.clicked.connect(lambda: QtWidgets.QMessageBox.information(None, "Update", "Update implant (TODO)"))
        self.pushButton_9.clicked.connect(lambda: QtWidgets.QMessageBox.information(None, "Delete", "Delete implant (TODO)"))

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # Keep references for spawned windows
        self._rec_win = None
        self._rec_ui  = None

    # ----------------------------
    # Actions / Logic
    # ----------------------------
    def browse_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Patient Folder", "")
        if directory:
            self.lineEdit.setText(directory)

    def browse_dicomdir(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select DICOMDIR", "", "DICOMDIR (DICOMDIR);;All Files (*)")
        if fname:
            self.lineEdit_2.setText(fname)

    def load_jaw_curve(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Load Jaw Curve", "", "JSON/CSV (*.json *.csv);;All Files (*)")
        if fname:
            self.lineEdit_3.setText(fname)

    def run_convert(self):
        directory = self.lineEdit.text().strip()
        if not directory or not os.path.isdir(directory):
            QtWidgets.QMessageBox.warning(None, "Error", "Please select a valid directory first.")
            return

        try:
            # Example: run dcmodify (adjust tags/options to your needs)
            subprocess.run(
                ["dcmodify", "-nb", "-gin", "-i", "(0020,0010)=STUDY123", "-i", "(0020,0011)=1", "*."],
                cwd=directory, shell=True, check=True
            )
            # Example: run dcmmkdir (create DICOMDIR)
            subprocess.run(["dcmmkdir", "--recurse"], cwd=directory, shell=True, check=True)

            QtWidgets.QMessageBox.information(None, "Success", "Conversion finished successfully!")
        except subprocess.CalledProcessError as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Command failed:\n{e}")

    def show_recommendation(self):
        """
        Opens your 'recommendation' window which embeds Napari at the right of the table.
        Assumes you have 'recommendation.py' with class Ui_MainWindow (as we discussed earlier).
        """
        try:
            os.environ.setdefault("QT_API", "pyqt5")
            os.environ.setdefault("VISPY_USE_APP", "pyqt5")

            from recommendation import Ui_MainWindow  # your existing module
            self._rec_win = QtWidgets.QMainWindow()
            self._rec_ui = Ui_MainWindow()
            self._rec_ui.setupUi(self._rec_win)
            self._rec_win.setWindowTitle("Recommendation + Napari")
            self._rec_win.show()
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(
                None, "Error",
                "Failed to open recommendation window:\n\n"
                + "".join(traceback.format_exception(type(e), e, e.__traceback__))
            )

    def open_ian_tool(self):
        exe_path = r"C:\Users\user\Documents\[Exe-Only] IAN Annotation Tools Work\exe.win-amd64-3.7\IAN Annotation Tool.exe"
        if not os.path.isfile(exe_path):
            QtWidgets.QMessageBox.warning(None, "Not found", f"Executable not found:\n{exe_path}")
            return
        try:
            subprocess.Popen([exe_path], cwd=os.path.dirname(exe_path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Launch failed", str(e))

    # ----------------------------
    # Texts
    # ----------------------------
    def retranslateUi(self, Dialog):
        _ = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_("Dialog", "Dental Implant Planning Suite"))

        # Tab 1
        self.pushButton.setText(_("Dialog", "Load Patient Folder"))
        self.pushButton_2.setText(_("Dialog", "Convert"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _("Dialog", "Convert to DICOMDIR"))
        self.pushButton_3.setText(_("Dialog", "Load Patient Folder"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _("Dialog", "Load DICOMDIR Files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _("Dialog", "1. Load Patient Folder"))

        # Tab 2
        self.pushButton_4.setText(_("Dialog", "Annotate The Jaw Curve"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _("Dialog", "2. Annotate The Jaw Curve"))

        # Tab 3
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _("Dialog", "3. Set Dental Implant Angle"))

        # Tab 4
        self.pushButton_5.setText(_("Dialog", "Load The Jaw Curve"))
        self.pushButton_6.setText(_("Dialog", "Get Dental Implant Recommendation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _("Dialog", "4. Dental Implant Recommendation"))

        # Tab 5
        self.label.setText(_("Dialog", "ID"))
        self.label_2.setText(_("Dialog", "Diameter (mm)"))
        self.label_3.setText(_("Dialog", "Height (mm)"))
        self.pushButton_7.setText(_("Dialog", "Insert"))
        self.pushButton_8.setText(_("Dialog", "Update"))
        self.pushButton_9.setText(_("Dialog", "Delete"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _("Dialog", "Dental Implant Database"))

# ------------------------------------------------
# Entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
