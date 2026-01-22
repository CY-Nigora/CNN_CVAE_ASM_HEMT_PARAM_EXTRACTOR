#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
used to  view generated h5 files to check if the have the same feature as the real meas data
"""


import sys
import os
import numpy as np
import h5py

from PyQt5 import QtCore, QtGui, QtWidgets

# Matplotlib embedding in PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=4.8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 1x3 subplots (left: rows vs VDS, middle: cols vs VGS, right: gm)
        self.ax_left, self.ax_right, self.ax_gm = self.fig.subplots(1, 3)
        super().__init__(self.fig)
        self.setParent(parent)
        # Do not call tight_layout here; we'll control layout in draw


class H5Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer — (N,7,121) samples")
        self.resize(1200, 700)

        # State
        self.h5file = None  # type: h5py.File | None
        self.dataset = None  # type: h5py.Dataset | None
        self.dataset_x = None  # h5py.Dataset for 'X'
        self.N = 0
        self.segment_len = 121
        self.num_rows = 7
        # Colorbar handles (left: VGS mapping; gm side: VDS mapping)
        self.cbar_left = None
        self.cbar_right = None
        # Physical axes ranges
        self.vds_start = -3.5
        self.vds_end = 8.5
        self.vgs_start = 1.0
        self.vgs_end = float(self.num_rows)

        # Widgets
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # File picker row
        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select or enter .h5 file path …")
        file_row.addWidget(self.path_edit, stretch=1)

        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self.on_browse)
        file_row.addWidget(self.browse_btn)

        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self.on_open)
        file_row.addWidget(self.open_btn)

        vbox.addLayout(file_row)

        # Canvas
        self.canvas = MplCanvas(self, width=12, height=4.8, dpi=100)
        vbox.addWidget(self.canvas, stretch=1)

        # Colorbar axes (attach to left & gm axes)
        self.div_left = make_axes_locatable(self.canvas.ax_left)
        self.cax_left = self.div_left.append_axes("right", size="3%", pad="2%")
        self.div_gm = make_axes_locatable(self.canvas.ax_gm)
        self.cax_gm = self.div_gm.append_axes("right", size="3%", pad="2%")

        # Bottom info panel (single row table)
        self.ax_info = self.canvas.fig.add_axes([0.06, 0.03, 0.88, 0.05])  # [left, bottom, width, height]
        self.ax_info.axis("off")
        # self.ax_info.set_title("Params (Y)")

        # Controls row
        ctrl_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("No file opened")
        ctrl_row.addWidget(self.status_label)
        ctrl_row.addStretch(1)

        ctrl_row.addWidget(QtWidgets.QLabel("Sample index:"))
        self.idx_spin = QtWidgets.QSpinBox()
        self.idx_spin.setRange(0, 0)
        self.idx_spin.setEnabled(False)
        self.idx_spin.valueChanged.connect(self.update_plot)
        ctrl_row.addWidget(self.idx_spin)

        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.on_prev)
        ctrl_row.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.on_next)
        ctrl_row.addWidget(self.next_btn)

        vbox.addLayout(ctrl_row)

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=self.on_prev)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=self.on_next)

    # --------------------------- UI Actions --------------------------- #
    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select HDF5 file", os.getcwd(), "HDF5 Files (*.h5 *.hdf5);;All Files (*)")
        if path:
            self.path_edit.setText(path)

    def on_open(self):
        path = self.path_edit.text().strip()
        if not path:
            self._error("Please enter or choose a file path")
            return
        if not os.path.exists(path):
            self._error("File does not exist: " + path)
            return
        try:
            # Quick sanity check
            if not h5py.is_hdf5(path):
                raise RuntimeError("Not a valid HDF5 file (.h5 signature mismatch)")

            # Close previous file if any
            if self.h5file is not None:
                try:
                    self.h5file.close()
                except Exception:
                    pass
                self.h5file = None
                self.dataset = None

            self.h5file = h5py.File(path, 'r')
            if 'X' in self.h5file:
                self.dataset_x = self.h5file['X']
            else:
                raise RuntimeError("Dataset 'X' not found in file")

            shape = self.dataset_x.shape
            if len(shape) != 3:
                raise RuntimeError(f"Dataset 'X' shape is {shape}, which is not 3D")

            self.N = int(shape[0])
            self.num_rows = int(shape[1])
            self.segment_len = int(shape[2])
            self.vgs_end = float(self.num_rows)

            self.idx_spin.blockSignals(True)
            self.idx_spin.setRange(0, max(0, self.N - 1))
            self.idx_spin.setValue(0)
            self.idx_spin.setEnabled(self.N > 0)
            self.idx_spin.blockSignals(False)

            self.prev_btn.setEnabled(self.N > 1)
            self.next_btn.setEnabled(self.N > 1)

            msg = f"Opened: {os.path.basename(path)}  | X dataset: '{self.dataset_x.name}'  | Shape: {self.dataset_x.shape}  | dtype: {self.dataset_x.dtype}"
            if (self.num_rows, self.segment_len) != (7, 121):
                msg += "  (⚠ differs from (7,121))"
            self.status_label.setText(msg)

            self._draw_sample(0)

        except Exception as e:
            self._error(f"Failed to open: {e}")

    def on_prev(self):
        if not self.idx_spin.isEnabled():
            return
        i = self.idx_spin.value()
        if i > 0:
            self.idx_spin.setValue(i - 1)
        else:
            self.idx_spin.setValue(self.N - 1)

    def on_next(self):
        if not self.idx_spin.isEnabled():
            return
        i = self.idx_spin.value()
        if i < self.N - 1:
            self.idx_spin.setValue(i + 1)
        else:
            self.idx_spin.setValue(0)

    # --------------------------- Helpers --------------------------- #
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            if self.h5file is not None:
                self.h5file.close()
        except Exception:
            pass
        event.accept()

    def update_plot(self):
        idx = self.idx_spin.value()
        self._draw_sample(idx)

    def _draw_sample(self, idx: int):
        if self.dataset_x is None:
            return
        try:
            sample = np.array(self.dataset_x[idx, :, :])  # (rows, seg)
            if sample.shape != (self.num_rows, self.segment_len):
                raise ValueError(f"Unexpected sample shape: got {sample.shape}, expected ({self.num_rows},{self.segment_len})")

            axL = self.canvas.ax_left
            axR = self.canvas.ax_right
            axG = self.canvas.ax_gm
            axL.clear(); axR.clear(); axG.clear()

            # Build physical ranges
            Vds_range = np.linspace(self.vds_start, self.vds_end, self.segment_len) if self.segment_len > 1 else np.array([self.vds_start])
            Vgs_range = np.linspace(self.vgs_start, self.vgs_end, self.num_rows) if self.num_rows > 1 else np.array([self.vgs_start])


            plt.rcParams.update({'font.size': 15})
            # ---------------- Left: rows vs VDS (color by VGS) ----------------
            legend_values = np.linspace(1, self.num_rows, self.num_rows)
            cmap_left = cm.winter
            norm_vgs = colors.Normalize(vmin=1.0, vmax=float(self.num_rows))
            for r in range(self.num_rows):
                color = cmap_left(norm_vgs(legend_values[r]))
                axL.plot(Vds_range, sample[r, :], linewidth=2.0, label=f"VGS={r+1} V", color=color)
            axL.set_xlim(self.vds_start, self.vds_end)
            y_min = float(np.nanmin(sample)); y_max = float(np.nanmax(sample))
            pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            axL.set_ylim(y_min - pad, y_max + pad)
            axL.set_xlabel("VDS (V)", fontsize=15)
            axL.set_ylabel("IDS (A)", fontsize=15)
            axL.set_title(f"Sample {idx} — IDS vs VDS", fontsize=15)
            axL.grid(True)
            if self.num_rows <= 12:
                axL.legend(loc='best', fontsize=15)

            # Left colorbar: VGS
            sm_left = cm.ScalarMappable(cmap=cmap_left, norm=norm_vgs)
            sm_left.set_array([])
            if self.cbar_left is None:
                self.cbar_left = self.canvas.fig.colorbar(sm_left, cax=self.cax_left)
                self.cbar_left.set_label("VGS (V)", fontsize=15)
            else:
                self.cbar_left.update_normal(sm_left)

            # ---------------- Middle: columns vs VGS (color by VDS) ----------------
            cmap_right = cm.plasma
            norm_vds = colors.Normalize(vmin=self.vds_start, vmax=self.vds_end)
            for c in range(self.segment_len):
                color = cmap_right(norm_vds(Vds_range[c]))
                axR.plot(Vgs_range, sample[:, c], linewidth=1.2, color=color)
            axR.set_xlim(self.vgs_start, self.vgs_end)
            axR.set_xlabel("VGS (V)", fontsize=15)
            axR.set_title(f"Sample {idx} — IDS vs VGS", fontsize=15)
            axR.grid(True)
            # no legend (too dense)

            # ---------------- Right: gm = ΔIDS/ΔVDS vs VGS (color by VDS) ----------------
            gm = np.diff(sample, axis=0).T  # (seg-1, rows)
            for c in range(gm.shape[0]):  # each VDS step
                color = cmap_right(norm_vds(Vds_range[c]))
                axG.plot(np.arange(1.5,7.5,1), gm[c, :], linewidth=1.1, color=color)
            axG.set_xlim(self.vgs_start, self.vgs_end)
            gmin, gmax = float(np.nanmin(gm)), float(np.nanmax(gm))
            gpad = 0.05 * (gmax - gmin if gmax > gmin else 1.0)
            axG.set_ylim(gmin - gpad, gmax + gpad)
            axG.set_xlabel("VGS (V)", fontsize=15)
            axG.set_title("gm = ΔIDS/ΔVGS", fontsize=15)
            axG.grid(True)

            # Right colorbar for VDS (attach next to gm)
            sm_right = cm.ScalarMappable(cmap=cmap_right, norm=norm_vds)
            sm_right.set_array([])
            if (self.cbar_right is None) or (getattr(self.cbar_right, "ax", None) is not self.cax_gm):
                if self.cbar_right is not None:
                    try:
                        self.cbar_right.remove()
                    except Exception:
                        pass
                self.cbar_right = self.canvas.fig.colorbar(sm_right, cax=self.cax_gm)
                self.cbar_right.set_label("VDS (V)", fontsize=15)
            else:
                self.cbar_right.update_normal(sm_right)

            # ----- Bottom info table: 11 params from dataset 'Y' -----
            self.ax_info.clear()
            self.ax_info.axis("off")
            # self.ax_info.set_title("Params (Y)")

            param_keys = ['VOFF','U0','NS0ACCS','NFACTOR','ETA0','VSAT','VDSCALE','CDSCD','LAMBDA','MEXPACCD','DELTA','TABR']
            values = None
            try:
                if 'Y' in self.h5file:
                    Y = self.h5file['Y']
                    if Y.ndim == 3 and Y.shape[0] == self.N and Y.shape[1] >= 11:
                        values = np.array(Y[idx, :11]).astype(float)
                    elif Y.ndim == 2 and Y.shape[1] == self.N and Y.shape[0] >= 11:
                        values = np.array(Y[:11, idx]).astype(float)
                    elif Y.ndim == 1 and Y.shape[0] >= 11:
                        values = np.array(Y[:11]).astype(float)
            except Exception:
                values = None
            if values is None:
                defaults = {'VOFF':'1.785', 'U0':'0.424', 'NS0ACCS':'2e+17', 'NFACTOR':'1', 'ETA0':'0.06',
                            'VSAT':'8e+4', 'VDSCALE':'5', 'CDSCD':'0.1', 'LAMBDA':'0.01', 'MEXPACCD':'1.5', 'DELTA':'3', 'TBAR':'9e-7'}
                values = np.array([float(defaults[k]) for k in param_keys], dtype=float)

            values = np.asarray(values, dtype=float).reshape(-1)[:len(param_keys)]
            pairs = [f"{k}: {float(v):.3e}" for k, v in zip(param_keys, values)]
            col_w = 1.0 / len(pairs)
            tbl = self.ax_info.table(
                cellText=[pairs], cellLoc='center', colLoc='center',
                colWidths=[col_w] * len(pairs), loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(15)
            tbl.scale(1.1, 4)
            for _, cell in tbl.get_celld().items():
                cell.set_linewidth(0.8)
                cell.set_edgecolor('black')

            # Layout: leave space for bottom info
            self.canvas.fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.13, wspace=0.25)
            self.canvas.draw_idle()
        except Exception as e:
            self._error(f"Plot failed: {e}")

    def _error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = H5Viewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
