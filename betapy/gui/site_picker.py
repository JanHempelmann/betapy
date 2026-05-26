"""
Reference Site Picker — Tool 2.

Left panel  : StructureView (3D, PyVista) with a red marker at the
              current reference site.
Right panel : coordinate controls, snap-to-atom, cutoff, export.

The user places the reference site by:
  a) Typing fractional coordinates into the spin boxes, or
  b) Using the snap-to-atom dropdown (precise), or
  c) Clicking an atom in the 3D view (PyVista's built-in picker).
"""

import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QLineEdit,
    QGroupBox, QFileDialog, QMessageBox, QComboBox,
    QGridLayout, QCheckBox, QProgressDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from betapy.core.io import read_refpos, write_refpos
from betapy.core.projection import find_refsite_pairs, refsite_results_to_dataframes
from betapy.gui.structure_view import StructureView
from betapy.gui.refsite_viewer import RefsitePFCWidget


class _RefsiteWorker(QThread):
    """Runs find_refsite_pairs off the main thread."""

    finished = pyqtSignal(list, list)   # offsite_results, onsite_results
    error    = pyqtSignal(str)

    def __init__(self, supercell, fc_data, ref_frac,
                 cutoff, exclude_species, parent=None):
        super().__init__(parent)
        self._supercell       = supercell
        self._fc_data         = fc_data
        self._ref_frac        = ref_frac
        self._cutoff          = cutoff
        self._exclude_species = exclude_species

    def run(self):
        try:
            offsite, onsite = find_refsite_pairs(
                self._supercell,
                self._fc_data['atomic_pairs'],
                self._fc_data['force_matrices'],
                self._ref_frac,
                self._cutoff,
                exclude_species=self._exclude_species,
                show_progress=False,
            )
            self.finished.emit(offsite, onsite)
        except Exception as e:
            self.error.emit(str(e))


class SitePickerWidget(QWidget):
    """Self-contained reference site picker tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.supercell  = None
        self.fc_data    = None
        self._ref_frac  = np.array([0.5, 0.5, 0.5])
        self._positions = []
        self._last_offsite = []
        self._last_onsite  = []
        self._last_label   = 'custom_site'

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Outer vertical splitter: 3D+controls on top, pFC viewer full-width below
        v_splitter = QSplitter(Qt.Vertical)
        outer.addWidget(v_splitter)

        # Top row: horizontal splitter — 3D view left, controls right
        splitter = QSplitter(Qt.Horizontal)

        self.structure_view = StructureView(self)
        # Connect PyVista's point picker so clicking an atom snaps the site
        self.structure_view.plotter.enable_point_picking(
            callback=self._on_3d_pick,
            show_message=False,
            show_point=False,
        )
        splitter.addWidget(self.structure_view)

        # --- Right: controls ---
        ctrl_widget = QWidget()
        ctrl_widget.setFixedWidth(270)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setAlignment(Qt.AlignTop)

        # Fractional coordinate spinboxes
        coord_box = QGroupBox('Reference site (fractional coords)')
        coord_grid = QGridLayout()

        # Position selector (hidden unless REFPOS has multiple positions)
        self._pos_label = QLabel('Position:')
        self._pos_combo = QComboBox()
        self._pos_combo.currentIndexChanged.connect(self._on_pos_changed)
        coord_grid.addWidget(self._pos_label, 0, 0)
        coord_grid.addWidget(self._pos_combo, 0, 1)
        self._pos_label.hide()
        self._pos_combo.hide()

        self._spin = {}
        for i, label in enumerate(['a', 'b', 'c']):
            coord_grid.addWidget(QLabel(label), i + 1, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(6)
            spin.setValue(self._ref_frac[i])
            spin.valueChanged.connect(self._on_spin_changed)
            self._spin[label] = spin
            coord_grid.addWidget(spin, i + 1, 1)
        coord_box.setLayout(coord_grid)
        ctrl_layout.addWidget(coord_box)

        # Label
        label_box = QGroupBox('Site label')
        label_layout = QVBoxLayout()
        self.label_edit = QLineEdit('custom_site')
        label_layout.addWidget(self.label_edit)
        label_box.setLayout(label_layout)
        ctrl_layout.addWidget(label_box)

        # Snap to atom
        snap_box = QGroupBox('Snap to atom')
        snap_layout = QVBoxLayout()
        self.snap_combo = QComboBox()
        self.snap_combo.setEditable(True)
        self.snap_combo.setInsertPolicy(QComboBox.NoInsert)
        self.snap_combo.addItem('(load structure first)')
        # Contains-based filtering so typing "V" matches "12  V  (0.1234, …)"
        self.snap_combo.completer().setFilterMode(Qt.MatchContains)
        self.snap_combo.completer().setCaseSensitivity(Qt.CaseInsensitive)
        btn_snap = QPushButton('Snap to selected atom')
        btn_snap.clicked.connect(self._snap_to_atom)
        snap_layout.addWidget(self.snap_combo)
        snap_layout.addWidget(btn_snap)
        snap_box.setLayout(snap_layout)
        ctrl_layout.addWidget(snap_box)

        # Cutoff
        cutoff_box = QGroupBox('Analysis cutoff (Å)')
        cutoff_layout = QVBoxLayout()
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 30.0)
        self.cutoff_spin.setSingleStep(0.5)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(5.0)
        cutoff_layout.addWidget(self.cutoff_spin)
        cutoff_box.setLayout(cutoff_layout)
        ctrl_layout.addWidget(cutoff_box)

        # Refsite connections
        conn_box = QGroupBox('Refsite connections')
        conn_layout = QVBoxLayout()
        conn_cutoff_row = QHBoxLayout()
        conn_cutoff_row.addWidget(QLabel('Cutoff (Å)'))
        self.conn_cutoff_spin = QDoubleSpinBox()
        self.conn_cutoff_spin.setRange(0.1, 30.0)
        self.conn_cutoff_spin.setSingleStep(0.5)
        self.conn_cutoff_spin.setDecimals(2)
        self.conn_cutoff_spin.setValue(6.0)
        conn_cutoff_row.addWidget(self.conn_cutoff_spin)
        conn_layout.addLayout(conn_cutoff_row)
        self.btn_connections = QPushButton('Show connections')
        self.btn_connections.setCheckable(True)
        self.btn_connections.clicked.connect(self._toggle_connections)
        conn_layout.addWidget(self.btn_connections)
        conn_box.setLayout(conn_layout)
        ctrl_layout.addWidget(conn_box)

        # Analysis options
        self.chk_exclude_refsite_species = QCheckBox('Exclude refsite-species pairs')
        self.chk_exclude_refsite_species.setChecked(True)
        self.chk_exclude_refsite_species.setToolTip(
            'Exclude off-site pairs where either atom is of the same\n'
            'species as the nearest atom to the reference site.'
        )
        ctrl_layout.addWidget(self.chk_exclude_refsite_species)

        # Actions
        btn_analyse = QPushButton('Run refsite analysis')
        btn_analyse.clicked.connect(self._run_analysis)
        ctrl_layout.addWidget(btn_analyse)

        btn_export_refpos = QPushButton('Export REFPOS…')
        btn_export_refpos.clicked.connect(self._export_refpos)
        ctrl_layout.addWidget(btn_export_refpos)

        btn_export_csv = QPushButton('Export pFCs to CSV…')
        btn_export_csv.clicked.connect(self._export_csv)
        ctrl_layout.addWidget(btn_export_csv)

        self.result_label = QLabel('')
        self.result_label.setWordWrap(True)
        ctrl_layout.addWidget(self.result_label)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_widget)
        splitter.setSizes([800, 270])
        v_splitter.addWidget(splitter)

        # Bottom: pFC viewer spans full width of the tab
        self.pfc_viewer = RefsitePFCWidget()
        self.pfc_viewer.set_structure_view(self.structure_view)
        self.structure_view.colours_changed.connect(self.pfc_viewer._refresh_plot)
        v_splitter.addWidget(self.pfc_viewer)
        v_splitter.setSizes([560, 370])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_unit(self, unit: str):
        self.pfc_viewer.set_unit(unit)

    def load_supercell(self, supercell, fc_data=None):
        """
        Load a supercell for 3D rendering and optionally force-constant data.
        fc_data may be None if FORCE_CONSTANTS has not been loaded yet —
        the structure will render but refsite analysis will be unavailable
        until fc_data is provided via a second call.
        """
        self.supercell = supercell
        if fc_data is not None:
            self.fc_data = fc_data

        self.structure_view.load_supercell(supercell)
        self.structure_view.set_ref_site(self._ref_frac)

        self.snap_combo.clear()
        for i in range(1, supercell.n_atoms + 1):
            sp   = supercell.species(i)
            frac = supercell.positions[i - 1]
            self.snap_combo.addItem(
                f'{i:4d}  {sp}  '
                f'({frac[0]:.4f}, {frac[1]:.4f}, {frac[2]:.4f})'
            )

    # ------------------------------------------------------------------
    # Coordinate controls
    # ------------------------------------------------------------------

    def _set_ref_frac(self, frac):
        """Update internal state and spinboxes, then refresh 3D marker."""
        self._ref_frac = np.asarray(frac, dtype=float)
        for spin in self._spin.values():
            spin.blockSignals(True)
        self._spin['a'].setValue(self._ref_frac[0])
        self._spin['b'].setValue(self._ref_frac[1])
        self._spin['c'].setValue(self._ref_frac[2])
        for spin in self._spin.values():
            spin.blockSignals(False)
        self.structure_view.set_ref_site(self._ref_frac)
        # Keep connection lines in sync when the site moves
        if self.btn_connections.isChecked():
            self.structure_view.set_refsite_bonds(self.conn_cutoff_spin.value())

    def _on_spin_changed(self):
        self._ref_frac = np.array([
            self._spin['a'].value(),
            self._spin['b'].value(),
            self._spin['c'].value(),
        ])
        self.structure_view.set_ref_site(self._ref_frac)

    def _snap_to_atom(self):
        idx = self.snap_combo.currentIndex()
        if self.supercell is None or idx < 0:
            return
        self._set_ref_frac(self.supercell.positions[idx])

    def _on_3d_pick(self, point):
        """
        Called by PyVista when the user clicks a point in the 3D view.
        Finds the nearest atom to the picked Cartesian point and snaps
        the reference site to it.
        """
        if self.supercell is None or point is None:
            return
        sc = self.supercell
        # Convert all atom positions to Cartesian
        cart_positions = sc.positions @ sc.lattice
        dists = np.linalg.norm(cart_positions - np.asarray(point), axis=1)
        nearest = int(np.argmin(dists))
        self._set_ref_frac(sc.positions[nearest])
        # Also sync the snap combo
        self.snap_combo.blockSignals(True)
        self.snap_combo.setCurrentIndex(nearest)
        self.snap_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # REFPOS loading
    # ------------------------------------------------------------------

    def load_refpos(self, path):
        """Read a REFPOS file and update the reference site position."""
        try:
            data = read_refpos(path)
        except Exception:
            return
        if not data['positions']:
            return
        self._positions = data['positions']
        self._last_label = data['label']
        self.label_edit.setText(data['label'])

        if len(self._positions) > 1:
            self._pos_combo.blockSignals(True)
            self._pos_combo.clear()
            for i, pos in enumerate(self._positions):
                self._pos_combo.addItem(
                    f'{i + 1}:  ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})'
                )
            self._pos_combo.setCurrentIndex(0)
            self._pos_combo.blockSignals(False)
            self._pos_label.show()
            self._pos_combo.show()
        else:
            self._pos_label.hide()
            self._pos_combo.hide()

        self._set_ref_frac(self._positions[0])

    def _on_pos_changed(self, index):
        if 0 <= index < len(self._positions):
            self._set_ref_frac(self._positions[index])

    # ------------------------------------------------------------------
    # Connection toggle
    # ------------------------------------------------------------------

    def _toggle_connections(self, checked):
        if checked:
            self.btn_connections.setText('Hide connections')
            self.structure_view.set_refsite_bonds(self.conn_cutoff_spin.value())
        else:
            self.btn_connections.setText('Show connections')
            self.structure_view.set_refsite_bonds(None)

    # ------------------------------------------------------------------
    # Analysis and export
    # ------------------------------------------------------------------

    def _run_analysis(self):
        if self.supercell is None or self.fc_data is None:
            QMessageBox.warning(self, 'No data', 'Load a structure first.')
            return

        sc     = self.supercell
        cutoff = self.cutoff_spin.value()

        exclude_sp = None
        self._analysis_ref_sp = None
        if self.chk_exclude_refsite_species.isChecked():
            dists    = [sc.distance_to_point(i + 1, self._ref_frac)
                        for i in range(sc.n_atoms)]
            near_idx = min(range(sc.n_atoms), key=lambda i: dists[i])
            self._analysis_ref_sp = sc.species(near_idx + 1)
            exclude_sp = {self._analysis_ref_sp}

        self._worker = _RefsiteWorker(
            sc, self.fc_data, self._ref_frac.copy(),
            cutoff, exclude_sp, parent=self,
        )
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)

        self._progress = QProgressDialog(
            'Running refsite analysis…', None, 0, 0, self,
        )
        self._progress.setWindowTitle('betapy')
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()

        self._worker.start()

    def _on_analysis_done(self, offsite, onsite):
        self._progress.close()

        self._last_offsite = offsite
        self._last_onsite  = onsite
        self._last_label   = self.label_edit.text() or 'custom_site'

        self.pfc_viewer.load_data(offsite, self.supercell)

        ref_sp  = self._analysis_ref_sp
        note    = f'  (excl. {ref_sp} pairs)\n' if ref_sp else ''
        pfc_sum = sum(r['mean_pfc'] for r in offsite) if offsite else 0.0
        self.result_label.setText(
            f'Found:\n'
            f'  {len(offsite)} off-site pairs\n'
            f'{note}'
            f'  {len(onsite)} on-site terms\n'
            f'cutoff: {self.cutoff_spin.value():.2f} Å\n'
            f'Σ pFC = {pfc_sum:+.4f} eV/Å²'
        )

    def _on_analysis_error(self, msg):
        self._progress.close()
        QMessageBox.critical(self, 'Analysis error', msg)

    def load_refsite_csv(self, path):
        """Load a refsite pFCs CSV into the pFC viewer panel."""
        self.pfc_viewer.load_from_csv(path)

    def _export_refpos(self):
        label = self.label_edit.text() or 'custom_site'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save REFPOS', 'REFPOS',
            'REFPOS files (REFPOS);;All files (*)',
        )
        if path:
            write_refpos(label, [self._ref_frac.tolist()], path)

    def _export_csv(self):
        if not self._last_offsite and not self._last_onsite:
            QMessageBox.warning(self, 'No results',
                                'Run the analysis first.')
            return
        df_off, df_on = refsite_results_to_dataframes(
            self._last_offsite, self._last_onsite, self._last_label
        )
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save pFC CSV',
            f'{self._last_label}_pFCs.csv',
            'CSV files (*.csv);;All files (*)',
        )
        if not path:
            return
        df_off.to_csv(path, index=False)
        onsite_path = Path(path).with_name(Path(path).stem + '_onsite.csv')
        df_on.to_csv(onsite_path, index=False)
        QMessageBox.information(
            self, 'Saved',
            f'Off-site → {path}\nOn-site  → {onsite_path}',
        )
