"""
Shared 3D structure renderer for betapy.

Features:
  - CPK/Jmol atom colours with per-species colour picker
  - Automatic bond drawing with per-species-pair enable/disable toggles
  - On pair selection: centers atom1 at cell centre (minimum-image convention)
    and recomputes bond endpoints from shifted positions so no bonds cross cell
  - Non-selected atoms dimmed when pair highlighted
  - Reference site marker sphere
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QColorDialog,
    QGroupBox, QScrollArea, QCheckBox, QComboBox, QDoubleSpinBox,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal

from betapy.data.elements import (
    covalent_radius, element_colour, display_radius, COLOUR_PRESETS,
)
from betapy.core.constants import SAME_SPECIES_METALS

# -----------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------

BOND_FACTOR           = 1.1
BOND_RADIUS           = 0.06
BOND_COLOUR           = (0.65, 0.65, 0.65)
BOND_OPACITY          = 0.5

HIGHLIGHT_COLOUR      = (1.0, 0.75, 0.0)
HIGHLIGHT_ATOM_FACTOR = 1.4
HIGHLIGHT_BOND_RADIUS = 0.10

REFSITE_COLOUR        = (1.0, 0.2, 0.2)
REFSITE_CUBE_SIZE     = 1.0     # Å per side — larger than any display-radius atom
REFSITE_CUBE_OPACITY  = 0.40
REFSITE_BOND_COLOUR   = (0.85, 0.30, 0.10)
REFSITE_BOND_RADIUS   = 0.04
REFSITE_BOND_OPACITY  = 0.80

DIM_OPACITY           = 0.22   # individual bond highlight: background atoms/bonds
SHELL_DIM_OPACITY     = 0.35   # shell highlight: gentler — context still readable
FULL_OPACITY          = 1.0

# Light frame colour for the unit-cell boundary box — needs to read clearly
# against the widget's dark (#1e1e1e) background without competing with
# species or highlight colours.
CELL_BOUNDARY_COLOUR  = (0.85, 0.85, 0.85)
CELL_BOUNDARY_OPACITY = 0.55
CELL_BOUNDARY_WIDTH   = 1.5


class StructureView(QWidget):
    """
    Embeddable 3D structure viewer backed by PyVista/VTK.
    """

    # Emitted when any species colour changes — listeners (e.g. pFC scatter
    # plot) can connect to this to stay in sync with the colour picker.
    colours_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.supercell        = None
        self._highlight_pair  = None   # single pair (i, j) for individual mode
        self._highlight_pairs = []     # list of (i, j) for shell mode
        self._highlight_pairs_gold = False  # if True, shell atoms also get gold spheres
        # Cell-aware chain highlight (see highlight_chain_with_cells()):
        # explicit Cartesian positions/species, not atom indices, so a chain
        # spanning a periodic-cell translation can be drawn as it actually
        # is instead of folded into the base cell by minimum-image.
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()  # base-cell atoms coinciding with a chain position
        self._show_cell_boundary   = True
        # bond_pairs: list of (i_1based, j_1based, sp_i, sp_j)
        self._bond_pairs      = []
        self._colours        = {}
        self._display_frac   = None
        # enabled bond types: set of frozenset({sp_i, sp_j})
        self._enabled_bond_types  = set()
        self._bond_checkboxes     = {}   # frozenset -> QCheckBox
        self._bond_factors        = {}   # frozenset -> float (per-pair stretch factor)
        self._expanded_bond_types = set()  # which rows are expanded in the UI
        # Reference site state (None = not set)
        self._refsite_frac        = None
        self._refsite_fracs       = []     # all positions for multi-site
        self._refsite_bonds_cutoff = None  # Å, or None to hide
        self._n_refsite_cubes     = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1e1e1e')
        self.plotter.add_axes(interactive=False)
        outer.addWidget(self.plotter, stretch=1)

        right_panel = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setFixedWidth(155)
        right_widget.setLayout(right_panel)

        # Colour preset switcher
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel('Preset:'))
        self._preset_combo = QComboBox()
        for name in COLOUR_PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self._preset_combo)
        right_panel.addLayout(preset_row)

        self._colour_group = self._build_colour_panel()
        self._bond_group   = self._build_bond_toggle_panel()
        right_panel.addWidget(self._colour_group)
        right_panel.addWidget(self._bond_group)

        # Projection toggle
        self._proj_btn = QPushButton('Parallel projection')
        self._proj_btn.setCheckable(True)
        self._proj_btn.setChecked(False)
        self._proj_btn.clicked.connect(self._toggle_projection)
        right_panel.addWidget(self._proj_btn)

        self._cell_boundary_chk = QCheckBox('Show cell boundary')
        self._cell_boundary_chk.setChecked(self._show_cell_boundary)
        self._cell_boundary_chk.setToolTip(
            'Outline the unit cell (the lattice vectors from the origin), '
            'useful context when atoms outside it are shown for a '
            'periodic-cell-translated interaction.')
        self._cell_boundary_chk.stateChanged.connect(self._toggle_cell_boundary)
        right_panel.addWidget(self._cell_boundary_chk)

        right_panel.addStretch()
        outer.addWidget(right_widget)

    def _toggle_cell_boundary(self, state):
        self._show_cell_boundary = bool(state)
        self._redraw()

    def _build_colour_panel(self):
        group = QGroupBox('Atom colours')
        self._colour_layout = QVBoxLayout()
        self._colour_layout.setAlignment(Qt.AlignTop)
        group.setLayout(self._colour_layout)
        scroll = QScrollArea()
        scroll.setWidget(group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        return scroll

    def _build_bond_toggle_panel(self):
        group = QGroupBox('Bonds')
        self._bond_layout = QVBoxLayout()
        self._bond_layout.setAlignment(Qt.AlignTop)
        group.setLayout(self._bond_layout)
        return group

    def _rebuild_colour_buttons(self):
        while self._colour_layout.count():
            item = self._colour_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for sp in sorted(self._colours.keys()):
            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(sp)
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            self._update_swatch(btn, self._colours[sp])
            btn.clicked.connect(
                lambda checked, s=sp, b=btn: self._pick_colour(s, b)
            )
            rl.addWidget(lbl)
            rl.addWidget(btn)
            self._colour_layout.addWidget(row)

    def _rebuild_bond_toggles(self):
        while self._bond_layout.count():
            item = self._bond_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._bond_checkboxes = {}

        bond_types = sorted(
            set(frozenset({sp_i, sp_j})
                for _, _, sp_i, sp_j in self._bond_pairs),
            key=lambda s: '-'.join(sorted(s))
        )
        for bt in bond_types:
            label      = '-'.join(sorted(bt))
            is_open    = bt in self._expanded_bond_types

            container  = QWidget()
            cl         = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)

            # Top row: checkbox + expand toggle
            top_row = QWidget()
            tl      = QHBoxLayout(top_row)
            tl.setContentsMargins(2, 1, 2, 1)
            cb = QCheckBox(label)
            cb.setChecked(bt in self._enabled_bond_types)
            cb.stateChanged.connect(
                lambda state, b=bt: self._on_bond_toggle(b, state)
            )
            expand_btn = QPushButton('▼' if is_open else '▶')
            expand_btn.setFixedSize(18, 18)
            expand_btn.setFlat(True)
            expand_btn.setStyleSheet('font-size: 8px; padding: 0px;')
            tl.addWidget(cb)
            tl.addStretch()
            tl.addWidget(expand_btn)

            # Factor row (hidden unless this bond type was expanded)
            factor_row = QWidget()
            fl         = QHBoxLayout(factor_row)
            fl.setContentsMargins(16, 0, 2, 3)
            fl.addWidget(QLabel('factor:'))
            spin = QDoubleSpinBox()
            spin.setRange(0.8, 2.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(self._bond_factors.get(bt, BOND_FACTOR))
            spin.setFixedWidth(60)
            fl.addWidget(spin)
            fl.addStretch()
            factor_row.setVisible(is_open)

            expand_btn.clicked.connect(
                lambda _checked, b=bt, btn=expand_btn, fr=factor_row:
                self._toggle_bond_expand(b, btn, fr)
            )
            spin.valueChanged.connect(
                lambda val, b=bt: self._on_bond_factor_change(b, val)
            )

            cl.addWidget(top_row)
            cl.addWidget(factor_row)
            self._bond_layout.addWidget(container)
            self._bond_checkboxes[bt] = cb

    def _toggle_bond_expand(self, bond_type, btn, factor_row):
        if bond_type in self._expanded_bond_types:
            self._expanded_bond_types.discard(bond_type)
            btn.setText('▶')
            factor_row.setVisible(False)
        else:
            self._expanded_bond_types.add(bond_type)
            btn.setText('▼')
            factor_row.setVisible(True)

    def _on_bond_factor_change(self, bond_type, value):
        if self.supercell is None:
            return
        self._bond_factors[bond_type] = value
        self._bond_pairs = self._compute_bonds(self.supercell)
        # Sync enabled set: preserve user's choices, add any newly-appearing
        # types with the same default rule used at load time.
        all_types = {frozenset({sp_i, sp_j})
                     for _, _, sp_i, sp_j in self._bond_pairs}
        for bt in all_types:
            if bt not in self._enabled_bond_types:
                if not (len(bt) == 1 and next(iter(bt)) in SAME_SPECIES_METALS):
                    self._enabled_bond_types.add(bt)
        self._enabled_bond_types &= all_types
        self._rebuild_bond_toggles()
        self._redraw()

    def _on_bond_toggle(self, bond_type, state):
        if state == Qt.Checked:
            self._enabled_bond_types.add(bond_type)
        else:
            self._enabled_bond_types.discard(bond_type)
        self._redraw()

    def _update_swatch(self, btn, rgb_float):
        r, g, b = [int(c * 255) for c in rgb_float]
        btn.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 1px solid #888;'
        )

    def _pick_colour(self, species, btn):
        current = self._colours.get(species, (0.5, 0.5, 0.5))
        r, g, b = [int(c * 255) for c in current]
        colour  = QColorDialog.getColor(
            QColor(r, g, b), self, f'{species} colour'
        )
        if colour.isValid():
            new_rgb = (colour.redF(), colour.greenF(), colour.blueF())
            self._colours[species] = new_rgb
            self._update_swatch(btn, new_rgb)
            self._redraw()
            self.colours_changed.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_supercell(self, supercell):
        """Render all atoms and compute bonds. Call once per structure load."""
        self.supercell            = supercell
        self._highlight_pair      = None
        self._highlight_pairs     = []
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()
        self._display_frac        = supercell.positions.copy()
        self._refsite_frac        = None
        self._refsite_fracs       = []
        self._refsite_bonds_cutoff = None
        self._n_refsite_cubes     = 0
        self._colours             = {
            sp: element_colour(sp) for sp in supercell.chem_symbols
        }
        self._bond_factors        = {}
        self._expanded_bond_types = set()

        # Compute bond connectivity (indices + species)
        self._bond_pairs = self._compute_bonds(supercell)

        # Default: all bond types enabled except same-species metal pairs
        # (user can override via checkboxes)
        all_types = set(
            frozenset({sp_i, sp_j})
            for _, _, sp_i, sp_j in self._bond_pairs
        )
        self._enabled_bond_types = {
            bt for bt in all_types
            if not (len(bt) == 1 and next(iter(bt)) in SAME_SPECIES_METALS)
        }

        self._rebuild_colour_buttons()
        self._rebuild_bond_toggles()
        self._redraw()
        self.plotter.reset_camera()
        self.plotter.render()

    def highlight_bond(self, atom1_idx_1based, atom2_idx_1based):
        """Select a pair: center atom1, dim background, highlight bond."""
        self._highlight_pair       = (atom1_idx_1based, atom2_idx_1based)
        self._highlight_pairs      = []
        self._highlight_pairs_gold = False
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()
        if self._refsite_bonds_cutoff is None:
            self._update_display_frac(atom1_idx_1based)
        self._redraw()

    def highlight_bonds(self, pairs, center_on=None, highlight_atoms=False):
        """
        Highlight a set of bonds (shell mode).

        Parameters
        ----------
        pairs           : list of (atom1_idx_1based, atom2_idx_1based)
        center_on       : int or None — if given, center the display on this
                          1-based atom index (same convention as highlight_bond).
        highlight_atoms : bool — if True, atoms at bond endpoints also receive
                          gold spheres (same as single-pair mode).  Use this for
                          multicenter chains where every atom in the chain should
                          be visually prominent.  Default False preserves the
                          original shell-view behaviour (full species color).
        """
        self._highlight_pairs      = list(pairs)
        self._highlight_pair       = None
        self._highlight_pairs_gold = highlight_atoms
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()
        if self.supercell is not None:
            if center_on is not None:
                self._update_display_frac(center_on)
            else:
                self._display_frac = self.supercell.positions.copy()
        self._redraw()

    def highlight_chain_with_cells(self, entries, highlight_atoms=True):
        """
        Highlight a chain given as (atom_idx_1based, cell_offset) pairs,
        drawing the real, unwrapped geometry instead of folding every atom
        to its nearest image (as highlight_bonds() does).

        This matters whenever the same atom index appears more than once in
        a chain at different periodic-cell translations (e.g. a LOBSTER
        cobiBetween chain crossing a cell boundary) — highlight_bonds() has
        no way to tell such a repeat apart from revisiting the atom at cell
        [0,0,0], so it collapses onto the same position. Here, each entry's
        true position (base fractional position + its own integer cell
        offset, unwrapped) is used directly, so a translated atom renders
        as a distinct "ghost" point outside the displayed unit cell — see
        also the "Show cell boundary" toggle, useful context for judging
        how far outside the box these points sit.

        Parameters
        ----------
        entries         : list of (atom_idx_1based, cell_offset), where
                          cell_offset is an (dx, dy, dz) integer tuple.
        highlight_atoms : bool, kept for API symmetry with highlight_bonds();
                          this method always highlights the chain atoms since
                          drawing the ghost positions without them would
                          defeat its purpose. Default True.
        """
        self._highlight_pair       = None
        self._highlight_pairs      = []
        self._highlight_pairs_gold = False
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()

        if self.supercell is None or not entries:
            self._redraw()
            return

        sc = self.supercell
        true_fracs = [sc.positions[idx - 1] + np.asarray(cell, dtype=float)
                     for idx, cell in entries]
        ref = true_fracs[0]
        self._display_frac = self._center_display_on_frac(ref)

        self._chain_cart_positions = [
            (0.5 + (f - ref)) @ sc.lattice for f in true_fracs
        ]
        self._chain_species = [sc.species(idx) for idx, _ in entries]
        self._chain_zero_cell_idxs = {
            idx for idx, cell in entries if tuple(int(c) for c in cell) == (0, 0, 0)
        }
        self._redraw()
        self.plotter.reset_camera()

    def clear_highlight(self):
        """Remove highlight and restore original view."""
        self._highlight_pair       = None
        self._highlight_pairs      = []
        self._highlight_pairs_gold = False
        self._chain_cart_positions = []
        self._chain_species        = []
        self._chain_zero_cell_idxs = set()
        if self.supercell is not None:
            self._display_frac = self.supercell.positions.copy()
        self._redraw()

    def set_ref_site(self, frac_coords):
        """Place or move the reference site cube marker."""
        self.set_ref_sites([frac_coords])

    def set_ref_sites(self, frac_coords_list):
        """Place cube markers for one or more reference sites."""
        self._remove_refsite_cubes(render=False)
        if not frac_coords_list:
            self._refsite_fracs = []
            return
        self._refsite_fracs = [np.asarray(f, dtype=float) for f in frac_coords_list]
        self._refsite_frac  = self._refsite_fracs[0]
        if self.supercell is None:
            return
        if self._refsite_bonds_cutoff is not None:
            self._center_display_on_refsite()
            self._redraw()
            return
        s = REFSITE_CUBE_SIZE
        for i, frac in enumerate(self._refsite_fracs):
            cart = frac @ self.supercell.lattice
            cube = pv.Cube(center=cart, x_length=s, y_length=s, z_length=s)
            is_last = (i == len(self._refsite_fracs) - 1)
            self.plotter.add_mesh(
                cube, color=REFSITE_COLOUR, opacity=REFSITE_CUBE_OPACITY,
                name=f'refsite_cube_{i}', render=is_last,
            )
        self._n_refsite_cubes = len(self._refsite_fracs)

    def _remove_refsite_cubes(self, render=True):
        for i in range(getattr(self, '_n_refsite_cubes', 1)):
            self.plotter.remove_actor(f'refsite_cube_{i}', render=False)
        # Legacy name used before multi-site support
        self.plotter.remove_actor('refsite_cube', render=render)
        self._n_refsite_cubes = 0

    def set_refsite_bonds(self, cutoff):
        """
        Draw tubes from the refsite to all atoms within cutoff Å.
        Centers the display on the refsite (same convention as pair highlight).
        Pass None to clear bonds and restore the original atom positions.
        """
        self._refsite_bonds_cutoff = cutoff
        if self.supercell is None:
            return
        if cutoff is not None:
            self._center_display_on_refsite()
        else:
            self._display_frac = self.supercell.positions.copy()
        self._redraw()

    def _toggle_projection(self, checked):
        """Switch between perspective and parallel projection."""
        if checked:
            self.plotter.enable_parallel_projection()
            self._proj_btn.setText('Perspective projection')
        else:
            self.plotter.disable_parallel_projection()
            self._proj_btn.setText('Parallel projection')
        self.plotter.render()

    def _apply_preset(self, preset_name):
        """Reset all species colours from the named preset and redraw."""
        if self.supercell is None:
            return
        palette = COLOUR_PRESETS.get(preset_name, COLOUR_PRESETS['Jmol'])
        self._colours = {
            sp: palette.get(sp, (0.50, 0.50, 0.50))
            for sp in self.supercell.chem_symbols
        }
        self._rebuild_colour_buttons()
        self._redraw()
        self.colours_changed.emit()

    def get_species_colours(self):
        """
        Return current per-species colour dict as {species: (R,G,B) float tuple}.
        Called by PFCViewerWidget to sync scatter plot colours with structure view.
        """
        return dict(self._colours)

    def pair_colours_hex(self, sp1, sp2):
        """
        Return (hex1, hex2) for a species pair.
        Same species → (colour, colour).
        Mixed → (colour_sp1, colour_sp2) for split-circle rendering.
        """
        def to_hex(rgb):
            if isinstance(rgb, str):
                return rgb
            r, g, b = [int(c * 255) for c in rgb]
            return f'#{r:02x}{g:02x}{b:02x}'
        c1 = to_hex(self._colours.get(sp1, (0.5, 0.5, 0.5)))
        c2 = to_hex(self._colours.get(sp2, (0.5, 0.5, 0.5)))
        return c1, c2

    # ------------------------------------------------------------------
    # Bond computation (once on load)
    # ------------------------------------------------------------------

    def _compute_bonds(self, supercell):
        """
        Find all bonded pairs using per-species-pair bond factors.
        Returns list of (i_1based, j_1based, species_i, species_j).
        """
        sc         = supercell
        lat        = sc.lattice
        cart_pos   = sc.positions @ lat
        N          = sc.n_atoms
        max_r      = max(covalent_radius(sp) for sp in sc.chem_symbols)
        max_factor = max(self._bond_factors.values(), default=BOND_FACTOR)
        q_radius   = 2 * max_r * max_factor * 1.05

        # Expand to all 27 cells (original + 26 periodic images) so that bonds
        # crossing the supercell boundary are found.  all_cart[k*N + i] is image
        # k of atom i; idx % N maps any expanded index back to the original atom.
        all_cart = np.vstack(
            [cart_pos + np.array([di, dj, dk], dtype=float) @ lat
             for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)]
        )
        tree = cKDTree(all_cart)

        seen  = set()
        bonds = []
        for i in range(N):
            sp_i   = sc.species(i + 1)
            r_i    = covalent_radius(sp_i)
            for idx in tree.query_ball_point(cart_pos[i], q_radius):
                j = idx % N
                if j == i:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                sp_j   = sc.species(j + 1)
                bt     = frozenset({sp_i, sp_j})
                factor = self._bond_factors.get(bt, BOND_FACTOR)
                d      = sc.atom_distance(i + 1, j + 1)
                if d <= (r_i + covalent_radius(sp_j)) * factor:
                    seen.add(pair)
                    i1, j1 = pair
                    bonds.append((i1 + 1, j1 + 1,
                                  sc.species(i1 + 1), sc.species(j1 + 1)))
        return bonds

    # ------------------------------------------------------------------
    # Display coordinate management
    # ------------------------------------------------------------------

    def _center_display_on_frac(self, ref_frac):
        """
        Shift all atoms' fractional coords so that ref_frac (any fractional
        point — not necessarily an atom's own position) sits at
        (0.5, 0.5, 0.5), using the minimum-image convention. Returns the new
        (N, 3) display-fractional array; does not mutate self._display_frac.
        """
        sc      = self.supercell
        display = np.empty_like(sc.positions)
        for k in range(sc.n_atoms):
            diff       = sc.positions[k] - ref_frac
            diff      -= np.floor(diff + 0.5)
            display[k] = 0.5 + diff
        return display

    def _update_display_frac(self, center_idx_1based):
        """
        Shift all fractional coords so center_idx is at (0.5, 0.5, 0.5)
        using the minimum-image convention.
        Bond endpoints are recomputed from these coords in _redraw.
        """
        self._display_frac = self._center_display_on_frac(
            self.supercell.positions[center_idx_1based - 1])

    def _center_display_on_refsite(self):
        """
        Same as _update_display_frac but centered on _refsite_frac instead of
        an atom. After this call, the refsite sits at (0.5, 0.5, 0.5) and all
        atoms are at their nearest periodic images around it.
        """
        self._display_frac = self._center_display_on_frac(self._refsite_frac)

    # ------------------------------------------------------------------
    # Scene drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        """Full scene redraw from current display state."""
        if self.supercell is None:
            return

        self.plotter.clear_actors()
        sc            = self.supercell
        highlight     = self._highlight_pair
        selected_idxs = set(highlight) if highlight else set()

        # Atoms involved in the current shell highlight (both endpoints)
        shell_atom_idxs = set()
        for i_1, j_1 in self._highlight_pairs:
            shell_atom_idxs.add(i_1)
            shell_atom_idxs.add(j_1)

        # Pre-compute refsite neighbours once — used for both dimming and
        # bond drawing so atoms_within is only called once per redraw.
        refsite_nearby: list = []
        if (self._refsite_frac is not None
                and self._refsite_bonds_cutoff is not None):
            refsite_nearby = sc.atoms_within(self._refsite_frac,
                                             self._refsite_bonds_cutoff)
        active_refsite_idxs = {idx for idx, _ in refsite_nearby}

        # Choose bond background opacity depending on highlight mode.
        # Chain-gold mode (highlight_atoms=True) uses the same deep dim as
        # single-pair mode so chain atoms pop out clearly against the background.
        chain_gold_active  = shell_atom_idxs and self._highlight_pairs_gold
        chain_cells_active = bool(self._chain_cart_positions)
        if selected_idxs or active_refsite_idxs or chain_gold_active or chain_cells_active:
            bg_bond_opacity = DIM_OPACITY
        elif shell_atom_idxs:
            bg_bond_opacity = SHELL_DIM_OPACITY
        else:
            bg_bond_opacity = BOND_OPACITY

        # Cartesian positions from current display fractional coords
        cart = self._display_frac @ sc.lattice

        # --- Background bonds (batched, per-bond minimum-image endpoints) ---
        active_bonds = [
            (i_1, j_1) for i_1, j_1, sp_i, sp_j in self._bond_pairs
            if frozenset({sp_i, sp_j}) in self._enabled_bond_types
        ]
        if active_bonds:
            bond_points, bond_lines, pt_idx = [], [], 0
            for i_1, j_1 in active_bonds:
                p1     = cart[i_1 - 1]
                frac_i = self._display_frac[i_1 - 1]
                frac_j = self._display_frac[j_1 - 1]
                diff   = frac_j - frac_i
                diff  -= np.floor(diff + 0.5)
                p2     = p1 + diff @ sc.lattice
                bond_points.extend([p1, p2])
                bond_lines.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2
                # If the bond crosses the cell boundary, also draw a matching
                # stub from j's side so neither atom looks stranded.
                p2_frac = frac_i + diff
                if np.any(p2_frac < -1e-6) or np.any(p2_frac > 1.0 + 1e-6):
                    p2_j = cart[j_1 - 1]
                    bond_points.extend([p2_j, p2_j - diff @ sc.lattice])
                    bond_lines.extend([2, pt_idx, pt_idx + 1])
                    pt_idx += 2
            bond_mesh        = pv.PolyData()
            bond_mesh.points = np.array(bond_points)
            bond_mesh.lines  = np.array(bond_lines)
            bond_tubed       = bond_mesh.tube(radius=BOND_RADIUS, n_sides=6)
            self.plotter.add_mesh(
                bond_tubed,
                color=BOND_COLOUR,
                name='all_bonds',
                opacity=bg_bond_opacity,
                render=False,
            )

        # --- Atoms — split into full-opacity and dim groups ---
        # Individual bond mode : selected_idxs drawn separately as gold;
        #                        all others dimmed at DIM_OPACITY.
        # Shell mode           : shell_atom_idxs at full species colour;
        #                        all others dimmed at SHELL_DIM_OPACITY.
        # Refsite-bonds mode   : atoms within cutoff at full opacity;
        #                        all others dimmed at DIM_OPACITY.
        # Default              : all atoms full opacity.
        full_groups      = {}   # sp -> [pos, ...]  full opacity, species colour
        dim_groups       = {}   # sp -> [pos, ...]  DIM_OPACITY
        shell_dim_groups = {}   # sp -> [pos, ...]  SHELL_DIM_OPACITY

        for i in range(sc.n_atoms):
            idx = i + 1
            if idx in selected_idxs:
                continue   # drawn separately as gold below
            # When highlight_atoms is set, chain atoms are also drawn as gold
            if shell_atom_idxs and self._highlight_pairs_gold and idx in shell_atom_idxs:
                continue   # drawn separately as gold below
            if idx in self._chain_zero_cell_idxs:
                continue   # drawn separately as a chain-cell ghost sphere below
            sp  = sc.species(idx)
            pos = cart[i]
            if active_refsite_idxs:
                if idx in active_refsite_idxs:
                    full_groups.setdefault(sp, []).append(pos)
                else:
                    dim_groups.setdefault(sp, []).append(pos)
            elif selected_idxs or chain_gold_active or chain_cells_active:
                # Single-pair or chain-gold: background atoms strongly dimmed
                dim_groups.setdefault(sp, []).append(pos)
            elif shell_atom_idxs:
                if idx in shell_atom_idxs:
                    full_groups.setdefault(sp, []).append(pos)
                else:
                    shell_dim_groups.setdefault(sp, []).append(pos)
            else:
                full_groups.setdefault(sp, []).append(pos)

        for sp, positions in full_groups.items():
            colour = self._colours.get(sp, (0.5, 0.5, 0.5))
            radius = display_radius(sp)
            cloud  = pv.PolyData(np.array(positions))
            proto  = pv.Sphere(radius=radius,
                               theta_resolution=8, phi_resolution=8)
            glyphs = cloud.glyph(geom=proto, scale=False, orient=False)
            self.plotter.add_mesh(
                glyphs, color=colour,
                name=f'atoms_{sp}_full',
                opacity=FULL_OPACITY, render=False,
            )

        for sp, positions in dim_groups.items():
            colour = self._colours.get(sp, (0.5, 0.5, 0.5))
            radius = display_radius(sp)
            cloud  = pv.PolyData(np.array(positions))
            proto  = pv.Sphere(radius=radius,
                               theta_resolution=8, phi_resolution=8)
            glyphs = cloud.glyph(geom=proto, scale=False, orient=False)
            self.plotter.add_mesh(
                glyphs, color=colour,
                name=f'atoms_{sp}_dim',
                opacity=DIM_OPACITY, render=False,
            )

        for sp, positions in shell_dim_groups.items():
            colour = self._colours.get(sp, (0.5, 0.5, 0.5))
            radius = display_radius(sp)
            cloud  = pv.PolyData(np.array(positions))
            proto  = pv.Sphere(radius=radius,
                               theta_resolution=8, phi_resolution=8)
            glyphs = cloud.glyph(geom=proto, scale=False, orient=False)
            self.plotter.add_mesh(
                glyphs, color=colour,
                name=f'atoms_{sp}_shell_dim',
                opacity=SHELL_DIM_OPACITY, render=False,
            )

        # --- Selected atoms (gold, larger) — pair-highlight mode only ---
        for idx_1 in selected_idxs:
            sp     = sc.species(idx_1)
            r      = display_radius(sp) * HIGHLIGHT_ATOM_FACTOR
            sphere = pv.Sphere(radius=r, center=cart[idx_1 - 1],
                               theta_resolution=18, phi_resolution=18)
            self.plotter.add_mesh(
                sphere, color=HIGHLIGHT_COLOUR,
                name=f'selected_{idx_1}',
                opacity=FULL_OPACITY, render=False,
            )

        # --- Chain atoms (gold, larger) — shell mode with highlight_atoms=True ---
        if self._highlight_pairs_gold:
            for idx_1 in shell_atom_idxs:
                sp     = sc.species(idx_1)
                r      = display_radius(sp) * HIGHLIGHT_ATOM_FACTOR
                sphere = pv.Sphere(radius=r, center=cart[idx_1 - 1],
                                   theta_resolution=18, phi_resolution=18)
                self.plotter.add_mesh(
                    sphere, color=HIGHLIGHT_COLOUR,
                    name=f'chain_atom_{idx_1}',
                    opacity=FULL_OPACITY, render=False,
                )

        # --- Highlighted bond (individual mode) ---
        if highlight:
            i_1, j_1 = highlight
            p1      = cart[i_1 - 1]
            frac_i  = self._display_frac[i_1 - 1]
            frac_j  = self._display_frac[j_1 - 1]
            diff    = frac_j - frac_i
            diff   -= np.floor(diff + 0.5)
            p2      = p1 + diff @ sc.lattice
            tube = pv.Line(p1, p2).tube(
                radius=HIGHLIGHT_BOND_RADIUS, n_sides=16
            )
            self.plotter.add_mesh(
                tube, color=HIGHLIGHT_COLOUR,
                name='highlight_bond',
                opacity=FULL_OPACITY, render=False,
            )
            p2_frac = frac_i + diff
            if np.any(p2_frac < -1e-6) or np.any(p2_frac > 1.0 + 1e-6):
                p2_j  = cart[j_1 - 1]
                tube2 = pv.Line(p2_j, p2_j - diff @ sc.lattice).tube(
                    radius=HIGHLIGHT_BOND_RADIUS, n_sides=16
                )
                self.plotter.add_mesh(
                    tube2, color=HIGHLIGHT_COLOUR,
                    name='highlight_bond_wrap',
                    opacity=FULL_OPACITY, render=False,
                )

        # --- Multi-bond highlight (shell mode) ---
        # Each bond needs its own tube actor — pv.PolyData with multiple
        # disconnected 2-point line cells only tubes the first segment.
        for k, (i_1, j_1) in enumerate(self._highlight_pairs):
            if i_1 == j_1:
                # Self-referential pair: not a real bond, just a request to
                # include this atom in shell_atom_idxs (e.g. a chain that
                # revisits the same atom index via a periodic-cell
                # translation this widget has no other way to represent).
                # pv.Line(p, p) is a zero-length, zero-point mesh, which
                # add_mesh() rejects outright — nothing to draw here anyway.
                continue
            p1     = cart[i_1 - 1]
            frac_i = self._display_frac[i_1 - 1]
            frac_j = self._display_frac[j_1 - 1]
            diff   = frac_j - frac_i
            diff  -= np.floor(diff + 0.5)
            p2     = p1 + diff @ sc.lattice
            tube   = pv.Line(p1, p2).tube(radius=HIGHLIGHT_BOND_RADIUS, n_sides=12)
            self.plotter.add_mesh(
                tube, color=HIGHLIGHT_COLOUR,
                name=f'highlight_bond_multi_{k}',
                opacity=FULL_OPACITY, render=False,
            )
            p2_frac = frac_i + diff
            if np.any(p2_frac < -1e-6) or np.any(p2_frac > 1.0 + 1e-6):
                p2_j  = cart[j_1 - 1]
                tube2 = pv.Line(p2_j, p2_j - diff @ sc.lattice).tube(
                    radius=HIGHLIGHT_BOND_RADIUS, n_sides=12
                )
                self.plotter.add_mesh(
                    tube2, color=HIGHLIGHT_COLOUR,
                    name=f'highlight_bond_multi_{k}_wrap',
                    opacity=FULL_OPACITY, render=False,
                )

        # --- Refsite cube markers ---
        fracs_to_draw = self._refsite_fracs if self._refsite_fracs else (
            [self._refsite_frac] if self._refsite_frac is not None else []
        )
        s = REFSITE_CUBE_SIZE
        for i, frac in enumerate(fracs_to_draw):
            if self._refsite_bonds_cutoff is not None:
                cart_ref = np.array([0.5, 0.5, 0.5]) @ sc.lattice
            else:
                cart_ref = frac @ sc.lattice
            cube = pv.Cube(center=cart_ref, x_length=s, y_length=s, z_length=s)
            self.plotter.add_mesh(
                cube, color=REFSITE_COLOUR, opacity=REFSITE_CUBE_OPACITY,
                name=f'refsite_cube_{i}', render=False,
            )

            # --- Refsite bonds (use precomputed nearby list) ---
            if refsite_nearby:
                bond_points, bond_lines, pt_idx = [], [], 0
                for atom_idx, _ in refsite_nearby:
                    bond_points.extend([cart_ref, cart[atom_idx - 1]])
                    bond_lines.extend([2, pt_idx, pt_idx + 1])
                    pt_idx += 2
                ref_bond_mesh        = pv.PolyData()
                ref_bond_mesh.points = np.array(bond_points)
                ref_bond_mesh.lines  = np.array(bond_lines)
                ref_bond_tubed = ref_bond_mesh.tube(
                    radius=REFSITE_BOND_RADIUS, n_sides=8
                )
                self.plotter.add_mesh(
                    ref_bond_tubed,
                    color=REFSITE_BOND_COLOUR, opacity=REFSITE_BOND_OPACITY,
                    name='refsite_bonds', render=False,
                )

        # --- Cell-aware chain highlight (ghost atoms + real bond geometry) ---
        # Positions here are already fully resolved (base position + integer
        # cell offset, unwrapped) by highlight_chain_with_cells() — drawn
        # directly, with no minimum-image folding, so a chain that spans a
        # periodic-cell translation shows atoms genuinely outside the
        # displayed unit cell rather than folded back into it.
        if self._chain_cart_positions:
            for k, (pos, sp) in enumerate(
                    zip(self._chain_cart_positions, self._chain_species)):
                r      = display_radius(sp) * HIGHLIGHT_ATOM_FACTOR
                sphere = pv.Sphere(radius=r, center=pos,
                                   theta_resolution=18, phi_resolution=18)
                self.plotter.add_mesh(
                    sphere, color=HIGHLIGHT_COLOUR,
                    name=f'chain_cell_atom_{k}',
                    opacity=FULL_OPACITY, render=False,
                )
            for k in range(len(self._chain_cart_positions) - 1):
                p1, p2 = self._chain_cart_positions[k], self._chain_cart_positions[k + 1]
                if np.linalg.norm(p2 - p1) < 1e-6:
                    continue   # degenerate (identical) consecutive positions
                tube = pv.Line(p1, p2).tube(radius=HIGHLIGHT_BOND_RADIUS, n_sides=12)
                self.plotter.add_mesh(
                    tube, color=HIGHLIGHT_COLOUR,
                    name=f'chain_cell_bond_{k}',
                    opacity=FULL_OPACITY, render=False,
                )

        # --- Unit cell boundary (light wireframe box, lattice vectors from origin) ---
        if self._show_cell_boundary:
            a, b, c = sc.lattice[0], sc.lattice[1], sc.lattice[2]
            corners = np.array([
                [0., 0., 0.], a, b, c, a + b, a + c, b + c, a + b + c,
            ])
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7),
            ]
            box_points, box_lines, pt_idx = [], [], 0
            for i, j in edges:
                box_points.extend([corners[i], corners[j]])
                box_lines.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2
            box_mesh        = pv.PolyData()
            box_mesh.points = np.array(box_points)
            box_mesh.lines  = np.array(box_lines)
            self.plotter.add_mesh(
                box_mesh, color=CELL_BOUNDARY_COLOUR,
                opacity=CELL_BOUNDARY_OPACITY, line_width=CELL_BOUNDARY_WIDTH,
                name='cell_boundary', render=False,
            )

        self.plotter.render()
