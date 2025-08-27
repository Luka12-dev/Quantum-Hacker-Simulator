import sys
import math
import cmath
import random
import json
import itertools
from typing import List, Tuple, Dict, Optional

import numpy as np

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTextEdit,
    QGridLayout,
    QComboBox,
    QMessageBox,
    QDialog,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize


# ------------------------------- Styling -----------------------------------
# Two theme stylesheets (light and dark). Keep them visually pleasing and modern.

STYLE_LIGHT = """
QWidget { background: #f6f8fb; color: #1b1f23; font-family: 'Segoe UI', Roboto, Arial; }
QLabel#title { font-weight: 700; font-size: 16px; }
QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #e6eefc); 
             border: 1px solid #c6d2ee; border-radius: 8px; padding: 6px; }
QPushButton:hover { background: #eaf2ff; }
QListWidget { background: #ffffff; border: 1px solid #dfe7f7; border-radius: 8px; }
QTextEdit { background: #ffffff; border: 1px solid #dfe7f7; border-radius: 8px; }
QProgressBar { border-radius: 8px; background: #eaeef6; height: 12px; }
QProgressBar::chunk { border-radius: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4b8bf5, stop:1 #6ac1ff); }
QComboBox, QSpinBox { background: #ffffff; border: 1px solid #dfe7f7; border-radius: 6px; padding: 3px; }
QToolTip { background: #1b1f23; color: white; }
"""

STYLE_DARK = """
QWidget { background: #0f1115; color: #e6eef6; font-family: 'Segoe UI', Roboto, Arial; }
QLabel#title { font-weight: 700; font-size: 16px; }
QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #14161a, stop:1 #1c2026); 
             border: 1px solid #2b2f36; border-radius: 8px; padding: 6px; color: #dbe9ff }
QPushButton:hover { background: #22252b; }
QListWidget { background: #0b0d10; border: 1px solid #2b2f36; border-radius: 8px; }
QTextEdit { background: #0b0d10; border: 1px solid #2b2f36; border-radius: 8px; }
QProgressBar { border-radius: 8px; background: #101214; height: 12px; }
QProgressBar::chunk { border-radius: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #8ab4ff, stop:1 #4b8bf5); }
QComboBox, QSpinBox { background: #0b0d10; border: 1px solid #2b2f36; border-radius: 6px; padding: 3px; color: #e6eef6 }
QToolTip { background: #e6eef6; color: #0f1115; }
"""


# ----------------------------- Quantum Core --------------------------------


def kron_n(matrices: List[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.array([[1.0]])
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)


def cnot_matrix(num_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 2 ** num_qubits
    M = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (num_qubits - 1 - b)) & 1 for b in range(num_qubits)]
        if bits[control] == 1:
            bits[target] = 1 - bits[target]
        j = 0
        for b in bits:
            j = (j << 1) | b
        M[j, i] = 1
    return M


def single_qubit_operator_on_n(num_qubits: int, qubit_idx: int, gate: np.ndarray) -> np.ndarray:
    mats = []
    for i in range(num_qubits):
        if i == qubit_idx:
            mats.append(gate)
        else:
            mats.append(I)
    return kron_n(mats)


class QuantumState:
    def __init__(self, num_qubits: int):
        self.n = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0

    def set_state(self, amplitudes: np.ndarray):
        assert amplitudes.shape == self.state.shape
        self.state = amplitudes.astype(complex)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm == 0:
            return
        self.state = self.state / norm

    def apply_operator(self, op: np.ndarray):
        assert op.shape == (2 ** self.n, 2 ** self.n)
        self.state = op.dot(self.state)
        self.normalize()

    def apply_single_qubit_gate(self, qubit_idx: int, gate: np.ndarray):
        op = single_qubit_operator_on_n(self.n, qubit_idx, gate)
        self.apply_operator(op)

    def apply_cnot(self, control: int, target: int):
        op = cnot_matrix(self.n, control, target)
        self.apply_operator(op)

    def probabilities(self) -> np.ndarray:
        probs = np.abs(self.state) ** 2
        probs = np.real_if_close(probs)
        return probs

    def amplitudes(self) -> np.ndarray:
        return self.state.copy()

    def copy(self) -> "QuantumState":
        s = QuantumState(self.n)
        s.set_state(self.state.copy())
        return s


# ----------------------------- Puzzle / Level -------------------------------

class GateSpec:
    def __init__(self, name: str, target: int, control: Optional[int] = None):
        self.name = name
        self.target = target
        self.control = control

    def __repr__(self):
        if self.control is None:
            return f"{self.name}(q{self.target})"
        else:
            return f"{self.name}(q{self.control}->{self.target})"


class Level:
    def __init__(self, num_qubits: int, init_state: np.ndarray, target_state: np.ndarray,
                 max_moves: int, desc: str = ""):
        self.n = num_qubits
        self.init = init_state
        self.target = target_state
        self.max_moves = max_moves
        self.desc = desc

    def is_solved(self, state: QuantumState, tol: float = 1e-6) -> bool:
        a = np.vdot(self.target, state.amplitudes())
        fidelity = abs(a) ** 2
        return fidelity >= (1 - tol)


AVAILABLE_GATES = ["X", "H", "Z", "S", "T", "CNOT"]


def apply_gate_by_spec(qstate: QuantumState, spec: GateSpec):
    if spec.name == "X":
        qstate.apply_single_qubit_gate(spec.target, X)
    elif spec.name == "H":
        qstate.apply_single_qubit_gate(spec.target, H)
    elif spec.name == "Z":
        qstate.apply_single_qubit_gate(spec.target, Z)
    elif spec.name == "S":
        qstate.apply_single_qubit_gate(spec.target, S)
    elif spec.name == "T":
        qstate.apply_single_qubit_gate(spec.target, T)
    elif spec.name == "CNOT":
        assert spec.control is not None
        qstate.apply_cnot(spec.control, spec.target)
    else:
        raise ValueError(f"Unknown gate {spec}")


# ------------------------------- Hint Solver --------------------------------

from collections import deque


def state_close(a: np.ndarray, b: np.ndarray, tol=1e-6) -> bool:
    return np.linalg.norm(a - b) <= tol


def find_short_solution(level: Level, max_depth: int = 4) -> Optional[List[GateSpec]]:
    n = level.n
    init = QuantumState(n)
    init.set_state(level.init)
    target = level.target

    specs: List[GateSpec] = []
    for q in range(n):
        for name in ["X", "H", "Z", "S", "T"]:
            specs.append(GateSpec(name, q))
    for c in range(n):
        for t in range(n):
            if t == c:
                continue
            specs.append(GateSpec("CNOT", t, control=c))

    start_vec = init.amplitudes()
    start_key = tuple(np.round(start_vec.real, 6).tolist() + np.round(start_vec.imag, 6).tolist())

    q = deque()
    q.append((start_vec, []))
    seen = {start_key}

    while q:
        vec, seq = q.popleft()
        if state_close(vec, target, tol=1e-6):
            return seq
        if len(seq) >= max_depth:
            continue
        for s in specs:
            qs = QuantumState(n)
            qs.set_state(vec.copy())
            apply_gate_by_spec(qs, s)
            new_vec = qs.amplitudes()
            key = tuple(np.round(new_vec.real, 6).tolist() + np.round(new_vec.imag, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            q.append((new_vec, seq + [s]))
    return None


# ----------------------------- UI Components --------------------------------

class StateDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.title = QLabel("State")
        self.title.setObjectName('title')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        layout.addLayout(self.grid)

        self.setLayout(layout)
        self.prog_bars: List[QProgressBar] = []
        self.amp_labels: List[QLabel] = []

    def build_for_n(self, num_qubits: int):
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.prog_bars = []
        self.amp_labels = []
        dim = 2 ** num_qubits
        cols = 2
        rows = math.ceil(dim / cols)
        for i in range(dim):
            label = QLabel(f"|{i:0{num_qubits}b}>: 0.00")
            pb = QProgressBar()
            pb.setRange(0, 1000)
            pb.setValue(0)
            pb.setTextVisible(False)
            self.grid.addWidget(label, i // cols * 2, i % cols)
            self.grid.addWidget(pb, i // cols * 2 + 1, i % cols)
            self.amp_labels.append(label)
            self.prog_bars.append(pb)

    def update_state(self, amplitudes: np.ndarray):
        probs = np.abs(amplitudes) ** 2
        dim = len(probs)
        for i in range(dim):
            amp = amplitudes[i]
            prob = float(probs[i])
            if i < len(self.amp_labels):
                amp_text = f"|{i:0{int(math.log2(len(amplitudes)))}b}>: {amp.real:+.3f}{amp.imag:+.3f}j |p={prob*100:.2f}%"
                self.amp_labels[i].setText(amp_text)
                if i < len(self.prog_bars):
                    self.prog_bars[i].setValue(int(prob * 1000))


class GatePalette(QWidget):
    gate_selected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.title = QLabel("Gate Palette")
        self.title.setObjectName('title')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.buttons: Dict[str, QPushButton] = {}

        for name in ["X", "H", "Z", "S", "T"]:
            b = QPushButton(name)
            b.clicked.connect(lambda checked, n=name: self.emit_gate(n))
            layout.addWidget(b)
            self.buttons[name] = b

        self.cnot_label = QLabel("CNOT: control -> target")
        layout.addWidget(self.cnot_label)
        self.cnot_controls = QHBoxLayout()
        self.control_combo = QComboBox()
        self.target_combo = QComboBox()
        self.cnot_add = QPushButton("Add CNOT")
        self.cnot_add.clicked.connect(self.add_cnot)
        self.cnot_controls.addWidget(self.control_combo)
        self.cnot_controls.addWidget(self.target_combo)
        self.cnot_controls.addWidget(self.cnot_add)
        layout.addLayout(self.cnot_controls)

        self.setLayout(layout)

    def setup_qubits(self, n: int):
        self.control_combo.clear()
        self.target_combo.clear()
        for i in range(n):
            self.control_combo.addItem(str(i))
            self.target_combo.addItem(str(i))

    def emit_gate(self, name: str):
        self.gate_selected.emit(name)

    def add_cnot(self):
        c = int(self.control_combo.currentText())
        t = int(self.target_combo.currentText())
        if c == t:
            QMessageBox.warning(self, "CNOT error", "Control and target must differ")
            return
        self.gate_selected.emit(f"CNOT {c} {t}")


class CircuitList(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.title = QLabel("Circuit")
        self.title.setObjectName('title')
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.list = QListWidget()
        layout.addWidget(self.list)
        btns = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        self.export_btn = QPushButton("Export")
        btns.addWidget(self.undo_btn)
        btns.addWidget(self.redo_btn)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.save_btn)
        btns.addWidget(self.load_btn)
        btns.addWidget(self.export_btn)
        layout.addLayout(btns)
        self.setLayout(layout)

    def add_gate(self, spec: GateSpec):
        item = QListWidgetItem(repr(spec))
        item.setData(Qt.ItemDataRole.UserRole, spec)
        self.list.addItem(item)

    def pop_last(self) -> Optional[GateSpec]:
        if self.list.count() == 0:
            return None
        item = self.list.takeItem(self.list.count() - 1)
        spec = item.data(Qt.ItemDataRole.UserRole)
        return spec

    def clear(self):
        self.list.clear()

    def to_specs(self) -> List[GateSpec]:
        specs = []
        for i in range(self.list.count()):
            specs.append(self.list.item(i).data(Qt.ItemDataRole.UserRole))
        return specs

    def save_to_file(self, filename: str):
        specs = self.to_specs()
        data = []
        for s in specs:
            data.append({'name': s.name, 'target': s.target, 'control': s.control})
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.clear()
        for d in data:
            spec = GateSpec(d['name'], d['target'], control=d.get('control'))
            self.add_gate(spec)

    def export_to_clipboard(self):
        specs = self.to_specs()
        stext = " -> ".join(repr(x) for x in specs)
        QApplication.clipboard().setText(stext)
        return stext


# ----------------------------- Settings Dialog ------------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings if settings is not None else {}
        layout = QVBoxLayout()

        # Resolution group
        res_group = QGroupBox("Window & Resolution")
        f = QFormLayout()
        self.res_combo = QComboBox()
        self.res_combo.addItem("800 x 600")
        self.res_combo.addItem("1100 x 700")
        self.res_combo.addItem("1280 x 800")
        self.res_combo.addItem("Fullscreen")
        f.addRow("Resolution:", self.res_combo)
        res_group.setLayout(f)
        layout.addWidget(res_group)

        # Hint settings
        hint_group = QGroupBox("Hints")
        hf = QFormLayout()
        self.hint_depth = QSpinBox()
        self.hint_depth.setRange(1, 8)
        self.hint_depth.setValue(self.settings.get('hint_depth', 4))
        self.autohint_checkbox = QCheckBox("Show next-step hint on level start")
        self.autohint_checkbox.setChecked(self.settings.get('auto_hint', False))
        hf.addRow("Hint search depth:", self.hint_depth)
        hf.addRow(self.autohint_checkbox)
        hint_group.setLayout(hf)
        layout.addWidget(hint_group)

        # Theme choice
        theme_group = QGroupBox("Theme")
        tf = QFormLayout()
        self.dark_mode = QCheckBox("Dark Mode")
        self.dark_mode.setChecked(self.settings.get('dark_mode', True))
        tf.addRow(self.dark_mode)
        theme_group.setLayout(tf)
        layout.addWidget(theme_group)

        btns = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

        self.apply_btn.clicked.connect(self.on_apply)
        self.cancel_btn.clicked.connect(self.reject)

    def on_apply(self):
        sel = self.res_combo.currentText()
        if sel == "Fullscreen":
            res = "fullscreen"
        else:
            w, h = sel.split(' x ')
            res = (int(w.strip()), int(h.strip()))
        self.settings['resolution'] = res
        self.settings['hint_depth'] = int(self.hint_depth.value())
        self.settings['auto_hint'] = bool(self.autohint_checkbox.isChecked())
        self.settings['dark_mode'] = bool(self.dark_mode.isChecked())
        self.accept()


# ----------------------------- Main Window ---------------------------------

class QuantumHackerMain(QWidget):
    def __init__(self, start_fullscreen: bool = False):
        super().__init__()
        self.setWindowTitle("Quantum Hacker Simulator")
        self.setWindowIcon(QIcon("iconfile55.ico"))

        # Default settings
        self.settings = {
            'resolution': (1100, 700),
            'hint_depth': 4,
            'auto_hint': False,
            'dark_mode': True,
        }

        self.apply_theme()
        self.resize(*self.settings['resolution'])

        main_layout = QHBoxLayout()

        # Left column
        left = QVBoxLayout()
        self.gate_palette = GatePalette()
        left.addWidget(self.gate_palette)
        self.circuit = CircuitList()
        left.addWidget(self.circuit)
        main_layout.addLayout(left, 2)

        # Center
        center = QVBoxLayout()
        self.state_display = StateDisplay()
        center.addWidget(self.state_display)

        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run Circuit")
        self.step_btn = QPushButton("Step")
        self.reset_btn = QPushButton("Reset")
        self.hint_btn = QPushButton("Hint")
        self.randomize_btn = QPushButton("Randomize Level")
        controls.addWidget(self.run_btn)
        controls.addWidget(self.step_btn)
        controls.addWidget(self.reset_btn)
        controls.addWidget(self.hint_btn)
        controls.addWidget(self.randomize_btn)
        center.addLayout(controls)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        center.addWidget(self.info, 1)
        main_layout.addLayout(center, 3)

        # Right
        right = QVBoxLayout()
        self.target_display = StateDisplay()
        right.addWidget(self.target_display)

        lvl_controls = QVBoxLayout()
        self.level_combo = QComboBox()
        self.level_combo.addItem("Tutorial: Single qubit H -> |1>")
        self.level_combo.addItem("Bell pair: Create entanglement")
        self.level_combo.addItem("Superposition race")
        self.level_combo.addItem("Custom random 2-qubit")
        lvl_controls.addWidget(QLabel("Select Level"))
        lvl_controls.addWidget(self.level_combo)
        self.start_level_btn = QPushButton("Start Level")
        lvl_controls.addWidget(self.start_level_btn)
        right.addLayout(lvl_controls)

        # Extra controls: fullscreen, settings
        extra = QHBoxLayout()
        self.fullscreen_btn = QPushButton("Toggle Fullscreen")
        self.settings_btn = QPushButton("Settings")
        self.theme_btn = QPushButton("Toggle Theme")
        extra.addWidget(self.fullscreen_btn)
        extra.addWidget(self.settings_btn)
        extra.addWidget(self.theme_btn)
        right.addLayout(extra)

        self.feedback = QLabel("")
        self.feedback.setWordWrap(True)
        right.addWidget(self.feedback)

        main_layout.addLayout(right, 2)
        self.setLayout(main_layout)

        # Game state
        self.current_level: Optional[Level] = None
        self.qubit_count = 1
        self.qstate: Optional[QuantumState] = None
        self.undo_stack: List[GateSpec] = []
        self.redo_stack: List[GateSpec] = []
        self.last_run_sequence: List[GateSpec] = []

        # Connect
        self.gate_palette.gate_selected.connect(self.on_gate_selected)
        self.circuit.undo_btn.clicked.connect(self.on_undo)
        self.circuit.redo_btn.clicked.connect(self.on_redo)
        self.circuit.clear_btn.clicked.connect(self.on_clear)
        self.circuit.save_btn.clicked.connect(self.on_save_circuit)
        self.circuit.load_btn.clicked.connect(self.on_load_circuit)
        self.circuit.export_btn.clicked.connect(self.on_export_circuit)
        self.run_btn.clicked.connect(self.on_run)
        self.step_btn.clicked.connect(self.on_step)
        self.reset_btn.clicked.connect(self.on_reset)
        self.hint_btn.clicked.connect(self.on_hint)
        self.randomize_btn.clicked.connect(self.on_randomize_level)
        self.start_level_btn.clicked.connect(self.on_start_level)
        self.level_combo.currentIndexChanged.connect(self.on_level_change)
        self.fullscreen_btn.clicked.connect(self.on_toggle_fullscreen)
        self.settings_btn.clicked.connect(self.on_settings)
        self.theme_btn.clicked.connect(self.on_toggle_theme)

        self.populate_levels()
        self.on_level_change(0)

        self.step_timer = QTimer()
        self.step_timer.setInterval(380)
        self.step_timer.timeout.connect(self._perform_step)
        self.step_index = 0

        self.info.setHtml(
            "<b>Welcome to Quantum Hacker Simulator!</b><br>"
            "Polished UI, themes, fullscreen and improved hints are enabled."
        )
        if start_fullscreen:
            self.showFullScreen()
        else:
            self.show()

    # ------------------------ Theme & settings -------------------------------

    def apply_theme(self):
        if self.settings.get('dark_mode', True):
            QApplication.instance().setStyleSheet(STYLE_DARK)
        else:
            QApplication.instance().setStyleSheet(STYLE_LIGHT)

    def on_toggle_theme(self):
        self.settings['dark_mode'] = not self.settings.get('dark_mode', True)
        self.apply_theme()

    def on_settings(self):
        dlg = SettingsDialog(self, settings=self.settings.copy())
        # Preselect current resolution
        res = self.settings.get('resolution')
        if res == 'fullscreen':
            dlg.res_combo.setCurrentText('Fullscreen')
        else:
            dlg.res_combo.setCurrentText(f"{res[0]} x {res[1]}")
        dlg.hint_depth.setValue(self.settings.get('hint_depth', 4))
        dlg.autohint_checkbox.setChecked(self.settings.get('auto_hint', False))
        dlg.dark_mode.setChecked(self.settings.get('dark_mode', True))
        if dlg.exec():
            self.settings.update(dlg.settings)
            # Apply resolution
            res = self.settings['resolution']
            if res == 'fullscreen':
                self.showFullScreen()
            else:
                if self.isFullScreen():
                    self.showNormal()
                self.resize(*res)
            # Apply theme
            self.apply_theme()
            self.info.append("Settings applied.")

    # ------------------------ Level generation -------------------------------

    def populate_levels(self):
        self.levels: List[Level] = []
        n = 1
        init = np.zeros(2 ** n, dtype=complex)
        init[0] = 1
        target = np.zeros_like(init)
        target[1] = 1
        self.levels.append(Level(n, init, target, max_moves=2, desc="Flip the qubit to |1>."))

        n = 2
        init2 = np.zeros(2 ** n, dtype=complex)
        init2[0] = 1
        target2 = np.zeros_like(init2)
        target2[0] = 1 / math.sqrt(2)
        target2[3] = 1 / math.sqrt(2)
        self.levels.append(Level(n, init2, target2, max_moves=3, desc="Create a Bell pair."))

        n = 2
        init3 = np.zeros(2 ** n, dtype=complex)
        init3[0] = 1
        target3 = np.ones_like(init3, dtype=complex) / math.sqrt(4)
        self.levels.append(Level(n, init3, target3, max_moves=3, desc="Equal superposition."))

        self.levels.append(self.generate_random_level(2, max_moves=4))

    def generate_random_level(self, num_qubits: int, max_moves: int = 4) -> Level:
        qs = QuantumState(num_qubits)
        seq = []
        for _ in range(random.randint(1, max_moves)):
            gate_type = random.choice(["X", "H", "Z", "S", "T", "CNOT"])
            if gate_type == "CNOT" and num_qubits >= 2:
                c = random.randrange(num_qubits)
                t = random.randrange(num_qubits)
                while t == c:
                    t = random.randrange(num_qubits)
                spec = GateSpec("CNOT", t, control=c)
                apply_gate_by_spec(qs, spec)
                seq.append(spec)
            else:
                q = random.randrange(num_qubits)
                spec = GateSpec(gate_type, q)
                apply_gate_by_spec(qs, spec)
                seq.append(spec)
        desc = "Random challenge generated by applying a short random sequence. Reverse it!"
        return Level(num_qubits, QuantumState(num_qubits).amplitudes(), qs.amplitudes(), max_moves=max_moves, desc=desc)

    # ------------------------ UI event handlers ------------------------------

    def on_level_change(self, idx: int):
        self.info.append("Level preview updated.")

    def on_start_level(self):
        idx = self.level_combo.currentIndex()
        if idx < 0 or idx >= len(self.levels):
            return
        self.current_level = self.levels[idx]
        self.qubit_count = self.current_level.n
        self.qstate = QuantumState(self.qubit_count)
        self.qstate.set_state(self.current_level.init.copy())
        self.circuit.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.last_run_sequence.clear()
        self.state_display.build_for_n(self.qubit_count)
        self.target_display.build_for_n(self.qubit_count)
        self.state_display.update_state(self.qstate.amplitudes())
        self.target_display.update_state(self.current_level.target)
        self.gate_palette.setup_qubits(self.qubit_count)
        self.feedback.setText(f"Level started: {self.current_level.desc}")
        # Auto-hint if enabled
        if self.settings.get('auto_hint', False):
            self.on_hint(next_only=True)

    def on_gate_selected(self, s: str):
        if s.startswith("CNOT"):
            parts = s.split()
            _, c, t = parts
            spec = GateSpec("CNOT", int(t), control=int(c))
        else:
            name = s
            q, ok = self._choose_qubit_dialog(self.qubit_count, f"Place {name} on which qubit?")
            if not ok:
                return
            spec = GateSpec(name, q)
        self.circuit.add_gate(spec)
        self.undo_stack.append(spec)
        self.redo_stack.clear()

    def _choose_qubit_dialog(self, n: int, text: str) -> Tuple[int, bool]:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Choose Qubit")
        dialog.setText(text)
        combo = QComboBox(dialog)
        for i in range(n):
            combo.addItem(str(i))
        dialog.layout().addWidget(combo, 1, 1)
        dialog.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        res = dialog.exec()
        if dialog.clickedButton() and dialog.clickedButton().text() == "OK":
            return int(combo.currentText()), True
        return 0, False

    def on_undo(self):
        spec = self.circuit.pop_last()
        if spec is None:
            return
        self.redo_stack.append(spec)
        self._recompute_state_from_circuit()

    def on_redo(self):
        if not self.redo_stack:
            return
        spec = self.redo_stack.pop()
        self.circuit.add_gate(spec)
        self._recompute_state_from_circuit()

    def on_clear(self):
        self.circuit.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._recompute_state_from_circuit()

    def _recompute_state_from_circuit(self):
        if self.current_level is None:
            return
        self.qstate = QuantumState(self.qubit_count)
        self.qstate.set_state(self.current_level.init.copy())
        for spec in self.circuit.to_specs():
            apply_gate_by_spec(self.qstate, spec)
        self.state_display.update_state(self.qstate.amplitudes())
        solved = self.current_level.is_solved(self.qstate)
        if solved:
            self.feedback.setText("Solved! Absolute hacker flex.")
        else:
            self.feedback.setText("")

    def on_save_circuit(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save circuit", filter="JSON Files (*.json)")
        if not fn:
            return
        try:
            self.circuit.save_to_file(fn)
            self.info.append(f"Circuit saved to {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def on_load_circuit(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load circuit", filter="JSON Files (*.json)")
        if not fn:
            return
        try:
            self.circuit.load_from_file(fn)
            self._recompute_state_from_circuit()
            self.info.append(f"Circuit loaded from {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def on_export_circuit(self):
        s = self.circuit.export_to_clipboard()
        self.info.append(f"Exported: {s}")

    def on_run(self):
        specs = self.circuit.to_specs()
        if not specs:
            QMessageBox.information(self, "No gates", "Add some gates first.")
            return
        self.last_run_sequence = specs
        self.step_index = 0
        self.qstate = QuantumState(self.qubit_count)
        self.qstate.set_state(self.current_level.init.copy())
        self.state_display.update_state(self.qstate.amplitudes())
        self.step_timer.start()

    def _perform_step(self):
        if self.step_index >= len(self.last_run_sequence):
            self.step_timer.stop()
            solved = self.current_level.is_solved(self.qstate)
            if solved:
                self.feedback.setText("Level solved during run — pogodi me sada.")
            else:
                self.feedback.setText("Run finished. Keep trying.")
            return
        spec = self.last_run_sequence[self.step_index]
        apply_gate_by_spec(self.qstate, spec)
        self.state_display.update_state(self.qstate.amplitudes())
        self.step_index += 1

    def on_step(self):
        specs = self.circuit.to_specs()
        if not specs:
            QMessageBox.information(self, "No gates", "Add gates first.")
            return
        # Apply the next gate by replaying applied count
        applied = 0
        tmp = QuantumState(self.qubit_count)
        tmp.set_state(self.current_level.init.copy())
        # find how many gates already applied by comparing state vectors
        # fallback: just apply and remove last for UX simplicity
        spec = self.circuit.pop_last()
        if spec is None:
            return
        self.circuit.add_gate(spec)
        apply_gate_by_spec(self.qstate, spec)
        self.state_display.update_state(self.qstate.amplitudes())
        if self.current_level.is_solved(self.qstate):
            self.feedback.setText("Solved by stepping — legend.")

    def on_reset(self):
        if self.current_level is None:
            return
        self.qstate = QuantumState(self.qubit_count)
        self.qstate.set_state(self.current_level.init.copy())
        self.state_display.update_state(self.qstate.amplitudes())
        self.feedback.setText("State reset to initial.")

    def on_hint(self, next_only: bool = False):
        if self.current_level is None:
            return
        self.feedback.setText("Searching for hint...")
        QApplication.processEvents()
        depth = int(self.settings.get('hint_depth', 4))
        sol = find_short_solution(self.current_level, max_depth=depth)
        if sol is None:
            self.feedback.setText(f"No solution found up to depth {depth}.")
            return
        # Full sequence hint
        seq_text = " -> ".join(repr(x) for x in sol)
        if next_only:
            next_gate = sol[0]
            self.feedback.setText(f"Next-step hint: {repr(next_gate)}")
            # show small explanation
            self.info.append(self._explain_gate(next_gate))
        else:
            self.feedback.setText(f"Hint (depth {len(sol)}): {seq_text}")
            self.info.append("Explanation: " + ", ".join(self._explain_gate(s) for s in sol))

    def _explain_gate(self, spec: GateSpec) -> str:
        if spec.name == 'H':
            return f"Hadamard on q{spec.target}: creates superposition of |0> and |1>."
        if spec.name == 'X':
            return f"Pauli-X on q{spec.target}: flips |0> <-> |1>."
        if spec.name == 'Z':
            return f"Pauli-Z on q{spec.target}: phase flip (|1> gets -1)."
        if spec.name == 'S':
            return f"Phase S on q{spec.target}: adds i phase to |1>."
        if spec.name == 'T':
            return f"T gate on q{spec.target}: pi/4 phase to |1>."
        if spec.name == 'CNOT':
            return f"CNOT control q{spec.control} -> target q{spec.target}: entangling flip if control is 1."
        return f"Gate {spec}"

    def on_randomize_level(self):
        lvl = self.generate_random_level(2, max_moves=4)
        self.levels.append(lvl)
        self.level_combo.addItem("Random gen: new")
        self.level_combo.setCurrentIndex(self.level_combo.count() - 1)
        self.on_start_level()

    def on_toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.info.append("Exited fullscreen.")
        else:
            self.showFullScreen()
            self.info.append("Entered fullscreen.")


# ------------------------------- Utilities ----------------------------------

def pretty_print_state(amplitudes: np.ndarray) -> str:
    lines = []
    n = int(math.log2(len(amplitudes)))
    for i, a in enumerate(amplitudes):
        lines.append(f"|{i:0{n}b}>: {a.real:+.3f}{a.imag:+.3f}j (p={abs(a)**2:.3f})")
    return "\n".join(lines)


# --------------------------------- Main ------------------------------------

def main():
    app = QApplication(sys.argv)
    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    window = QuantumHackerMain(start_fullscreen=True)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()