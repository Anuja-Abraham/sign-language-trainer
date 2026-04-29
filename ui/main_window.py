from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.content import CONTENT_SETS, DYNAMIC_SIGNS, SIGN_GUIDE_TEXT, ensure_local_reference_assets
from core.dynamic_signs import DynamicMatcher
from core.recognizer import DetectionResult, Recognizer
from core.session import Mode, SessionController
from core.settings import AppSettings, SettingsStore


class MainWindow(QMainWindow):
    def __init__(self, app_root: Path) -> None:
        super().__init__()
        self.setWindowTitle("Sign Language Trainer (Python Desktop)")
        self.resize(1400, 900)

        self.app_root = app_root
        self.settings_store = SettingsStore(app_root / "config" / "settings.json")
        self.settings = self.settings_store.load()
        self.reference_map = ensure_local_reference_assets(app_root)

        self.session = SessionController()
        self.recognizer = Recognizer()
        self.dynamic_matcher = DynamicMatcher()

        self.cap: Optional[cv2.VideoCapture] = None
        self.hand_landmarker = self._init_hand_landmarker()
        self.hand_connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        )

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._on_camera_tick)
        self.timer_tick = QTimer(self)
        self.timer_tick.timeout.connect(self._on_second_tick)

        self.calibrating = False
        self.calibration_samples: List[float] = []
        self.calibration_seconds_left = 10

        self._build_ui()
        self._apply_settings()
        self._refresh_full_ui("Click Start Camera")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_camera()
        self._persist_settings()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        super().closeEvent(event)

    def _init_hand_landmarker(self):
        model_path = self.app_root / "assets" / "models" / "hand_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path,
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _draw_landmarks(self, frame: cv2.typing.MatLike, landmarks: List[Tuple[float, float, float]]) -> None:
        h, w = frame.shape[:2]
        for a, b in self.hand_connections:
            xa, ya = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            xb, yb = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, (xa, ya), (xb, yb), (34, 197, 94), 2)
        for x, y, _ in landmarks:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (37, 99, 235), -1)

    def _reference_pixmap(self, path: str, width: int, height: int) -> QPixmap:
        pix = QPixmap(path)
        if not pix.isNull():
            return pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return QPixmap()
        if len(image.shape) == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy()).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        self.video_label = QLabel("Camera preview")
        self.video_label.setMinimumHeight(500)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111827;color:#e5e7eb;border-radius:8px;")
        outer.addWidget(self.video_label, stretch=3)
        self.reference_strip = QHBoxLayout()
        self.ref_prev = QLabel()
        self.ref_curr = QLabel()
        self.ref_next = QLabel()
        self.ref_prev_text = QLabel()
        self.ref_curr_text = QLabel()
        self.ref_next_text = QLabel()
        for label in (self.ref_prev, self.ref_curr, self.ref_next):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(160, 140)
            label.setStyleSheet("background:#0f172a;border-radius:6px;")
        for text_label in (self.ref_prev_text, self.ref_curr_text, self.ref_next_text):
            text_label.setAlignment(Qt.AlignCenter)
        for image, text in (
            (self.ref_prev, self.ref_prev_text),
            (self.ref_curr, self.ref_curr_text),
            (self.ref_next, self.ref_next_text),
        ):
            col = QVBoxLayout()
            col.addWidget(image)
            col.addWidget(text)
            holder = QWidget()
            holder.setLayout(col)
            self.reference_strip.addWidget(holder)
        outer.addLayout(self.reference_strip, stretch=1)

        controls_wrap = QHBoxLayout()
        outer.addLayout(controls_wrap, stretch=2)

        actions_box = QGroupBox("Session Actions")
        actions_layout = QGridLayout(actions_box)
        self.start_camera_btn = QPushButton("Start Camera")
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.start_calibration_btn = QPushButton("Start Calibration")
        self.prev_sign_btn = QPushButton("Previous Sign")
        self.next_sign_btn = QPushButton("Next Sign")
        actions_layout.addWidget(self.start_camera_btn, 0, 0)
        actions_layout.addWidget(self.stop_camera_btn, 0, 1)
        actions_layout.addWidget(self.start_calibration_btn, 1, 0)
        actions_layout.addWidget(self.prev_sign_btn, 1, 1)
        actions_layout.addWidget(self.next_sign_btn, 2, 0, 1, 2)
        controls_wrap.addWidget(actions_box)

        mode_box = QGroupBox("Modes")
        mode_layout = QGridLayout(mode_box)
        self.learn_btn = QPushButton("Learn")
        self.quiz_btn = QPushButton("Quiz (45s)")
        self.challenge_btn = QPushButton("Challenge")
        self.drill_ae_btn = QPushButton("Drill A/E")
        self.drill_bd_btn = QPushButton("Drill B/D")
        mode_layout.addWidget(self.learn_btn, 0, 0)
        mode_layout.addWidget(self.quiz_btn, 0, 1)
        mode_layout.addWidget(self.challenge_btn, 1, 0)
        mode_layout.addWidget(self.drill_ae_btn, 1, 1)
        mode_layout.addWidget(self.drill_bd_btn, 2, 0)
        controls_wrap.addWidget(mode_box)

        setup_box = QGroupBox("Setup")
        setup_layout = QGridLayout(setup_box)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["beginner", "intermediate", "advanced"])
        self.content_combo = QComboBox()
        self.content_combo.addItems(["asl-ae", "asl-az", "numbers"])
        self.challenge_duration_combo = QComboBox()
        self.challenge_duration_combo.addItems(["30", "60"])
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.dual_view_check = QCheckBox("Dual View")
        self.instructor_check = QCheckBox("Instructor Mode")
        setup_layout.addWidget(QLabel("Difficulty"), 0, 0)
        setup_layout.addWidget(self.difficulty_combo, 0, 1)
        setup_layout.addWidget(QLabel("Content"), 1, 0)
        setup_layout.addWidget(self.content_combo, 1, 1)
        setup_layout.addWidget(QLabel("Challenge s"), 2, 0)
        setup_layout.addWidget(self.challenge_duration_combo, 2, 1)
        setup_layout.addWidget(QLabel("Theme"), 3, 0)
        setup_layout.addWidget(self.theme_combo, 3, 1)
        setup_layout.addWidget(self.dual_view_check, 4, 0)
        setup_layout.addWidget(self.instructor_check, 4, 1)
        controls_wrap.addWidget(setup_box)

        info_box = QGroupBox("Live Stats")
        info_layout = QGridLayout(info_box)
        self.mode_label = QLabel("-")
        self.timer_label = QLabel("--")
        self.calibration_label = QLabel("Not calibrated")
        self.target_label = QLabel("-")
        self.detected_label = QLabel("-")
        self.accuracy_label = QLabel("0%")
        self.status_label = QLabel("Ready")
        self.hint_label = QLabel("-")
        self.coach_label = QLabel("-")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.ref_label = QLabel()
        self.ref_label.setAlignment(Qt.AlignCenter)
        self.ref_label.setMinimumHeight(180)
        info_layout.addWidget(QLabel("Mode"), 0, 0)
        info_layout.addWidget(self.mode_label, 0, 1)
        info_layout.addWidget(QLabel("Timer"), 1, 0)
        info_layout.addWidget(self.timer_label, 1, 1)
        info_layout.addWidget(QLabel("Calibration"), 2, 0)
        info_layout.addWidget(self.calibration_label, 2, 1)
        info_layout.addWidget(QLabel("Target"), 3, 0)
        info_layout.addWidget(self.target_label, 3, 1)
        info_layout.addWidget(QLabel("Detected"), 4, 0)
        info_layout.addWidget(self.detected_label, 4, 1)
        info_layout.addWidget(QLabel("Accuracy"), 5, 0)
        info_layout.addWidget(self.accuracy_label, 5, 1)
        info_layout.addWidget(QLabel("Status"), 6, 0)
        info_layout.addWidget(self.status_label, 6, 1)
        info_layout.addWidget(QLabel("Hint"), 7, 0)
        info_layout.addWidget(self.hint_label, 7, 1)
        info_layout.addWidget(QLabel("Coach"), 8, 0)
        info_layout.addWidget(self.coach_label, 8, 1)
        info_layout.addWidget(self.progress, 9, 0, 1, 2)
        info_layout.addWidget(self.ref_label, 10, 0, 1, 2)
        controls_wrap.addWidget(info_box)

        self.start_camera_btn.clicked.connect(self._start_camera)
        self.stop_camera_btn.clicked.connect(self._stop_camera)
        self.start_calibration_btn.clicked.connect(self._start_calibration)
        self.prev_sign_btn.clicked.connect(self._prev_sign)
        self.next_sign_btn.clicked.connect(self._next_sign)
        self.learn_btn.clicked.connect(self._set_learn)
        self.quiz_btn.clicked.connect(self._start_quiz)
        self.challenge_btn.clicked.connect(self._start_challenge)
        self.drill_ae_btn.clicked.connect(self._start_drill_ae)
        self.drill_bd_btn.clicked.connect(self._start_drill_bd)
        self.difficulty_combo.currentTextChanged.connect(self._change_difficulty)
        self.content_combo.currentTextChanged.connect(self._change_content)
        self.challenge_duration_combo.currentTextChanged.connect(self._change_challenge_duration)
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        self.dual_view_check.stateChanged.connect(self._toggle_dual_view)
        self.instructor_check.stateChanged.connect(self._toggle_instructor_mode)

    def _apply_settings(self) -> None:
        self.difficulty_combo.setCurrentText(self.settings.difficulty)
        self.content_combo.setCurrentText(self.settings.content_set)
        self.challenge_duration_combo.setCurrentText(str(self.settings.challenge_duration))
        self.theme_combo.setCurrentText(self.settings.theme)
        self.dual_view_check.setChecked(self.settings.dual_view)
        self.instructor_check.setChecked(self.settings.instructor_mode)
        self.session.set_content_set(self.settings.content_set)
        self.session.set_difficulty(self.settings.difficulty)
        self.session.state.challenge_duration = self.settings.challenge_duration
        self.session.state.dual_view = self.settings.dual_view
        self.session.state.instructor_mode = self.settings.instructor_mode
        self._apply_theme(self.settings.theme)

    def _persist_settings(self) -> None:
        settings = AppSettings(
            difficulty=self.difficulty_combo.currentText(),
            content_set=self.content_combo.currentText(),
            theme=self.theme_combo.currentText(),
            challenge_duration=int(self.challenge_duration_combo.currentText()),
            dual_view=self.dual_view_check.isChecked(),
            instructor_mode=self.instructor_check.isChecked(),
        )
        self.settings_store.save(settings)

    def _start_camera(self) -> None:
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Camera unavailable.")
            self.cap = None
            return
        self.camera_timer.start(30)
        self.timer_tick.start(1000)
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.status_label.setText("Camera running. Match the target sign.")

    def _stop_camera(self) -> None:
        self.camera_timer.stop()
        self.timer_tick.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.status_label.setText("Camera stopped. Click Start Camera.")
        self.detected_label.setText("-")
        self.accuracy_label.setText("0%")
        self.progress.setValue(0)
        self.dynamic_matcher.reset()
        self.recognizer.reset()

    def _start_calibration(self) -> None:
        if self.cap is None:
            self.status_label.setText("Start camera first.")
            return
        self.calibrating = True
        self.calibration_seconds_left = 10
        self.calibration_samples.clear()
        self.calibration_label.setText("Calibrating... 10s")
        self.status_label.setText("Hold hand centered for calibration.")

    def _next_sign(self) -> None:
        self.session.set_target(self.session.state.target_index + 1)
        self._refresh_target()
        self.status_label.setText(f"Now practice {self.session.state.target}.")

    def _prev_sign(self) -> None:
        self.session.set_target(self.session.state.target_index - 1)
        self._refresh_target()
        self.status_label.setText(f"Now practice {self.session.state.target}.")

    def _set_learn(self) -> None:
        self.session.set_mode(Mode.LEARN)
        self.session.state.timed_remaining = 0
        self._refresh_mode_labels("Learn mode active")

    def _start_quiz(self) -> None:
        if self.cap is None:
            self.status_label.setText("Start camera first.")
            return
        self.session.start_timed_mode(Mode.QUIZ, 45)
        self._refresh_mode_labels("QUIZ started.")

    def _start_challenge(self) -> None:
        if self.cap is None:
            self.status_label.setText("Start camera first.")
            return
        seconds = int(self.challenge_duration_combo.currentText())
        self.session.start_timed_mode(Mode.CHALLENGE, seconds)
        self._refresh_mode_labels("CHALLENGE started.")

    def _start_drill_ae(self) -> None:
        if self.cap is None:
            self.status_label.setText("Start camera first.")
            return
        self.session.start_timed_mode(Mode.DRILL_AE, 60, ["A", "E"])
        self._refresh_mode_labels("DRILL AE started.")

    def _start_drill_bd(self) -> None:
        if self.cap is None:
            self.status_label.setText("Start camera first.")
            return
        self.session.start_timed_mode(Mode.DRILL_BD, 60, ["B", "D"])
        self._refresh_mode_labels("DRILL BD started.")

    def _change_difficulty(self, value: str) -> None:
        self.session.set_difficulty(value)
        self.status_label.setText(f"Difficulty: {value}")

    def _change_content(self, key: str) -> None:
        self.session.set_content_set(key)
        self._refresh_target()
        self.status_label.setText(f"Loaded content: {key}")

    def _change_challenge_duration(self, value: str) -> None:
        self.session.state.challenge_duration = int(value)

    def _apply_theme(self, theme: str) -> None:
        if theme == "light":
            self.setStyleSheet("QWidget { background: #f8fafc; color: #0f172a; }")
        else:
            self.setStyleSheet("QWidget { background: #0b1220; color: #e2e8f0; }")

    def _toggle_dual_view(self, state: int) -> None:
        self.session.state.dual_view = state == Qt.Checked
        self.ref_label.setVisible(self.session.state.dual_view)

    def _toggle_instructor_mode(self, state: int) -> None:
        self.session.state.instructor_mode = state == Qt.Checked
        self.next_sign_btn.setEnabled(not self.session.state.instructor_mode and self.session.state.mode == Mode.LEARN)

    def _on_second_tick(self) -> None:
        if self.session.state.timed_remaining > 0:
            completed = self.session.tick_timer()
            if completed:
                self.status_label.setText("Timed mode complete.")
        if self.calibrating:
            self.calibration_seconds_left -= 1
            self.calibration_label.setText(f"Calibrating... {max(0, self.calibration_seconds_left)}s")
            if self.calibration_seconds_left <= 0:
                avg = sum(self.calibration_samples) / max(len(self.calibration_samples), 1)
                self.session.set_calibration_scale(avg / 0.12 if avg else 1.0)
                self.calibrating = False
                self.calibration_label.setText("Calibrated")
                self.status_label.setText("Calibration complete.")
        self._refresh_mode_labels()

    def _on_camera_tick(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)
        display = frame.copy()

        if results.hand_landmarks:
            landmarks_proto = results.hand_landmarks[0]
            landmarks: List[Tuple[float, float, float]] = [(lm.x, lm.y, lm.z) for lm in landmarks_proto]
            self._draw_landmarks(display, landmarks)
            lm_map = {i: (landmarks[i][0], landmarks[i][1]) for i in range(len(landmarks))}
            self.dynamic_matcher.update(lm_map)
            if self.calibrating:
                palm = ((landmarks[5][0] - landmarks[17][0]) ** 2 + (landmarks[5][1] - landmarks[17][1]) ** 2) ** 0.5
                self.calibration_samples.append(palm)
            result = self.recognizer.detect_sign(landmarks)
            motion_ok = self.dynamic_matcher.matches_motion(self.session.state.target)
            progress_info = self.session.process_result(result, motion_ok)
            self._apply_result(result, progress_info)
        else:
            self.detected_label.setText("-")
            self.accuracy_label.setText("0%")
            self.status_label.setText("No hand detected.")
            self.progress.setValue(0)
            self.dynamic_matcher.reset()
            self.recognizer.reset()

        bgr = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = bgr.shape
        image = QImage(bgr.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _apply_result(self, result: DetectionResult, progress_info: Dict[str, object]) -> None:
        target = self.session.state.target
        self.detected_label.setText(result.sign)
        self.accuracy_label.setText(f"{round(result.confidence * 100)}%")
        if target in DYNAMIC_SIGNS:
            self.hint_label.setText(f"Dynamic sign: draw {target} motion path.")
        else:
            self.hint_label.setText(result.hint)
        self.coach_label.setText(self._coach_tip(target, result))
        self.progress.setValue(int(progress_info["progress"]))
        self.status_label.setText(str(progress_info["status"]))
        self._refresh_mode_labels()
        self._refresh_target()

    def _coach_tip(self, target: str, result: DetectionResult) -> str:
        m = result.meta
        if target in {"B", "4", "5"} and int(m.get("openCount", 0)) < 4:
            return "Open all four fingers clearly."
        if target in {"D", "1", "L"} and not bool(m.get("indexUp", False)):
            return "Lift your index finger and hold steady."
        if target in {"A", "S", "Y"} and not bool(m.get("thumbOut", False)):
            return "Push thumb farther out."
        if target in {"E", "M", "N", "T"} and bool(m.get("thumbOut", False)):
            return "Tuck thumb inward and keep fist compact."
        return "Great position. Keep your hand steady."

    def _refresh_mode_labels(self, status_override: Optional[str] = None) -> None:
        self.mode_label.setText(self.session.state.mode.value.upper().replace("-", " "))
        self.timer_label.setText(f"{self.session.state.timed_remaining}s" if self.session.state.timed_remaining else "--")
        learn_enabled = not self.session.state.instructor_mode and self.session.state.mode == Mode.LEARN
        self.next_sign_btn.setEnabled(learn_enabled)
        self.prev_sign_btn.setEnabled(learn_enabled)
        if status_override:
            self.status_label.setText(status_override)

    def _refresh_target(self) -> None:
        target = self.session.state.target
        self.target_label.setText(target)
        ref_path = self.reference_map.get(target)
        if ref_path and self.session.state.dual_view:
            pix = self._reference_pixmap(ref_path, 240, 180)
            if not pix.isNull():
                self.ref_label.setPixmap(pix)
            else:
                self.ref_label.setText(f"Target: {target}\n{SIGN_GUIDE_TEXT.get(target, '')}")
        else:
            self.ref_label.setText(f"Target: {target}\n{SIGN_GUIDE_TEXT.get(target, '')}")
        self._refresh_reference_strip()

    def _refresh_reference_strip(self) -> None:
        signs = self.session.state.current_signs
        if not signs:
            return
        idx = self.session.state.target_index
        window = [
            signs[(idx - 1) % len(signs)],
            signs[idx],
            signs[(idx + 1) % len(signs)],
        ]
        widgets = [
            (self.ref_prev, self.ref_prev_text),
            (self.ref_curr, self.ref_curr_text),
            (self.ref_next, self.ref_next_text),
        ]
        for (image_label, text_label), sign in zip(widgets, window):
            path = self.reference_map.get(sign, "")
            pix = self._reference_pixmap(path, 160, 140) if path else QPixmap()
            if not pix.isNull():
                image_label.setPixmap(pix)
            else:
                image_label.setText(sign)
            guide = SIGN_GUIDE_TEXT.get(sign, "")
            text_label.setText(f"{sign} - {guide}")

    def _refresh_full_ui(self, status: str) -> None:
        self.status_label.setText(status)
        self.stop_camera_btn.setEnabled(False)
        self._refresh_mode_labels()
        self._refresh_target()
