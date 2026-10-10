from collections import deque
import math
import threading
import time
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel, QFormLayout,
    QComboBox, QScrollArea, QTabWidget
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtGui import (
    QGuiApplication, QSurfaceFormat
)
import time

from robot_lqr import RobotLqr
from robot_nn import RobotNN, RobotINT8
from nn_model import DEFAULT_MODEL, NNPredictor
from int8_model import DEFAULT_INT8_MODEL, INT8Predictor
from training_panel import TrainingPanel
from auto_commands import (GUI_SPEED_LIMIT, GUI_YAW_LIMIT, AutoCommands,
                           AutoRunStatus, SCENARIOS, DEFAULT_DURATION)
from control_timing import CONTROL_DT, control_substeps
from rl_control import RobotRL, RLConfig, body_tilt, invalid_physics, reset_pose
from rl_panel import RLPanel


format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2,0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


class Viewport(QOpenGLWindow):

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn, data_lock) -> None:
        super().__init__()

        self.model = model
        self.data = data
        self.data_lock = data_lock
        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        """ Sets a scale factor that is used to scale the OpenGL window to accommodate
        the high DPI scaling Qt does.
        """
        self.scale = scaleFactor

    def paintGL(self) -> None:
        t = time.time()
        # Scene preparation accesses MjData, including its scratch storage.
        # Rendering the completed scene no longer needs the data lock.
        with self.data_lock:
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None, data_lock=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.data_lock = data_lock if data_lock is not None else threading.Lock()
        self.running = True
        self.control_substeps = control_substeps(model)

        self.robot = RobotLqr(model, data)

        # robot control parameters
        self.speed = 0.0
        self.yaw = 0.0
        self.auto_enabled = False
        self.auto_episode = 0
        self.auto_falls = 0
        self.auto_commands = None
        self.auto_rng = np.random.default_rng(42)

        # Initialize physics, controller state, and the simulation clock together.
        self.reset()

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    @property
    def control_status(self):
        return self.robot.control_status

    @property
    def auto_status(self):
        with self.data_lock:
            return AutoRunStatus(
                self.auto_enabled, self.auto_episode,
                SCENARIOS[(self.auto_episode - 1) % len(SCENARIOS)] if self.auto_enabled else '',
                self.physics_steps * self.model.opt.timestep,
                self.speed, self.yaw, self.auto_falls)

    def run(self) -> None:
        while self.running:
            # don't step the simulation past real time
            # without this the sim usually finishes before it's
            # even visible
            with self.data_lock:
                should_step = self.data.time < self.real_time / 1_000_000_000
                if should_step:
                    self._step_locked()
            if not should_step:
                time.sleep(0.00001)

    def _step_locked(self):
        """Advance one physics step; caller holds the simulation/render lock."""
        if self.physics_steps % self.control_substeps == 0:
            if self.auto_enabled:
                tilt = body_tilt(self.data)
                fell = invalid_physics(self.data) or not math.isfinite(tilt) or tilt > math.pi / 4
                elapsed = self.physics_steps * self.model.opt.timestep
                if fell or self.physics_steps >= round(DEFAULT_DURATION / CONTROL_DT) * self.control_substeps:
                    self.auto_falls += int(fell)
                    self.auto_episode += 1
                    self._reset_locked()
                    elapsed = 0.0
                self.speed, self.yaw = self.auto_commands.step(elapsed, CONTROL_DT)
                if isinstance(self.robot, RobotRL):
                    self.speed = float(np.clip(self.speed, -self.robot.config.target_limit,
                                               self.robot.config.target_limit))
            self.robot.set_velocity_linear_set_point(self.speed)
            self.robot.set_yaw(self.yaw)
            self.robot.update_motor_speed()
        mujoco.mj_step(self.model, self.data)
        self.physics_steps += 1

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        # QThread methods called by the GUI execute on the GUI thread. Protect
        # the entire reset against both stepping and scene preparation.
        with self.data_lock:
            if self.auto_enabled:
                self._restart_auto_locked()
            self._reset_locked()

    def _restart_auto_locked(self):
        self.auto_episode = 1
        self.auto_falls = 0
        self.auto_rng = np.random.default_rng(42)

    def _reset_locked(self):
        mujoco.mj_resetData(self.model, self.data)
        self.speed = 0.0
        self.yaw = 0.0
        self.robot.reset()
        if self.auto_enabled:
            # Start near upright for every controller, using MuJoCo's w,x,y,z order.
            config = self.robot.config if isinstance(self.robot, RobotRL) else RLConfig()
            reset_pose(self.model, self.data, self.auto_rng, config)
            scenario = SCENARIOS[(self.auto_episode - 1) % len(SCENARIOS)]
            self.auto_commands = AutoCommands(scenario, DEFAULT_DURATION, self.auto_rng)
        # robot.reset changes qpos; refresh derived state afterwards.
        mujoco.mj_forward(self.model, self.data)
        self.real_time_start = time.monotonic_ns()
        self.physics_steps = 0

    def set_auto(self, enabled):
        with self.data_lock:
            if self.auto_enabled == bool(enabled):
                return
            self.auto_enabled = bool(enabled)
            self.speed = self.yaw = 0.0
            if self.auto_enabled:
                self._restart_auto_locked()
                self._reset_locked()
            else:
                self.auto_commands = None
                self.robot.set_velocity_linear_set_point(0.0)
                self.robot.set_yaw(0.0)

    def set_speed(self, speed: float) -> None:
        with self.data_lock:
            if not self.auto_enabled:
                limit = self.robot.config.target_limit if isinstance(self.robot, RobotRL) else GUI_SPEED_LIMIT
                self.speed = float(np.clip(speed, -limit, limit))

    def set_yaw(self, yaw: float) -> None:
        with self.data_lock:
            if not self.auto_enabled:
                self.yaw = yaw

    def set_controller(self, name, predictor=None):
        if name not in ('LQR', 'Neural Network', 'Reinforcement Learning', 'NN INT8'):
            raise ValueError(f'Unknown controller: {name}')
        if name == 'Neural Network' and predictor is None:
            raise ValueError('Train or load a Neural Network model first')
        if name == 'Reinforcement Learning' and predictor is None:
            raise ValueError('Train or load a Reinforcement Learning model first')
        if name == 'NN INT8' and not isinstance(predictor, INT8Predictor):
            raise ValueError('Load a real TFLite INT8 model first')
        with self.data_lock:
            previous = self.robot
            robot = (RobotLqr(self.model, self.data) if name == 'LQR' else
                     RobotNN(self.model, self.data, predictor) if name == 'Neural Network' else
                     RobotINT8(self.model, self.data, predictor) if name == 'NN INT8' else
                     RobotRL(self.model, self.data, predictor))
            if name == 'Reinforcement Learning':
                self.speed = float(np.clip(self.speed, -robot.config.target_limit, robot.config.target_limit))
                robot.velocity_linear_set_point = float(np.clip(
                    previous.velocity_linear_set_point, -robot.config.target_limit, robot.config.target_limit))
                left = float(self.data.actuator('motor_l_wheel').ctrl[0])
                right = float(self.data.actuator('motor_r_wheel').ctrl[0])
                robot.previous_action = float(np.clip((right - left) / 2 / robot.config.action_scale, -1, 1))
            # Preserve the physics state and filter history for a live switch.
            robot.pitch_dot_filtered = previous.pitch_dot_filtered
            robot.velocity_angular_filtered = previous.velocity_angular_filtered
            robot.set_velocity_linear_set_point(self.speed)
            robot.set_yaw(self.yaw)
            self.robot = robot


class Window(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.nn_predictor = None
        self.rl_predictor = None
        self.int8_predictor = None

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('scene.xml')))
        self.data = mujoco.MjData(self.model)
        self.data_lock = threading.Lock()
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn, self.data_lock)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        layout_top.addWidget(reset_button)
        layout_robot_controls = QVBoxLayout()
        layout_robot_controls.setContentsMargins(0,0,0,0)
        layout_robot_controls.addWidget(self.create_top())
        layout_top.addLayout(layout_robot_controls)
        layout_top.setContentsMargins(8,0,8,0)
        layout_view = QHBoxLayout()
        layout_view.addWidget(QWidget.createWindowContainer(self.viewport), 1)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(self.create_control_status())
        self.training_panel = TrainingPanel()
        self.training_panel.modelReady.connect(self.install_nn_model)
        self.rl_panel = RLPanel()
        self.rl_panel.modelReady.connect(self.install_rl_model)
        self.training_tabs = QTabWidget()
        self.training_tabs.addTab(self.training_panel, 'Supervised NN')
        self.training_tabs.addTab(self.rl_panel, 'RL (PPO)')
        sidebar_layout.addWidget(self.training_tabs)
        sidebar_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(435)
        scroll.setWidget(sidebar)
        layout_view.addWidget(scroll)
        layout.addLayout(layout_view, 1)
        layout.addLayout(layout_top)
        layout.setContentsMargins(0,4,0,0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(1280, 900)

        self.th = UpdateSimThread(self.model, self.data, self, data_lock=self.data_lock)
        if DEFAULT_MODEL.exists():
            try:
                self.install_nn_model(NNPredictor(DEFAULT_MODEL))
            except Exception as exc:
                self.training_panel.message.setText(f'Model could not be loaded: {exc}')
        if DEFAULT_INT8_MODEL.exists():
            try:
                self.int8_predictor = INT8Predictor(DEFAULT_INT8_MODEL)
                self.controller_selector.model().item(3).setEnabled(True)
                self.controller_selector.setItemData(3, str(DEFAULT_INT8_MODEL), Qt.ToolTipRole)
            except Exception as exc:
                self.controller_selector.setItemData(3, f'INT8 unavailable: {exc}', Qt.ToolTipRole)
                self.int8_status.setText(f'NN INT8 unavailable: {exc}')
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.update_control_status)
        self.update_control_status()
        self.status_timer.start()
        self.th.start()

    def create_control_status(self):
        panel = QGroupBox("Control Status")
        panel.setMinimumWidth(380)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QFormLayout(panel)
        self.controller_label = QLabel()
        self.controller_label.setWordWrap(True)
        layout.addRow("Controller:", self.controller_label)
        self.control_status_labels = {}
        fields = (
            ("pitch", "Pitch [deg]", "Controller pitch sign convention."),
            ("pitch_rate", "Pitch Rate [deg/s]", "Filtered for LQR/NN; raw local X angular velocity for RL."),
            ("left_wheel_velocity", "Left Wheel Velocity [rad/s]", "Raw left joint velocity."),
            ("right_wheel_velocity", "Right Wheel Velocity [rad/s]", "Raw right joint velocity."),
            ("average_wheel_velocity", "Average Wheel Velocity [rad/s]", "Forward-positive average: (-left + right) / 2."),
            ("target_linear_velocity", "Target Linear Velocity [m/s]", "Target used in this control update."),
            ("velocity_error", "Velocity Error [m/s]", "LQR/NN: target minus filtered wheel speed. RL: target minus body forward speed."),
            ("control_output", "Output [rad/s]", "Controller output before motor limiting and yaw mixing."),
            ("left_motor_command", "Left Motor Command", "Command written to the left actuator."),
            ("right_motor_command", "Right Motor Command", "Command written to the right actuator."),
            ("yaw_command", "Yaw Command", "Yaw used in this control update."),
            ("simulation_time", "Simulation Time [s]", "Simulation time of this control sample."),
        )
        for field, title, tooltip in fields:
            label = QLabel(title)
            value = QLabel("—")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setMinimumWidth(85)
            label.setToolTip(tooltip)
            value.setToolTip(tooltip)
            layout.addRow(label, value)
            self.control_status_labels[field] = value
            if field == "control_output":
                self.control_output_label = label
        return panel

    @Slot()
    def update_control_status(self):
        # Read one immutable snapshot, never live MuJoCo data or controller
        # calculations. Polling avoids queueing an event for every control tick.
        status = self.th.control_status
        self.controller_label.setText(status.controller)
        name = 'RL' if status.controller == 'Reinforcement Learning' else status.controller
        self.control_output_label.setText(f"{name} Output [rad/s]")
        for field, label in self.control_status_labels.items():
            value = getattr(status, field)
            if value is None:
                label.setText("—")
                continue
            if field in ("pitch", "pitch_rate"):
                value = math.degrees(value)
            digits = 2 if field in ("pitch", "pitch_rate", "simulation_time") else 3
            value = round(value, digits)
            label.setText(f"{0.0 if value == 0 else value:.{digits}f}")
        auto = self.th.auto_status
        if auto.enabled:
            self.auto_status_label.setText(
                f'AUTO • Episode {auto.episode} • {auto.scenario} • '
                f'{auto.elapsed:.1f} / {DEFAULT_DURATION:g} s • Falls {auto.falls}\n'
                f'Speed {auto.speed:+.3f} m/s • Yaw {auto.yaw:+.3f} rad/s')
            self._display_commands(auto.speed, auto.yaw)
        else:
            self.auto_status_label.setText('MANUAL • Adjust Speed / Yaw')

    @Slot(float)
    def show_runtime(self, fps: float):
        with self.data_lock:
            simulation_time = self.data.time
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {simulation_time:.0f}s"
        )

    def create_top(self):
        layout = QVBoxLayout()
        controller_layout = QHBoxLayout()
        controller_layout.addWidget(QLabel('Controller'))
        self.controller_selector = QComboBox()
        self.controller_selector.addItems(['LQR', 'Neural Network', 'Reinforcement Learning', 'NN INT8'])
        self.controller_selector.model().item(1).setEnabled(False)
        self.controller_selector.model().item(2).setEnabled(False)
        self.controller_selector.model().item(3).setEnabled(False)
        self.controller_selector.setItemData(1, 'NN FP32: original trained weights', Qt.ToolTipRole)
        self.controller_selector.setToolTip('Train or load a model to enable NN / RL. Startup always uses LQR.')
        self.controller_selector.currentTextChanged.connect(self.change_controller)
        controller_layout.addWidget(self.controller_selector)
        self.int8_status = QLabel('Neural Network = NN FP32; NN INT8 = ROBOT.TFL')
        self.int8_status.setWordWrap(True)
        controller_layout.addWidget(self.int8_status)
        controller_layout.addWidget(QLabel('Command mode'))
        self.command_mode = QComboBox()
        self.command_mode.addItems(['MANUAL', 'AUTO'])
        self.command_mode.setToolTip(
            'AUTO cycles driving commands and resets after a fall or 8 seconds. '
            'Use the training tabs to train models with their own automatic commands.')
        self.command_mode.currentTextChanged.connect(self.change_command_mode)
        controller_layout.addWidget(self.command_mode)
        layout.addLayout(controller_layout)
        self.auto_status_label = QLabel('MANUAL • Adjust Speed / Yaw')
        self.auto_status_label.setWordWrap(True)
        layout.addWidget(self.auto_status_label)
        # layout.setContentsMargins(0,0,0,0)
        label_width = 60

        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        # QSliders only support ints, so scale the values we want by 1000
        # and then remove this scale factor in the valueChanged handler
        self.speed_slider.setMinimum(-int(GUI_SPEED_LIMIT * 1000))
        self.speed_slider.setMaximum(int(GUI_SPEED_LIMIT * 1000))
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_label = QLabel("Speed")
        speed_label.setFixedWidth(label_width)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)

        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-int(GUI_YAW_LIMIT * 1000))
        self.yaw_slider.setMaximum(int(GUI_YAW_LIMIT * 1000))
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self._yaw_changed)
        yaw_label = QLabel("Yaw")
        yaw_label.setFixedWidth(label_width)
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_slider)

        layout.addLayout(speed_layout)
        layout.addLayout(yaw_layout)

        w = QGroupBox("Robot Control")
        w.setLayout(layout)
        return w

    def _display_commands(self, speed, yaw):
        for slider, value in ((self.speed_slider, speed), (self.yaw_slider, yaw)):
            previous = slider.blockSignals(True)
            slider.setValue(round(value * 1000))
            slider.blockSignals(previous)

    def change_command_mode(self, mode):
        auto = mode == 'AUTO'
        self.th.set_auto(auto)
        self.speed_slider.setEnabled(not auto)
        self.yaw_slider.setEnabled(not auto)
        self._display_commands(0, 0)
        self.update_control_status()

    def _speed_changed(self, value: int) -> None:
        speed = value / 1000
        self.th.set_speed(speed)

    def _yaw_changed(self, value: int) -> None:
        yaw = value / 1000
        self.th.set_yaw(yaw)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.th.reset()
        self.update_control_status()

    def install_nn_model(self, predictor):
        self.nn_predictor = predictor
        self.controller_selector.model().item(1).setEnabled(True)
        if self.controller_selector.currentText() == 'Neural Network':
            self.th.set_controller('Neural Network', predictor)

    def change_controller(self, name):
        predictor = (self.int8_predictor if name == 'NN INT8' else
                     self.rl_predictor if name == 'Reinforcement Learning' else self.nn_predictor)
        self.th.set_controller(name, predictor)
        self.update_speed_range()
        self.update_control_status()

    def update_speed_range(self):
        rl = self.controller_selector.currentText() == 'Reinforcement Learning'
        limit = self.rl_predictor.config.target_limit if rl else GUI_SPEED_LIMIT
        blocked = self.speed_slider.blockSignals(True)
        self.speed_slider.setRange(-round(limit * 1000), round(limit * 1000))
        self.speed_slider.setValue(round(self.th.speed * 1000))
        self.speed_slider.blockSignals(blocked)
        self.speed_slider.setToolTip(
            f'Target speed: ±{limit:g} m/s. ' +
            ('RL commands ramp at the training rate; velocity-stage training is required for driving.'
             if rl else 'Manual forward / reverse command.'))

    def install_rl_model(self, predictor):
        self.rl_predictor = predictor
        self.controller_selector.model().item(2).setEnabled(True)
        if self.controller_selector.currentText() == 'Reinforcement Learning':
            self.th.set_controller('Reinforcement Learning', predictor)
            self.update_speed_range()

    def closeEvent(self, event):
        self.status_timer.stop()
        self.viewport.timer.stop()
        self.th.stop()
        self.training_panel.shutdown()
        self.rl_panel.shutdown()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
