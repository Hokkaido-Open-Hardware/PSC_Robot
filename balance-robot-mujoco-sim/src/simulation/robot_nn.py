"""NN control with the same sensing, filters, limits, reset and yaw as LQR."""
import math

from robot_lqr import RobotLqr


class RobotNN(RobotLqr):
    controller_name = 'Neural Network'

    def __init__(self, model, data, predictor):
        super().__init__(model, data)
        self.predictor = predictor

    def calculate_motor_velocity(self):
        inputs = self.sample_control_input()
        output = float(self.predictor.predict(inputs))
        if not math.isfinite(output):
            raise ValueError('Neural Network produced a non-finite motor velocity')
        self.record_control_output(inputs[0], output)
        return output


class RobotINT8(RobotNN):
    controller_name = 'NN INT8'
