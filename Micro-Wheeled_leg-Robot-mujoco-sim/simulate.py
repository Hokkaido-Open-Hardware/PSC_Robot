"""Run the CAD-derived robot: supported demo, free physics, or balance control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from balance_controller import BalanceController, pitch_angle

HERE = Path(__file__).resolve().parent


def command(model, data, mode):
    if mode == 'supported':
        # Same hip rotation on both sides, with a small differential motion.
        data.ctrl[0] = .20 * np.sin(2*np.pi*.3*data.time)
        data.ctrl[1] = .20 * np.sin(2*np.pi*.3*data.time + .3) * (1-np.exp(-3*data.time))
        speed = 8 * np.sin(2*np.pi*.15*data.time)
        for i, side in enumerate(['left', 'right']):
            dof = model.joint(side+'_wheel_joint').dofadr[0]
            data.ctrl[2+i] = np.clip(.001*(speed-data.qvel[dof]), -.01, .01)
    else:
        # Hold the CAD leg angle. There is deliberately no balance controller.
        data.ctrl[:] = 0


def closure_error(model, data):
    return max(float(np.linalg.norm(data.site(side+'_calf_pin').xpos - data.site(side+'_follower_pin').xpos))
               for side in ['left', 'right'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['supported', 'free', 'balance'], default='supported',
                        help='supported: fixed chassis demo; free: no balance control; balance: four-state wheel feedback')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=None, help='default: 10 headless, unlimited GUI')
    parser.add_argument('--render', type=Path, help='save the final camera view as PNG (EGL in headless mode)')
    parser.add_argument('--report', type=Path, help='write simulation metrics as JSON')
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and positive')
    if args.headless:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco

    scene = HERE / ('scene_supported.xml' if args.mode == 'supported' else 'scene.xml')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = BalanceController(model, data) if args.mode == 'balance' else None
    limit = args.seconds if args.seconds is not None else (10 if args.headless else float('inf'))
    max_error, max_contacts, min_height, steps = 0., 0, float('inf'), 0
    joint_min, joint_max = data.qpos.copy(), data.qpos.copy()
    max_pitch = abs(pitch_angle(data))
    initial_position = data.body('chassis').xpos.copy()

    def step():
        nonlocal max_error, max_contacts, min_height, steps, max_pitch
        if controller is not None:
            controller.update(data)
        else:
            command(model, data, args.mode)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        steps += 1
        np.minimum(joint_min, data.qpos, out=joint_min)
        np.maximum(joint_max, data.qpos, out=joint_max)
        max_error = max(max_error, closure_error(model, data))
        max_contacts = max(max_contacts, data.ncon)
        min_height = min(min_height, float(data.body('chassis').xpos[2]))
        max_pitch = max(max_pitch, abs(pitch_angle(data)))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
            raise RuntimeError('Simulation produced nonfinite state or a MuJoCo warning')

    print(f'Mode: {args.mode}; {model.nmesh} CAD meshes; {model.nu} actuators; total mass {model.body_mass.sum():.3f} kg (estimated)', flush=True)
    labels = {'supported': 'FIXED CHASSIS - linkage and wheel demo',
              'free': 'FREE BODY - no balance controller',
              'balance': 'BALANCE - four-state feedback, extended legs'}
    print(labels[args.mode], flush=True)
    if args.headless:
        while data.time < limit - model.opt.timestep/2:
            step()
    else:
        import mujoco.viewer
        pending = []

        def key_callback(key):
            if key in [ord('R'), ord('r')]:
                pending.append('reset')

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, .055]
            viewer.cam.distance = .32
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -22
            viewer.opt.geomgroup[3] = 0
            label = labels[args.mode]
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, 'Micro Wheeled-leg Robot\n'+label+'\nR: reset', ''))
            print('R: reset; mouse: orbit/zoom; close the viewer to exit.', flush=True)
            while viewer.is_running() and data.time < limit - model.opt.timestep/2:
                start = time.monotonic()
                if pending:
                    pending.clear()
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    if controller is not None:
                        controller.reset(data)
                    max_error, max_contacts, min_height, steps = 0., 0, float('inf'), 0
                    max_pitch = abs(pitch_angle(data))
                    joint_min[:], joint_max[:] = data.qpos, data.qpos
                for _ in range(16):
                    if data.time >= limit - model.opt.timestep/2:
                        break
                    step()
                if controller is not None:
                    state = 'FALL STOP - press R to reset' if controller.fallen else 'ACTIVE'
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f'Micro Wheeled-leg Robot\n{label}\n{state}\n'
                                      f'Pitch: {np.rad2deg(pitch_angle(data)):.1f} deg\nR: reset', ''))
                viewer.sync()
                time.sleep(max(0, 16*model.opt.timestep-(time.monotonic()-start)))
    report = dict(mode=args.mode, simulated_seconds=float(data.time), steps=steps,
                  meshes=model.nmesh, actuators=model.nu, estimated_mass_kg=float(model.body_mass.sum()),
                  max_linkage_closure_error_m=max_error, max_contacts=max_contacts,
                  min_chassis_height_m=min_height, warnings=sum(w.number for w in data.warning),
                  max_abs_pitch_deg=float(np.rad2deg(max_pitch)),
                  final_pitch_deg=float(np.rad2deg(pitch_angle(data))),
                  chassis_displacement_m=(data.body('chassis').xpos-initial_position).tolist(),
                  joint_range_rad={name: [float(joint_min[model.joint(name).qposadr[0]]), float(joint_max[model.joint(name).qposadr[0]])]
                                   for name in ['left_hip', 'right_hip', 'left_wheel_joint', 'right_wheel_joint']})
    if controller is not None:
        report['balance'] = dict(fallen=controller.fallen,
                                 pitch_reference_deg=controller.gains.pitch_reference_deg,
                                 hip_target_rad=controller.gains.hip_target_rad,
                                 final_chassis_height_m=float(data.body('chassis').xpos[2]),
                                 max_abs_wheel_torque_nm=controller.max_abs_torque,
                                 final_mean_wheel_speed_rad_s=float(np.mean(data.qvel[controller.wheel_dof])))
    if args.render:
        from PIL import Image
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .055]
        camera.distance, camera.azimuth, camera.elevation = .32, 45, -22
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        with mujoco.Renderer(model, height=960, width=1280) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            args.render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.render)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
