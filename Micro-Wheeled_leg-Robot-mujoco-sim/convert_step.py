"""Convert the upstream STEP assembly into articulated MuJoCo models.

CAD dimensions/axes are measured; masses, damping and motor limits are estimates.
Requires cadquery-ocp 8.0.1.1.0 only when regenerating assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'external/Micro-Wheeled_leg-Robot/1.RobotModel/OriginalRobotModel.stp'
# CAD: mm, Y up, Z along axles. Robot: m, X forward, Y left, Z up.
ROTATION = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
CAD_ORIGIN = np.array([0, 0, 32])


def xyz(point):
    return ROTATION @ (np.asarray(point) - CAD_ORIGIN) * .001


def fmt(values):
    return ' '.join(f'{v:.9g}' for v in np.atleast_1d(values))


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def read_parts(path):
    doc = TDocStd_Document(TCollection_ExtendedString('robot'))
    reader = STEPCAFControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(doc):
        raise RuntimeError(f'Cannot read STEP assembly: {path}')
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = Sequence_TDF_Label()
    tool.GetFreeShapes(roots)
    parts, skipped = [], []

    def walk(label, parent_location, path_names):
        location = parent_location.Multiplied(tool.GetLocation_s(label))
        reference = TDF_Label()
        if tool.IsReference_s(label):
            tool.GetReferredShape_s(label, reference)
        else:
            reference = label
        attr = TDataStd_Name()
        name = attr.Get().ToExtString() if reference.FindAttribute(TDataStd_Name.GetID_s(), attr) else '?'
        children = Sequence_TDF_Label()
        tool.GetComponents_s(reference, children)
        if children.Length():
            for i in range(1, children.Length() + 1):
                walk(children.Value(i), location, path_names + [name])
            return
        shape = tool.GetShape_s(reference).Located(location)
        bounds = Bnd_Box()
        BRepBndLib.Add_s(shape, bounds)
        if bounds.IsVoid():
            skipped.append(name)
            return
        lo, hi = bounds.CornerMin(), bounds.CornerMax()
        parts.append(dict(id=len(parts), name=name, assembly=path_names,
                          cad_bounds_mm=[lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()], shape=shape))

    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), [])
    return parts, skipped


def triangulate(shape, origin):
    BRepMesh_IncrementalMesh(shape, .15, False, .35, True)
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    triangles = []
    while faces.More():
        face = TopoDS.Face(faces.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            points = []
            for i in range(1, mesh.NbNodes() + 1):
                p = mesh.Node(i).Transformed(location.Transformation())
                points.append(xyz([p.X(), p.Y(), p.Z()]) - origin)
            for i in range(1, mesh.NbTriangles() + 1):
                indices = list(mesh.Triangle(i).Get())
                if face.Orientation() == TopAbs_REVERSED:
                    indices[1], indices[2] = indices[2], indices[1]
                triangle = np.array([points[k - 1] for k in indices])
                if np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])) > 1e-16:
                    triangles.append(triangle)
        faces.Next()
    if not triangles:
        raise ValueError('Part has no nondegenerate triangles')
    return np.asarray(triangles)


def save_stl(path, triangles):
    with path.open('wb') as f:
        f.write(b'STEP converted to meters; source: MuShibo/Micro-Wheeled_leg-Robot'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            normal /= np.linalg.norm(normal)
            f.write(struct.pack('<12fH', *normal, *tri.reshape(-1), 0))


def configuration(parts):
    """Explicit reviewed grouping for this specific upstream assembly, not a generic CAD importer."""
    groups = {'chassis': list(range(len(parts)))}
    moving = {
        'left_thigh': [3, 22, 24, 25, 31],
        'right_thigh': [12, 23, 26, 27, 30],
        'left_calf': [18, 33, 34, 37, 48, 58, 62, 69],
        'right_calf': [20, 32, 35, 39, 50, 57, 63, 71],
        'left_follower': [28, 36],
        'right_follower': [29, 38],
        'left_wheel': [19, 65],
        'right_wheel': [21, 64],
    }
    assigned = [i for ids in moving.values() for i in ids]
    assert len(assigned) == len(set(assigned))
    groups['chassis'] = [i for i in groups['chassis'] if i not in assigned]
    groups.update(moving)
    # Pin/hole cylinder axes measured from the STEP B-rep (CAD millimeters).
    origins = {'chassis': xyz([0, 0, 32])}
    for side, z in [('left', 73.8), ('right', -9.8)]:
        origins[side+'_thigh'] = xyz([-4.79358, 15.5, z])
        origins[side+'_calf'] = xyz([43.84353, -3.1395, z])
        origins[side+'_follower'] = xyz([12.90107, 31.71416, z])
        origins[side+'_wheel'] = xyz([-2.89029, -20.91448, 75.65 if side == 'left' else -11.65])
    return groups, origins


def color(name):
    if '轮胎' in name:
        return '.07 .075 .08 1'
    if 'PCB' in name or '编码器' in name:
        return '.08 .35 .18 1'
    if any(word in name for word in ['腿', '侧板', '前板', '上壳']):
        return '.17 .19 .22 1'
    if '电池' in name:
        return '.12 .14 .18 1'
    return '.58 .61 .65 1'


def build_model(parts, groups, origins, fixed):
    root = ET.Element('mujoco', model='Micro Wheeled-leg Robot' + (' - supported demo' if fixed else ''))
    sub(root, 'compiler', angle='radian', meshdir='meshes', autolimits='true', inertiafromgeom='false')
    sub(root, 'option', timestep='.001', integrator='implicitfast', solver='Newton', iterations='80', tolerance='1e-10')
    sub(root, 'statistic', center='0 0 .06', extent='.26')
    visual = sub(root, 'visual')
    sub(visual, 'global', offwidth='1280', offheight='960', azimuth='135', elevation='-22')
    sub(visual, 'headlight', ambient='.45 .45 .45', diffuse='.65 .65 .65')
    defaults = sub(root, 'default')
    sub(defaults, 'joint', damping='.002', armature='1e-6')
    mesh_default = sub(defaults, 'default', **{'class': 'cad'})
    sub(mesh_default, 'geom', type='mesh', contype='0', conaffinity='0', group='2', density='0')
    collision_default = sub(defaults, 'default', **{'class': 'collision'})
    # Ground-only collisions avoid replacing detailed concave CAD with convex hulls.
    sub(collision_default, 'geom', contype='2', conaffinity='1', group='3', rgba='.3 .8 .4 .25',
        friction='.8 .005 .0001', solref='.006 1', condim='4')
    assets = sub(root, 'asset')
    sub(assets, 'texture', name='checker', type='2d', builtin='checker', width='256', height='256', rgb1='.22 .27 .32', rgb2='.28 .33 .38')
    sub(assets, 'material', name='floor', texture='checker', texrepeat='12 12', texuniform='true', reflectance='.1')
    for p in parts:
        sub(assets, 'mesh', name=f'part_{p["id"]:03d}', file=f'part_{p["id"]:03d}.stl')
    world = sub(root, 'worldbody')
    sub(world, 'light', pos='0 -.5 1', dir='0 .5 -1', directional='true')
    sub(world, 'geom', name='ground', type='plane', size='2 2 .01', material='floor', contype='1', conaffinity='2')
    sub(world, 'camera', name='overview', pos='.22 -.28 .17', xyaxes='.786 .618 0 -.235 .299 .925')
    # At q=0 the free model's wheel bottoms are 0.5 mm above the floor.
    chassis = sub(world, 'body', name='chassis', pos=fmt([0, 0, .065 if fixed else .03991448]))
    if not fixed:
        sub(chassis, 'freejoint', name='root')
    bodies = {'chassis': chassis}
    for side in ['left', 'right']:
        thigh, calf, wheel, follower = [side+'_'+suffix for suffix in ['thigh', 'calf', 'wheel', 'follower']]
        bodies[thigh] = sub(chassis, 'body', name=thigh, pos=fmt(origins[thigh]))
        sub(bodies[thigh], 'joint', name=side+'_hip', axis='0 1 0', range='-.45 .45', damping='.01')
        bodies[calf] = sub(bodies[thigh], 'body', name=calf, pos=fmt(origins[calf]-origins[thigh]))
        sub(bodies[calf], 'joint', name=side+'_knee', axis='0 1 0', range='-1.5 1.5')
        bodies[wheel] = sub(bodies[calf], 'body', name=wheel, pos=fmt(origins[wheel]-origins[calf]))
        sub(bodies[wheel], 'joint', name=side+'_wheel_joint', axis='0 1 0', damping='.00002')
        bodies[follower] = sub(chassis, 'body', name=follower, pos=fmt(origins[follower]))
        sub(bodies[follower], 'joint', name=side+'_follower_joint', axis='0 1 0', range='-1.5 1.5')
        sub(bodies[calf], 'site', name=side+'_calf_pin', pos=fmt(xyz([55.05965, 1.12649, 73.8 if side == 'left' else -9.8])-origins[calf]), size='.001', rgba='1 .5 0 1')
        sub(bodies[follower], 'site', name=side+'_follower_pin', pos=fmt(xyz([55.05965, 1.12649, 73.8 if side == 'left' else -9.8])-origins[follower]), size='.001', rgba='1 .5 0 1')
        sub(bodies[wheel], 'geom', name=side+'_tire_collision', **{'class': 'collision'}, type='cylinder', size='.0185 .0065', quat='.7071067812 .7071067812 0 0')
        sub(bodies[calf], 'geom', **{'class': 'collision'}, type='capsule', size='.004', fromto=fmt(np.r_[np.zeros(3), origins[wheel]-origins[calf]]))
        sub(bodies[thigh], 'geom', **{'class': 'collision'}, type='capsule', size='.006', fromto=fmt(np.r_[np.zeros(3), origins[calf]-origins[thigh]]))
    sub(chassis, 'geom', name='chassis_collision', **{'class': 'collision'}, type='box', pos='.002 0 .018', size='.028 .030 .018')
    # Lumped masses are placeholders, not STEP material properties or measurements.
    masses = {'chassis': .210, 'thigh': .014, 'calf': .021, 'follower': .002, 'wheel': .018}
    for group, ids in groups.items():
        body = bodies[group]
        mass = masses['chassis' if group == 'chassis' else group.split('_')[1]]
        points = np.concatenate([parts[i]['vertices'] for i in ids])
        lo, hi = points.min(axis=0), points.max(axis=0)
        center, lengths = (lo+hi)/2, np.maximum(hi-lo, .002)
        inertia = mass/12 * np.array([lengths[1]**2+lengths[2]**2, lengths[0]**2+lengths[2]**2, lengths[0]**2+lengths[1]**2])
        sub(body, 'inertial', pos=fmt(center), mass=mass, diaginertia=fmt(inertia))
        for i in ids:
            sub(body, 'geom', name=f'visual_{i:03d}', **{'class': 'cad'}, mesh=f'part_{i:03d}', rgba=color(parts[i]['name']))
    sub(chassis, 'site', name='imu', pos='0 0 .025', size='.001')
    equality = sub(root, 'equality')
    for side in ['left', 'right']:
        sub(equality, 'connect', name=side+'_four_bar', site1=side+'_calf_pin', site2=side+'_follower_pin', solref='.004 1')
    actuators = sub(root, 'actuator')
    for side in ['left', 'right']:
        sub(actuators, 'position', name=side+'_hip_servo', joint=side+'_hip', kp='2', kv='.04', ctrlrange='-.4 .4', forcerange='-.25 .25')
    for side in ['left', 'right']:
        sub(actuators, 'motor', name=side+'_wheel_motor', joint=side+'_wheel_joint', ctrlrange='-.03 .03')
    sensors = sub(root, 'sensor')
    sub(sensors, 'gyro', name='gyro', site='imu')
    sub(sensors, 'accelerometer', name='accelerometer', site='imu')
    sub(sensors, 'framequat', name='orientation', objtype='body', objname='chassis')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    parts, skipped = read_parts(args.source)
    # Numeric component mapping deliberately refuses a structurally different assembly.
    expected = {22: '舵盘', 24: '大腿', 28: '小腿副拉杆', 62: '小腿_2208电机版本', 65: '2208电机轮胎', 72: 'ESP32PCB'}
    if len(parts) != 73 or any(parts[i]['name'] != name for i, name in expected.items()):
        raise ValueError('STEP assembly changed; review component grouping before conversion.')
    groups, origins = configuration(parts)
    meshdir = HERE / 'meshes'
    meshdir.mkdir(exist_ok=True)
    for group, ids in groups.items():
        for i in ids:
            triangles = triangulate(parts[i].pop('shape'), origins[group])
            save_stl(meshdir / f'part_{i:03d}.stl', triangles)
            parts[i].update(body=group, triangles=len(triangles), vertices=triangles.reshape(-1,3))
    for name, fixed in [('scene.xml', False), ('scene_supported.xml', True)]:
        root = build_model(parts, groups, origins, fixed)
        ET.indent(root)
        ET.ElementTree(root).write(HERE / name, encoding='utf-8', xml_declaration=True)
    manifest = dict(source=str(args.source.relative_to(HERE.parent)) if args.source.is_relative_to(HERE.parent) else str(args.source),
                    sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    units='meters', cad_to_robot='x=-CAD.x/1000; y=(CAD.z-32)/1000; z=CAD.y/1000',
                    body_origins_m={k:v.tolist() for k,v in origins.items()}, skipped_empty_parts=skipped,
                    physical_parameters='Estimated masses and box inertias; not calibrated. Ground-only simplified collisions.',
                    parts=[{k:v for k,v in p.items() if k != 'vertices'} for p in parts])
    (HERE / 'conversion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Converted {len(parts)} parts, {sum(p["triangles"] for p in parts):,} triangles; {len(skipped)} empty part(s) skipped.')


if __name__ == '__main__':
    main()
