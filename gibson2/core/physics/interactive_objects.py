import pybullet as p
import os
import gibson2
import numpy as np


class YCBObject(object):
    def __init__(self, name, scale=1):
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                     'textured_simple.obj')
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                     'textured_simple_vhacd.obj')
        self.scale = scale
        self.body_id = None
    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.collision_filename,
                                              meshScale=self.scale)
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                              fileName=self.visual_filename,
                                              meshScale=self.scale)

        body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=[0, 0, 0],
                                    baseMass=0.1)

        self.body_id = body_id
        return body_id


class ShapeNetObject(object):
    def __init__(self, path, scale=1., position=[0, 0, 0], orientation=[0, 0, 0]):
        self.filename = path

        self.scale = scale
        self.position = position
        self.orientation = orientation

        self._default_mass = 3.
        self._default_transform = {
            'position': [0, 0, 0],
            'orientation_quat': [1. / np.sqrt(2), 0, 0, 1. / np.sqrt(2)],
        }
        pose = p.multiplyTransforms(positionA=self.position,
                                    orientationA=p.getQuaternionFromEuler(self.orientation),
                                    positionB=self._default_transform['position'],
                                    orientationB=self._default_transform['orientation_quat'])
        self.pose = {
            'position': pose[0],
            'orientation_quat': pose[1],
        }
        self.body_id = None

    def load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.filename,
                                              meshScale=self.scale)
        body_id = p.createMultiBody(basePosition=self.pose['position'],
                                    baseOrientation=self.pose['orientation_quat'],
                                    baseMass=self._default_mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=-1)
        self.body_id = body_id
        return body_id


class Pedestrian(object):
    def __init__(self, style='standing', pos=[0, 0, 0], orn=[1. / np.sqrt(2), 0, 0, 1. / np.sqrt(2)]):
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                               'person_{}'.format(style), 'meshes',
                                               'person_vhacd.obj')
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'person_meshes',
                                            'person_{}'.format(style), 'meshes', 'person.obj')
        self.body_id = None
        self.cid = None
        self.collision_id = None
        self.visual_id = None

        self.pos = pos
        self.orn = orn

    def load(self):
        self.collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.collision_filename)
        self.visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.visual_filename)
        body_id = p.createMultiBody(basePosition=self.pos,
                                    baseOrientation=self.orn,
                                    baseMass=60,
                                    baseCollisionShapeIndex=self.collision_id,
                                    baseVisualShapeIndex=self.visual_id)
        self.body_id = body_id

        self.cid = p.createConstraint(self.body_id,
                                      -1,
                                      -1,
                                      -1,
                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.pos,
                                      parentFrameOrientation=[-0.5, -0.5, -0.5,
                                                              0.5])    # facing x axis
        return body_id

    def reset_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)
        
    def set_position(self, pos, orn=None):
        if orn is None:
            _, orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        
    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return pos
    
class CylinderPedestrian(object):
    def __init__(self, radius=0.3, length=2.0, pos=[0, 0, 0],
                 orn=[1, 0, 0, 0], rgba_color=[0, 1, 1, 0.8]):

        self.mass = 100
        
        self.body_id = None
        self.cid = None
        self.collision_id = None
        self.visual_id = None

        self.radius = radius
        self.length = length
        self.rgba_color = rgba_color
        self.pos = pos
        self.orn = orn

    def load(self):
        self.collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.radius)
        visualShapeId = p.createVisualShape(p.GEOM_CYLINDER, radius=self.radius, length=self.length, rgbaColor=self.rgba_color)

        body_id = p.createMultiBody(baseMass=self.mass,
                                         baseCollisionShapeIndex=self.collision_id,
                                         baseVisualShapeIndex=visualShapeId,
                                         basePosition=self.pos,
                                         baseOrientation=self.orn)
        
        self.body_id = body_id 
        
        p.resetBasePositionAndOrientation(self.body_id, self.pos, self.orn)

        self.cid = p.createConstraint(self.body_id,
                                      -1,
                                      -1,
                                      -1,
                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      self.pos,
                                      parentFrameOrientation=[1, 0, 0, 0])    # facing x axis
        return body_id

    def reset_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)
        
    def set_position(self, pos, orn=None):
        if orn is None:
            _, orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        
    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return pos


class VisualMarker(object):
    def __init__(self,
                 visual_shape=p.GEOM_SPHERE,
                 rgba_color=[1, 0, 0, 0.5],
                 radius=1.0,
                 half_extents=[1, 1, 1],
                 length=1,
                 initial_offset=[0, 0, 0]):
        """
        create a visual shape to show in pybullet and MeshRenderer
        :param visual_shape: pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER, pybullet.GEOM_CAPSULE or pybullet.GEOM_SPHERE
        :param rgba_color: color
        :param radius: radius (for sphere)
        :param half_extents: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param length: parameters for pybullet.GEOM_BOX, pybullet.GEOM_CYLINDER or pybullet.GEOM_CAPSULE
        :param initial_offset: visualFramePosition for the marker
        """
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset
        self.body_id = None

    def load(self):
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        halfExtents=self.half_extents,
                                        visualFramePosition=self.initial_offset)
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        length=self.length,
                                        visualFramePosition=self.initial_offset)
        else:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        visualFramePosition=self.initial_offset)
        self.body_id = p.createMultiBody(baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)
        return self.body_id

    def set_position(self, pos, new_orn=None):
        if new_orn is None:
            _, new_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, new_orn)

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)
        
class VisualObject(object):
    def __init__(self,
                 visual_shape=p.GEOM_SPHERE,
                 rgba_color=[1, 0, 0, 0.5],
                 radius=1.0,
                 half_extents=[1, 1, 1],
                 length=1,
                 initial_offset=[0, 0, 0]):
        self.visual_shape = visual_shape
        self.rgba_color = rgba_color
        self.radius = radius
        self.half_extents = half_extents
        self.length = length
        self.initial_offset = initial_offset

    def load(self):
        if self.visual_shape == p.GEOM_BOX:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        halfExtents=self.half_extents,
                                        visualFramePosition=self.initial_offset)
        elif self.visual_shape in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        length=self.length,
                                        visualFramePosition=self.initial_offset)
        else:
            shape = p.createVisualShape(self.visual_shape,
                                        rgbaColor=self.rgba_color,
                                        radius=self.radius,
                                        visualFramePosition=self.initial_offset)
        self.body_id = p.createMultiBody(baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)

        return self.body_id

    def set_position(self, pos, new_orn=None):
        if new_orn is None:
            _, new_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, new_orn)

    def set_color(self, color):
        p.changeVisualShape(self.body_id, -1, rgbaColor=color)


class BoxShape(object):
    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], rgba_color=[1.0, 1.0, 1.0, 1.0], mass=1000):
        self.basePos = pos
        self.dimension = [dim[i] / 2.0 for i in range(3)]

        self.rgba_color = rgba_color
        self.collision_id = None

        self.body_id = None

        self.mass = mass


    def load(self):
        baseOrientation = [0, 0, 0, 1]

        self.collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.rgba_color)

        self.body_id = p.createMultiBody(baseMass=self.mass,
                                         baseCollisionShapeIndex=self.collision_id,
                                         baseVisualShapeIndex=visualShapeId,
                                         basePosition=self.basePos,
                                         baseOrientation=baseOrientation)

        p.resetBasePositionAndOrientation(self.body_id, self.basePos, baseOrientation)
        return self.body_id

    def set_position(self, pos):
        _, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)
        
    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return pos
    
    def get_dimensions(self):
        return self.dimension
    

class InteractiveObj(object):
    """
    Interactive Objects are represented as a urdf, but doesn't have motors
    """
    def __init__(self, filename, scale=1):
        self.filename = filename
        self.scale = scale
        self.body_id = None

    def load(self):
        self.body_id = p.loadURDF(self.filename, globalScaling=self.scale)
        self.mass = p.getDynamicsInfo(self.body_id, -1)[0]
        return self.body_id

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        org_pos, org_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, org_orn)

    def set_position_rotation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)


class RBOObject(InteractiveObj):
    def __init__(self, name, scale=1):
        filename = os.path.join(gibson2.assets_path, 'models', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)
