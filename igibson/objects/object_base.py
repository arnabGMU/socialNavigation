from abc import ABCMeta, abstractmethod

import numpy as np
import pybullet as p
from future.utils import with_metaclass


class BaseObject(with_metaclass(ABCMeta, object)):
    """This is simply an interface that all objects must implement that does not implement any features on its own."""

    def __init__(self):
        self.states = {}
        self._loaded = False

    def load(self):
        """Load object into pybullet and return list of loaded body ids."""
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._loaded = True
        return self._load()

    @abstractmethod
    def get_body_id(self):
        """
        Gets the body ID for the object.

        If the object somehow has multiple bodies, this will be the default body that the default manipulation functions
        will manipulate.
        """
        pass

    @abstractmethod
    def _load(self):
        pass

    def get_position(self):
        """Get object position in the format of Array[x, y, z]"""
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        pos, orn = p.getBasePositionAndOrientation(self.get_body_id())
        return np.array(pos), np.array(orn)

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        _, old_orn = p.getBasePositionAndOrientation(self.get_body_id())
        self.set_position_orientation(pos, old_orn)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        old_pos, _ = p.getBasePositionAndOrientation(self.get_body_id())
        self.set_position_orientation(old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        p.resetBasePositionAndOrientation(self.get_body_id(), pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        dynamics_info = p.getDynamicsInfo(self.get_body_id(), -1)
        inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
        pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
        self.set_position_orientation(pos, orn)

    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Load the state of the object other than what's not included in pybullet state."""
        pass


class NonRobotObject(BaseObject):
    # This class implements the object interface for non-robot objects.
    # Also allows us to identify non-robot objects until all simulator etc. call for importing etc. are unified.

    # TODO: This renderer_instances logic doesn't actually need to be specific to non-robot objects. Generalize this.
    def __init__(self, **kwargs):
        super(NonRobotObject, self).__init__(**kwargs)

        self.renderer_instances = []

    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)


class SingleBodyObject(NonRobotObject):
    """Provides convenience get_body_id() function for single-body objects."""

    # TODO: Merge this into BaseObject once URDFObject also becomes single-body.

    def __init__(self, **kwargs):
        super(SingleBodyObject, self).__init__(*kwargs)
        self._body_id = None

    def load(self):
        body_ids = super(NonRobotObject, self).load()
        assert len(body_ids) == 1, "SingleBodyObject loaded more than a single body ID."
        self._body_id = body_ids[0]
        return body_ids

    def get_body_id(self):
        return self._body_id
