from __future__ import annotations

"""GPU runtime checks for the dedicated kinematic ABD pose-history API."""

import math
import tempfile

import numpy as np
import uipc
from uipc.constitution import AffineBodyConstitution
from uipc.core import Engine, Scene, World
from uipc.geometry import label_surface, tetmesh
from uipc.unit import MPa


DT = 0.01


def _q(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (transform[:3, 3], transform[0, :3], transform[1, :3], transform[2, :3])
    )


def _transform(translation=(0.0, 0.0, 0.0), angle=0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = translation
    value[:3, :3] = (
        (math.cos(angle), -math.sin(angle), 0.0),
        (math.sin(angle), math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    return value


def _assert_state(world: World, body_id: int, previous: np.ndarray, current: np.ndarray) -> None:
    state = np.asarray(world.read_kinematic_abd_state_from_sim(body_id))
    assert state.shape == (6, 12)
    q_previous = _q(previous)
    q_current = _q(current)
    delta = q_current - q_previous
    assert np.allclose(state[0], q_previous)
    assert np.allclose(state[1], q_current)
    assert np.allclose(state[2], q_current)
    assert np.allclose(state[3], q_current)
    assert np.allclose(state[4], delta)
    assert np.allclose(state[5], delta / DT)


def main() -> None:
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["sanity_check"] = {"enable": False, "mode": "quiet"}
    config["contact"]["enable"] = False
    scene = Scene(config)
    constitution = AffineBodyConstitution()
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    for body_id in range(2):
        mesh = tetmesh(vertices + (2.0 * body_id, 0.0, 0.0), tetrahedra)
        constitution.apply_to(mesh, 100.0 * MPa)
        uipc.view(mesh.instances().find(uipc.builtin.is_fixed))[:] = 1
        label_surface(mesh)
        scene.objects().create(f"body_{body_id}").geometries().create(mesh)

    with tempfile.TemporaryDirectory(prefix="v61_abd_history_") as workspace:
        engine = Engine("cuda", workspace)
        world = World(engine)
        world.init(scene)
        world.retrieve()

        identity = _transform()
        translation = _transform((0.01, -0.02, 0.03))
        rotation = _transform(angle=0.2)
        assert world.write_kinematic_abd_pose_pair_to_sim(0, identity, translation, DT)
        assert world.write_kinematic_abd_pose_pair_to_sim(1, identity, rotation, DT)
        _assert_state(world, 0, identity, translation)
        _assert_state(world, 1, identity, rotation)

        # Translation + rotation, two consecutive poses, and three substeps.
        previous = translation
        for index in range(3):
            current = _transform((0.011 + index * 0.001, -0.019, 0.031), 0.05 * (index + 1))
            assert world.write_kinematic_abd_pose_pair_to_sim(0, previous, current, DT)
            _assert_state(world, 0, previous, current)
            previous = current

        # Static target has zero delta/velocity and body 1 remains isolated.
        body_1_before = np.asarray(world.read_kinematic_abd_state_from_sim(1)).copy()
        assert world.write_kinematic_abd_pose_pair_to_sim(0, previous, previous, DT)
        _assert_state(world, 0, previous, previous)
        assert np.array_equal(body_1_before, world.read_kinematic_abd_state_from_sim(1))

        nan_transform = identity.copy()
        nan_transform[0, 0] = np.nan
        shear = identity.copy()
        shear[0, 1] = 0.1
        reflection = np.diag((-1.0, 1.0, 1.0, 1.0))
        assert not world.write_kinematic_abd_pose_pair_to_sim(-1, identity, identity, DT)
        assert not world.write_kinematic_abd_pose_pair_to_sim(2, identity, identity, DT)
        assert not world.write_kinematic_abd_pose_pair_to_sim(0, nan_transform, identity, DT)
        assert not world.write_kinematic_abd_pose_pair_to_sim(0, shear, identity, DT)
        assert not world.write_kinematic_abd_pose_pair_to_sim(0, reflection, identity, DT)
        assert not world.write_kinematic_abd_pose_pair_to_sim(0, identity, identity, 0.0)

    print("kinematic ABD pose-history runtime validation: PASS")


if __name__ == "__main__":
    main()
