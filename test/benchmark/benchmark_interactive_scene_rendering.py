#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.utils.constants import AVAILABLE_MODALITIES
from gibson2.utils.utils import parse_config
from gibson2.utils.constants import NamedRenderingPresets
import os
import gibson2
import time
import random
import matplotlib.pyplot as plt
from gibson2.utils.assets_utils import get_ig_assets_version
from gibson2.utils.assets_utils import get_scene_path
import pickle as pkl

def benchmark_rendering(scene_list, rendering_presets_list, modality_list):
    config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))
    assets_version = get_ig_assets_version()
    print('assets_version', assets_version)
    result = {}
    for scene_name in scene_list:
        for rendering_preset in rendering_presets_list:
            scene = InteractiveIndoorScene(
                scene_name, texture_randomization=False, object_randomization=False)
            settings = NamedRenderingPresets[rendering_preset]
            s = Simulator(mode='headless',
                          image_width=512,
                          image_height=512,
                          device_idx=0,
                          rendering_settings=settings,
                          physics_timestep=1/240.0
                          )
            s.import_ig_scene(scene)
            turtlebot = Turtlebot(config)
            s.import_robot(turtlebot)
            for mode in modality_list:
                for _ in range(10):
                    _ = s.renderer.render_robot_cameras(modes=(mode))
                start = time.time()
                for _ in range(200):
                    _ = s.renderer.render_robot_cameras(modes=(mode))
                end = time.time()
                fps = 200 / (end - start)
                result[(scene_name, rendering_preset, mode)] = fps
            s.disconnect()
        return result

def main():
    result = benchmark_rendering(
        ['Rs_int', 'Beechwood_1_int'],
        list(NamedRenderingPresets.keys()),
        list(AVAILABLE_MODALITIES)
    )
    print(result)
    plt.figure(figsize=(5,30))
    plt.tight_layout()
    plt.barh(["-".join(item) for item in result.keys()], result.values())
    plt.xlabdel('fps')
    plt.savefig('benchmark_rendering.pdf', bbox_inches = "tight")
    pkl.dump(result, open('rendering_benchmark_results.pkl', 'wb'))

if __name__ == "__main__":
    main()
