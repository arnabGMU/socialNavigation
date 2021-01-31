from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.objects.articulated_object import URDFObject
from gibson2.object_properties.utils import sample_kinematics
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import json
import random
import argparse

def main(args):
    scene_names = ['Beechwood_1_int','Benevolence_1_int','Ihlen_0_int','Merom_0_int','Pomaria_0_int','Pomaria_2_int',
                   'Wainscott_0_int','Beechwood_0_int','Benevolence_0_int','Benevolence_2_int','Ihlen_1_int','Merom_1_int',
                   'Pomaria_1_int','Rs_int','Wainscott_1_int']
    if args.scene_name not in scene_names:
        print('%s is not a valid scene name'%args.scene_name)
        return

    objects_to_sample = []
    object_id_dict = {}
    object_cat_dirs = {}
    with open(args.csv_name, 'r') as f:
        for line in f:
            parts = line.split(',')
            cat = parts[0]
            count = int(parts[1])

            object_cat_dir = 'data/ig_dataset/objects/%s'%(cat)
            if not os.path.isdir(object_cat_dir):
                print('%s is not a valid object'%(cat))
                return

            object_cat_dirs[cat] = object_cat_dir
            objects_to_sample.append((cat, count))

            object_ids = os.listdir(object_cat_dir)
            object_id_dict[cat] = object_ids

    settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
    s = Simulator(mode='headless', image_width=800,
                  image_height=800, rendering_settings=settings)
    simulator = s
    scene = InteractiveIndoorScene(args.scene_name, texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)
    renderer = s.renderer

    category_supporting_objects = {}
    for obj_name in scene.objects_by_name:
        obj = scene.objects_by_name[obj_name]
        if not obj.supporting_surfaces:
            continue
        cat = obj.category
        room = obj.in_rooms[0][:-2]
        if cat not in category_supporting_objects:
            category_supporting_objects[(cat,room)] = []
        category_supporting_objects[(cat,room)].append(obj)

    for category, count in objects_to_sample:
        ids = object_id_dict[category]
        for i in range(count):
            for attempt in range(args.num_attempts):
                object_id = random.choice(ids)
                urdf_path = '%s/%s/%s.urdf'%(object_cat_dirs[category], object_id, object_id)
                urdf_object = URDFObject(urdf_path)
                placement_rules_path = os.path.join(urdf_object.model_path, 'misc', 'placement_probs.json')
                with open(placement_rules_path, 'r') as f:
                    placement_rules = json.load(f)
                valid_placement_rules = {}
                for placement_rule in placement_rules.keys():
                    support_obj_cat, room, predicate = placement_rule.split('-')
                    if (support_obj_cat,room) in category_supporting_objects:
                        valid_placement_rules[placement_rule] = placement_rules[placement_rule]
                if len(valid_placement_rules) == 0:
                    continue
                simulator.import_object(urdf_object)
                placement_rule = random.choices(list(valid_placement_rules.keys()),
                                                weights=valid_placement_rules.values(), k=1)[0]
                support_obj_cat, room, predicate = placement_rule.split('-')
                support_objs = category_supporting_objects[(support_obj_cat,room)]
                chosen_support_obj = random.choice(support_objs)
                print('Sampling %s %s %s %s in %s'%(category, object_id, predicate, support_obj_cat,room))
                result = sample_kinematics(predicate, urdf_object, chosen_support_obj, True)
                if not result:
                    print('Failed kinematic sampling! Attempt %d'%attempt)
                    continue
                if args.save_images:
                    pos = urdf_object.get_position()
                    camera_pos = np.array([pos[0]-0.5, pos[1], pos[2]+0.5])
                    renderer.set_camera(camera_pose, pos, [0, 0, 1])
                    frame = renderer.render(modes=('rgb'))[0]
                    plt.imshow(frame)
                    plt.savefig('placement_imgs/overhead_%d.png'%step_num)
                    plt.close()


    s.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure which surfaces and containers in a scene an object might go in.')
    parser.add_argument('scene_name', type=str)
    parser.add_argument('csv_name', type=str)
    parser.add_argument('--num_attempts', type=int, default=10)
    parser.add_argument('--save_images', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
