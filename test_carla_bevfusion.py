import carla
import random
import numpy as np
import cv2
import torch
import copy
from queue import Queue
from queue import Empty

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from torchpack.utils.config import configs
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Helper to load config
def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def get_matrix(transform):
    """Converts Carla Transform to 4x4 Matrix"""
    m = transform.get_matrix()
    return np.array(m)

# ==============================================================================
# Classes
# ==============================================================================

class MetricEvaluator:
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.classes = [0] # Assuming only car for now
        
    def add(self, preds, gts):
        # preds: dict with 'boxes_3d', 'scores_3d', 'labels_3d'
        # gts: dict with 'boxes_3d', 'labels_3d'
        self.predictions.append(preds)
        self.ground_truths.append(gts)
        
    def compute_map(self, iou_thr=0.5):
        aps = []
        print(f"Computing mAP with IoU threshold {iou_thr}...")
        for cls_id in self.classes:
            all_preds = []
            all_gts = []
            
            for i in range(len(self.predictions)):
                p_mask = self.predictions[i]['labels_3d'] == cls_id
                p_boxes = self.predictions[i]['boxes_3d'][p_mask]
                p_scores = self.predictions[i]['scores_3d'][p_mask]
                
                g_mask = self.ground_truths[i]['labels_3d'] == cls_id
                g_boxes = self.ground_truths[i]['boxes_3d'][g_mask]
                
                all_preds.append({'boxes': p_boxes, 'scores': p_scores, 'frame_id': i})
                all_gts.append({'boxes': g_boxes, 'matched': np.zeros(len(g_boxes), dtype=bool)})
            
            flat_preds = []
            for p in all_preds:
                for j in range(len(p['boxes'])):
                    flat_preds.append({
                        'box': p['boxes'][j],
                        'score': p['scores'][j],
                        'frame_id': p['frame_id']
                    })
            
            flat_preds.sort(key=lambda x: x['score'], reverse=True)
            
            tp = np.zeros(len(flat_preds))
            fp = np.zeros(len(flat_preds))
            
            for i, p in enumerate(flat_preds):
                frame_id = p['frame_id']
                pred_box = torch.tensor(p['box'], device='cuda').unsqueeze(0) # 1x7
                gt_boxes = torch.tensor(all_gts[frame_id]['boxes'], device='cuda') # Nx7
                
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                
                iou = bbox_overlaps_3d(pred_box, gt_boxes, coordinate='lidar') # 1xN
                max_iou, max_idx = torch.max(iou, dim=1)
                max_iou = max_iou.item()
                max_idx = max_idx.item()
                
                if max_iou > iou_thr:
                    if not all_gts[frame_id]['matched'][max_idx]:
                        tp[i] = 1
                        all_gts[frame_id]['matched'][max_idx] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            num_gts = sum([len(g['boxes']) for g in all_gts])
            if num_gts == 0:
                ap = 0
            else:
                recalls = tp_cumsum / num_gts
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                ap = self.compute_ap(recalls, precisions)
            aps.append(ap)
            print(f"Class {cls_id} AP: {ap:.4f}")
            
        return np.mean(aps)

    def compute_ap(self, recalls, precisions):
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap

class BEVFusionWrapper:
    def __init__(self, config_path, checkpoint_path):
        print(f"Loading model from {config_path}...")
        configs.load(config_path, recursive=True)
        self.cfg = Config(recursive_eval(configs), filename=config_path)
        
        self.model = build_model(self.cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        self.model = self.model.cuda()
        self.model.eval()
        print("Model loaded.")

    def process_data(self, agent_data):
        # Prepare inputs for BEVFusion
        
        # 1. Images
        imgs = []
        img_metas = {}
        
        # nuScenes order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
        sensor_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        processed_imgs = []
        camera_intrinsics = []
        camera2ego = []
        lidar2camera = []
        lidar2image = []
        camera2lidar = []
        img_aug_matrix = []
        
        lidar_data = agent_data['LIDAR_TOP']

        ego_transform = agent_data['ego_transform']
        lidar_transform = lidar_data.transform
        
        ego_matrix = get_matrix(ego_transform)
        lidar_matrix = get_matrix(lidar_transform)
        
        lidar2ego_carla = np.linalg.inv(ego_matrix) @ lidar_matrix
        
        # Coordinate conversion for Lidar: CARLA (Y-right) -> Model (Y-left)
        T_C2M = np.eye(4)
        T_C2M[1, 1] = -1
        
        lidar2ego_model = T_C2M @ lidar2ego_carla @ np.linalg.inv(T_C2M)
        
        for name in sensor_names:
            sensor_data = agent_data[name]
            
            # Image processing
            array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (sensor_data.height, sensor_data.width, 4)) # BGRA
            img = array[:, :, :3] # BGR
            
            # Resize and Crop
            h, w = img.shape[:2]
            resize_scale = 0.48
            new_w = int(w * resize_scale)
            new_h = int(h * resize_scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            crop_h, crop_w = 256, 704
            start_h = (new_h - crop_h) // 2
            start_w = (new_w - crop_w) // 2
            img_cropped = img_resized[start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
            img_norm = (img_cropped - mean) / std
            
            processed_imgs.append(torch.tensor(img_norm).permute(2, 0, 1)) # C, H, W
            
            # Matrices
            fov = sensor_data.fov
            f = w / (2.0 * np.tan(fov * np.pi / 360.0))
            K = np.identity(3)
            K[0, 0] = K[1, 1] = f
            K[0, 2] = w / 2.0
            K[1, 2] = h / 2.0
            
            # Augmentation Matrix
            aug_mat = np.eye(4)
            aug_mat[0, 0] = resize_scale
            aug_mat[1, 1] = resize_scale
            aug_mat[0, 3] = -start_w
            aug_mat[1, 3] = -start_h
            img_aug_matrix.append(aug_mat)
            
            # Transforms
            cam_world_transform = get_matrix(sensor_data.transform)
            lidar_world_transform = get_matrix(lidar_data.transform)
            
            lidar2cam_carla = np.linalg.inv(cam_world_transform) @ lidar_world_transform
            
            # Coordinate conversion for Camera: CARLA -> Model
            R_C_C2M = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
            T_C_C2M = np.eye(4)
            T_C_C2M[:3, :3] = R_C_C2M
            
            l2c_model = T_C_C2M @ lidar2cam_carla @ np.linalg.inv(T_C2M)
            lidar2camera.append(l2c_model)
            camera2lidar.append(np.linalg.inv(l2c_model))
            
            K_4x4 = np.eye(4)
            K_4x4[:3, :3] = K
            camera_intrinsics.append(K_4x4)
            
            l2i = K_4x4 @ l2c_model
            lidar2image.append(l2i)
            
            c2e = lidar2ego_model @ np.linalg.inv(l2c_model)
            camera2ego.append(c2e)

        # 2. Lidar Processing
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4)).copy()
        points[:, 1] = -points[:, 1] # Flip Y
        
        # Transform to Ego Frame (Model Convention)
        # The model expects points in Ego Frame because the fusion happens in Ego Frame.
        points[:, 2] += lidar2ego_carla[2, 3]
        
        points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0) # Pad to 5 channels
        points_tensor = torch.tensor(points, dtype=torch.float32)
        
        imgs_tensor = torch.stack(processed_imgs).unsqueeze(0) # B, N, C, H, W
        
        metas = {
            'camera_intrinsics': torch.tensor(np.stack(camera_intrinsics), dtype=torch.float32).unsqueeze(0),
            'camera2ego': torch.tensor(np.stack(camera2ego), dtype=torch.float32).unsqueeze(0),
            'lidar2ego': torch.tensor(lidar2ego_model, dtype=torch.float32).unsqueeze(0),
            'lidar2camera': torch.tensor(np.stack(lidar2camera), dtype=torch.float32).unsqueeze(0),
            'lidar2image': torch.tensor(np.stack(lidar2image), dtype=torch.float32).unsqueeze(0),
            'camera2lidar': torch.tensor(np.stack(camera2lidar), dtype=torch.float32).unsqueeze(0),
            'img_aug_matrix': torch.tensor(np.stack(img_aug_matrix), dtype=torch.float32).unsqueeze(0),
            'lidar_aug_matrix': torch.eye(4, dtype=torch.float32).unsqueeze(0),
            'box_type_3d': LiDARInstance3DBoxes
        }
        
        return {
            'img': imgs_tensor.cuda(),
            'points': [points_tensor.cuda()],
            'metas': [metas],
            'img_metas': [metas],
            'camera2ego': metas['camera2ego'].cuda(),
            'lidar2ego': metas['lidar2ego'].cuda(),
            'lidar2camera': metas['lidar2camera'].cuda(),
            'lidar2image': metas['lidar2image'].cuda(),
            'camera_intrinsics': metas['camera_intrinsics'].cuda(),
            'camera2lidar': metas['camera2lidar'].cuda(),
            'img_aug_matrix': metas['img_aug_matrix'].cuda(),
            'lidar_aug_matrix': metas['lidar_aug_matrix'].cuda(),
            'depths': None,
            'lidar2ego_carla': lidar2ego_carla
        }

    def predict(self, inputs):
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        result = outputs[0]
        boxes = result['boxes_3d'].tensor.cpu().numpy()
        scores = result['scores_3d'].cpu().numpy()
        labels = result['labels_3d'].cpu().numpy()
        
        return boxes, scores, labels

class Visualizer:
    def __init__(self):
        pass

    def draw_3d_box_on_image(self, img, box, lidar2image, color=(0, 255, 0)):
        # box: [x, y, z, dx, dy, dz, rot]
        # Assumes box is in LiDARInstance3DBoxes format (z is Bottom Center)
        x, y, z_bottom, dx, dy, dz, rot = box[:7]
        
        # Convert Bottom Center to Geometric Center for corner calculation
        z_center = z_bottom + dz / 2
        
        # Corners relative to geometric center
        x_corners = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
        y_corners = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
        z_corners = [dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2]
        
        corners = np.vstack([x_corners, y_corners, z_corners]) # 3x8
        
        # Rotate
        c = np.cos(rot)
        s = np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = R @ corners
        
        # Translate (using geometric center)
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z_center
        
        # Homogeneous
        corners_hom = np.vstack([corners, np.ones((1, 8))]) # 4x8
        
        # Project
        corners_img = lidar2image @ corners_hom
        
        # Check if box is behind camera
        if np.any(corners_img[2, :] <= 0):
            return

        # Normalize
        corners_img = corners_img[:3, :]
        corners_img /= corners_img[2, :]
        corners_img = corners_img.T # 8x3

        def draw_line(i, j):
            p1 = (int(corners_img[i, 0]), int(corners_img[i, 1]))
            p2 = (int(corners_img[j, 0]), int(corners_img[j, 1]))
            cv2.line(img, p1, p2, color, 2)
            
        # Draw lines
        lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        for i, j in lines:
            draw_line(i, j)

    def generate_bev(self, points, boxes, gt_boxes=None, x_range=(-50, 50), y_range=(-50, 50), resolution=0.1):
        # Create canvas
        h = int((x_range[1] - x_range[0]) / resolution)
        w = int((y_range[1] - y_range[0]) / resolution)
        bev_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw points (points is N x 5)
        # Filter
        mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
               (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
        valid_points = points[mask]
        
        # Map to pixel
        # Image (0,0) is Top-Left.
        # World X is Up (Top). World Y is Left.
        # row = (x_max - x) / res
        # col = (y_max - y) / res
        
        x_img = ((x_range[1] - valid_points[:, 0]) / resolution).astype(np.int32)
        y_img = ((y_range[1] - valid_points[:, 1]) / resolution).astype(np.int32)
        
        # Clip
        x_img = np.clip(x_img, 0, h-1)
        y_img = np.clip(y_img, 0, w-1)
        
        bev_img[x_img, y_img] = (200, 200, 200)
        
        def draw_boxes(boxes_to_draw, color):
            for box in boxes_to_draw:
                x, y, z, dx, dy, dz, rot = box[:7]
                
                corners = np.array([
                    [dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2]
                ])
                
                # Rotate
                c = np.cos(rot)
                s = np.sin(rot)
                R = np.array([[c, -s], [s, c]])
                corners = corners @ R.T
                
                # Translate
                corners += np.array([x, y])
                
                # Map to pixel
                c_x = ((x_range[1] - corners[:, 0]) / resolution).astype(np.int32)
                c_y = ((y_range[1] - corners[:, 1]) / resolution).astype(np.int32)
                
                pts = np.stack([c_y, c_x], axis=1) # OpenCV uses (col, row) -> (x, y)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(bev_img, [pts], True, color, 2)
                
                # Draw direction (Front center)
                front_pt = np.array([dx/2, 0])
                front_pt = front_pt @ R.T + np.array([x, y])
                f_x = int((x_range[1] - front_pt[0]) / resolution)
                f_y = int((y_range[1] - front_pt[1]) / resolution)
                
                center_x = int((x_range[1] - x) / resolution)
                center_y = int((y_range[1] - y) / resolution)
                
                cv2.line(bev_img, (center_y, center_x), (f_y, f_x), (0, 0, 255), 2)

        # Draw GT boxes (Blue)
        if gt_boxes is not None:
            draw_boxes(gt_boxes, (255, 0, 0))

        # Draw Pred boxes (Green)
        draw_boxes(boxes, (0, 255, 0))
            
        return bev_img

class CarlaAgent:
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        
        self.sensors = []
        self.sensor_queues = {}
        self.sensor_data = {}
        self.ego_vehicle = None
        self.other_actors = []
        
        self.img_size = (1600, 900)
        
    def setup(self):
        # Clean up all existing vehicles to ensure a fresh start
        print("Cleaning up all existing vehicles...")
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        if len(all_vehicles) > 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in all_vehicles])

        # Configure Traffic Manager first
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(0)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.global_percentage_speed_difference(0.0)
        self.tm.set_respawn_dormant_vehicles(True)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(70.0)

        # Configure World
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        
        # Spawn Ego
        bp = self.bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        for transform in spawn_points:
            self.ego_vehicle = self.world.try_spawn_actor(bp, transform)
            if self.ego_vehicle:
                break
        
        if not self.ego_vehicle:
             raise RuntimeError("Could not spawn ego vehicle")

        self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        
        self.world.tick()
        self.spawn_traffic()
        self.spawn_sensors()
        self.world.tick()
        
    def spawn_traffic(self, num_vehicles=100):
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        ego_loc = self.ego_vehicle.get_location()
        spawn_points = [p for p in spawn_points if p.location.distance(ego_loc) > 10.0]
        
        vehicle_bps = self.bp_lib.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
        
        count = 0
        for transform in spawn_points:
            if count >= num_vehicles:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            vehicle = self.world.try_spawn_actor(bp, transform)
            if vehicle:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.other_actors.append(vehicle)
                count += 1
        print(f"Spawned {count} vehicles")

    def spawn_sensors(self):
        cam_configs = [
            {'name': 'CAM_FRONT', 'pos': (1.5, 0, 1.6), 'rot': (0, 0, 0)},
            {'name': 'CAM_FRONT_RIGHT', 'pos': (1.5, 0.5, 1.6), 'rot': (0, 55, 0)},
            {'name': 'CAM_FRONT_LEFT', 'pos': (1.5, -0.5, 1.6), 'rot': (0, -55, 0)},
            {'name': 'CAM_BACK', 'pos': (-1.5, 0, 1.6), 'rot': (0, 180, 0)},
            {'name': 'CAM_BACK_LEFT', 'pos': (-1.5, -0.5, 1.6), 'rot': (0, -110, 0)},
            {'name': 'CAM_BACK_RIGHT', 'pos': (-1.5, 0.5, 1.6), 'rot': (0, 110, 0)},
        ]
        
        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.img_size[0]))
        cam_bp.set_attribute('image_size_y', str(self.img_size[1]))
        cam_bp.set_attribute('fov', '70')
        
        for cfg in cam_configs:
            transform = carla.Transform(
                carla.Location(*cfg['pos']),
                carla.Rotation(pitch=cfg['rot'][0], yaw=cfg['rot'][1], roll=cfg['rot'][2])
            )
            sensor = self.world.spawn_actor(cam_bp, transform, attach_to=self.ego_vehicle)
            self.sensors.append(sensor)
            q = Queue()
            self.sensor_queues[sensor.id] = q
            sensor.listen(lambda data, q=q: q.put(data))
            self.sensor_data[cfg['name']] = {'sensor': sensor, 'transform': transform}

        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '1300000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.84), carla.Rotation())
        lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.sensors.append(lidar_sensor)
        q = Queue()
        self.sensor_queues[lidar_sensor.id] = q
        lidar_sensor.listen(lambda data, q=q: q.put(data))
        self.sensor_data['LIDAR_TOP'] = {'sensor': lidar_sensor, 'transform': lidar_transform}

    def get_data(self):
        try:
            self.world.tick()
        except RuntimeError as e:
            print(f"RuntimeError during tick: {e}")
            return None
        
        data = {}
        data['ego_transform'] = self.ego_vehicle.get_transform()
        for name, info in self.sensor_data.items():
            sensor = info['sensor']
            try:
                sensor_data = self.sensor_queues[sensor.id].get(timeout=2.0)
                data[name] = sensor_data
            except Empty:
                print(f"Timeout waiting for {name}")
                return None
        return data

    def get_ground_truth(self):
        # We need Actor -> LiDAR (Model Frame)
        # Actor (Carla World) -> LiDAR (Carla World) -> LiDAR (Model Frame)
        
        gt_boxes = []
        gt_labels = []
        
        # Get current LiDAR World Transform
        lidar_sensor = self.sensor_data['LIDAR_TOP']['sensor']
        lidar_world_transform = lidar_sensor.get_transform()
        lidar_world_matrix = get_matrix(lidar_world_transform)
        
        # T_C2M for flipping Y
        T_C2M = np.eye(4)
        T_C2M[1, 1] = -1
        
        for actor in self.other_actors:
            if not actor.is_alive:
                continue
                
            # 1. Get Actor Transform in Carla World
            actor_trans = actor.get_transform()
            actor_matrix = get_matrix(actor_trans)
            
            # 2. Apply Bounding Box Offset (to get center of box)
            # bb_loc is relative to actor origin
            bb_loc = actor.bounding_box.location
            offset_matrix = np.eye(4)
            offset_matrix[0, 3] = bb_loc.x
            offset_matrix[1, 3] = bb_loc.y
            offset_matrix[2, 3] = bb_loc.z
            
            # Center of box in World Frame
            actor_center_matrix = actor_matrix @ offset_matrix
            
            # 3. Actor Center -> LiDAR (Carla)
            # L_w @ T_a2l = A_w  => T_a2l = inv(L_w) @ A_w
            actor_in_lidar_carla = np.linalg.inv(lidar_world_matrix) @ actor_center_matrix
            
            # 4. Convert to Model Frame (Flip Y)
            # Pose_Model = T_C2M @ Pose_Carla @ inv(T_C2M)
            actor_in_lidar_model = T_C2M @ actor_in_lidar_carla @ np.linalg.inv(T_C2M)
            
            # Extract Box parameters
            # Location
            x, y, z = actor_in_lidar_model[:3, 3]
            
            # Filter by range (same as BEV grid)
            if x < -30 or x > 30 or y < -30 or y > 30:
                continue
            
            extent = actor.bounding_box.extent
            dx = extent.x * 2
            dy = extent.y * 2
            dz = extent.z * 2
            
            # Adjust Z to bottom center (LiDARInstance3DBoxes convention)
            # We are currently at the center of the box (due to offset_matrix)
            # So we subtract half height
            z = z - extent.z
            
            # Rotation
            # We need yaw in LiDAR Model Frame.
            rot_mat = actor_in_lidar_model[:3, :3]
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            
            gt_boxes.append([x, y, z, dx, dy, dz, yaw])
            gt_labels.append(0) # Default to Car
            
        if not gt_boxes:
            return np.zeros((0, 7)), np.zeros((0,))
            
        return np.array(gt_boxes), np.array(gt_labels)

    def cleanup(self):
        print("Cleaning up...")
        commands = []
        for sensor in self.sensors:
            if sensor.is_alive:
                commands.append(carla.command.DestroyActor(sensor))
        
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.set_autopilot(False)
            commands.append(carla.command.DestroyActor(self.ego_vehicle))
            
        for actor in self.other_actors:
            if actor.is_alive:
                commands.append(carla.command.DestroyActor(actor))
        
        if commands:
            self.client.apply_batch(commands)
        
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception as e:
            print(f"Error resetting settings: {e}")
        print("Cleanup done.")

# ==============================================================================
# Main
# ==============================================================================

def main():
    config_file = 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
    checkpoint_file = 'pretrained/bevfusion-det.pth'
    
    model_wrapper = BEVFusionWrapper(config_file, checkpoint_file)
    agent = CarlaAgent(host='172.17.0.1')
    visualizer = Visualizer()
    evaluator = MetricEvaluator()
    
    try:
        agent.setup()
        print("CARLA setup done. Starting loop...")
        
        frame_count = 0
        while True:
            if not agent.ego_vehicle.is_alive:
                print("Ego vehicle destroyed. Exiting.")
                break
                
            data = agent.get_data()
            if data is None:
                continue
                
            # Inference
            inputs = model_wrapper.process_data(data)
            boxes, scores, labels = model_wrapper.predict(inputs)
            
            # Coordinate Correction:
            # 1. Model predicts Geometric Center (Z_geo), but we need Bottom Center (Z_bottom).
            #    Z_bottom = Z_geo - Height / 2
            boxes[:, 2] -= boxes[:, 5] / 2
            
            # 2. Ego Frame -> LiDAR Frame
            lidar2ego_carla = inputs['lidar2ego_carla']
            lidar_z_offset = lidar2ego_carla[2, 3]
            boxes[:, 2] -= lidar_z_offset

            # 3. Fix Rotation (Yaw)
            boxes[:, 6] *= -1

            # Filter
            mask = scores > 0.3
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Get Ground Truth
            gt_boxes, gt_labels = agent.get_ground_truth()
            
            # Add to evaluator
            # Ensure boxes are 7-dim
            eval_boxes = boxes[:, :7] if boxes.shape[1] > 7 else boxes
            eval_gt_boxes = gt_boxes[:, :7] if gt_boxes.shape[1] > 7 else gt_boxes
            
            evaluator.add(
                {'boxes_3d': eval_boxes, 'scores_3d': scores, 'labels_3d': labels},
                {'boxes_3d': eval_gt_boxes, 'labels_3d': gt_labels}
            )
            
            print(f"Frame {frame_count}: Detected {len(boxes)} objects. GT: {len(gt_boxes)}")
            
            # Visualization
            # 1. Camera View
            front_cam_data = data['CAM_FRONT']
            array = np.frombuffer(front_cam_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (front_cam_data.height, front_cam_data.width, 4))
            vis_img = array[:, :, :3].copy()
            
            lidar2img = inputs['lidar2image'][0][0].cpu().numpy()
            
            # Draw GT (Blue)
            for box in gt_boxes:
                visualizer.draw_3d_box_on_image(vis_img, box, lidar2img, color=(255, 0, 0))
                
            # Draw Pred (Green)
            for box in boxes:
                visualizer.draw_3d_box_on_image(vis_img, box, lidar2img, color=(0, 255, 0))
            
            # 2. BEV View
            points = inputs['points'][0].cpu().numpy()
            bev_img = visualizer.generate_bev(points, boxes, gt_boxes=gt_boxes)
            
            # Combine and Display
            # Resize BEV to match height of Camera image or vice versa
            # Cam: 1600x900. BEV: 1000x1000 (if 100m range / 0.1 res)
            
            # Let's resize both to a reasonable display size
            display_h = 600
            scale_cam = display_h / vis_img.shape[0]
            cam_w = int(vis_img.shape[1] * scale_cam)
            vis_img_small = cv2.resize(vis_img, (cam_w, display_h))
            
            scale_bev = display_h / bev_img.shape[0]
            bev_w = int(bev_img.shape[1] * scale_bev)
            bev_img_small = cv2.resize(bev_img, (bev_w, display_h))
            
            combined = np.hstack([vis_img_small, bev_img_small])
            
            
            cv2.imwrite('vis_output.jpg', combined)
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
        # Compute Metrics
        mAP = evaluator.compute_map()
        print(f"Final mAP: {mAP:.4f}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
