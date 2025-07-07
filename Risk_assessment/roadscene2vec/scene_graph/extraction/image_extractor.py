import os
import pdb
from pathlib import Path
import sys

detectron2_path = "C:/Users/riahi/carla/CARLA_0.9.15/carla_latest/PythonAPI/examples>"
sys.path.append(detectron2_path)


import cv2
from os.path import isfile, join
import roadscene2vec.data.dataset as ds
from roadscene2vec.scene_graph.extraction.extractor import Extractor as ex
from roadscene2vec.scene_graph.scene_graph import SceneGraph

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils import visualizer 
from detectron2.config import get_cfg
from detectron2 import model_zoo
from roadscene2vec.scene_graph.extraction.bev import bev
from tqdm import tqdm
#import traceback

'''RealExtractor initializes relational settings and creates an ImageSceneGraphSequenceGenerator object to extract scene graphs using raw image data.'''
class RealExtractor(ex):
    def __init__(self, config):
        super(RealExtractor, self).__init__(config) 

        self.input_path = self.conf.location_data['input_path']
        #self.input_path = self.conf.location_data['C:/Users/riahi/carla/CARLA_0.9.15/carla_latest/PythonAPI/examples/sequences']
        self.dataset = ds.SceneGraphDataset(self.conf)

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(self.input_path)

        # detectron setup
        model_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_path))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.coco_class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('thing_classes')
        self.predictor = DefaultPredictor(self.cfg)

        # bev setup
        self.bev = bev.BEV(config.image_settings['BEV_PATH'], mode='deploy')
        
        self.depth_dir = r"C:\Users\riahi\carla\CARLA_0.9.15\carla_latest\PythonAPI\examples\roadscene2vec\examples\raw_images"


    '''Load scenegraphs using raw image frame tensors'''
    def load(self): #seq_tensors[seq][frame/jpgname] = frame tensor
        try:
            print("Debug: self.input_path =", self.input_path)
            #pdb.set_trace()  # <---- add here, so you can step line by line

            all_sequence_dirs = [x for x in Path(self.input_path).iterdir() if x.is_dir()]
            for folder in all_sequence_dirs:
                print(f"Debug: Found directory '{folder.name}'")
            all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  
            self.dataset.folder_names = [path.stem for path in all_sequence_dirs]
            for path in tqdm(all_sequence_dirs):
                seq = int(path.stem.split('_')[0])
                label_path = (path/"label.txt").resolve()
                ignore_path = (path/"ignore.txt").resolve()
                if ignore_path.exists(): #record ignored sequences, and only load the sequences that were not ignored
                    with open(str(path/"ignore.txt"), 'r') as label_f:
                        ignore_label = int(label_f.read())
                        if ignore_label:
                            self.dataset.ignore.append(seq)
                            continue #skip to next seq if ignore path exists
                seq_images = self.load_images(path)
            
                self.dataset.scene_graphs[seq] = {}
                for frame, img in seq_images.items():
                    out_img_path = None
                    
                    #depth_path = os.path.join(path, f"{frame:04d}.raw")
                    #depth_path = os.path.join(self.depth_dir, f"{frame:04d}.npy")

                    
                    #print(f"depth path {depth_path}")
                    
                    #if os.path.exists(depth_path):
                    #    depth = self.load_depth_data(depth_path)
                    #    print(f"Loaded depth for frame {frame}: shape = {depth.shape}")
                    #else:
                    #    print(f"Warning: depth file for frame {frame} not found.")

                    
                                        # Load lidar and depth data for the current frame
                    #lidar = self.load_lidar_points_for_frame(path, frame)
                    #depth = self.load_depth_map_for_frame(path, frame)

                    # Create a new BEV instance using lidar and depth data
                    #self.bev = bev.BEV(use_lidar=True, lidar_points=lidar, depth_map=depth)
                    
                    bounding_boxes = self.get_bounding_boxes(img_tensor=img, out_img_path=out_img_path)
                    
                    
                    
                    scenegraph = SceneGraph(self.relation_extractor,    
                                                bounding_boxes = bounding_boxes, 
                                                bev = self.bev,
                                                coco_class_names=self.coco_class_names, 
                                                platform=self.dataset_type)
                                                #depth_points=depth)

                    self.dataset.scene_graphs[seq][frame] = scenegraph
                self.dataset.action_types[seq] = "lanechange" 
                if label_path.exists():
                    with open(str(path/'label.txt'), 'r') as label_file:
                        lines = label_file.readlines()
                        l0 = 1.0 if float(lines[0].strip().split(",")[0]) >= 0 else 0.0 
                        self.dataset.labels[seq] = l0

        except Exception as e:
            pdb.set_trace()
            import traceback
            print('We have problem creating the real image scenegraphs')
            
            #print(e)
            traceback.print_exc()
            print(e)
    
    #returns a numpy array representation of a sequence of images in format (H,W,C)
    def load_images(self, path):
        raw_images_loc = (path).resolve()#/'raw_images').resolve()
        #raw_images_loc = (path/'raw_images').resolve()
        images = sorted([Path(f) for f in os.listdir(raw_images_loc) if isfile(join(raw_images_loc, f)) and ".DS_Store" not in f and "Thumbs" not in f], key = lambda x: int(x.stem.split(".")[0]))
        images = [join(raw_images_loc,i) for i in images] 
        sequence_tensor = {}
        modulo = 0
        acc_number = 0
        if(self.framenum != None):
            modulo = int(len(images) / self.framenum)  #subsample to frame limit
        if(self.framenum == None or modulo == 0):
            modulo = 1
        for i in range(0, len(images)):
            if (i % modulo == 0 and self.framenum == None) or (i % modulo == 0 and acc_number < self.framenum):
                image_path = images[i]
                frame_num = int(Path(image_path).stem)
                im = cv2.imread(str(image_path), cv2.IMREAD_COLOR) 
                sequence_tensor[frame_num] = im 
                acc_number += 1
        return sequence_tensor
        
    def get_bounding_box_annotated_image(self, im):
        v = visualizer.Visualizer(im[:, :, ::-1], 
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
            scale=1.2)
        out = v.draw_instance_predictions(self.predictor(im)['instances'].to('cpu'))
        return out.get_image()[:, :, ::-1]
            
    def get_bounding_boxes(self, img_tensor, out_img_path=None):
        im = img_tensor
        outputs = self.predictor(im)
        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            out = self.get_bounding_box_annotated_image(im)
            cv2.imwrite(out_img_path, out)

        # todo: after done scp to server
        # crop im to remove ego car's hood
        # find threshold then remove from pred_boxes, pred_classes, check image_size
        bounding_boxes = outputs['instances'].pred_boxes, outputs['instances'].pred_classes, outputs['instances'].image_size
        return bounding_boxes

    
    '''Returns SceneGraphDataset object containing scengraphs, labels, and action types'''
    def getDataSet(self):
        try:
            return self.dataset
        except Exception as e:
            import traceback
            print('We have problem creating scenegraph dataset object from the extracted real image scenegraphs')
            print(e)
            traceback.print_exc()
            
    import numpy as np

    def load_depth_data(self, depth_filename, height=480, width=640):
        """
        Load a raw depth file saved from CARLA and return it as a 2D NumPy array.

        Parameters:
            depth_filename (str): Path to the .raw file.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            depth_image (np.ndarray): 2D array of depth values (shape: [height, width]).
        """
        
        '''
        # Read the raw depth data from file as float32
        depth_data = np.fromfile(depth_filename, dtype=np.float32)

        # Make sure the data matches the expected size
        if depth_data.size != height * width:
            raise ValueError(f"Depth data size mismatch! Expected {height * width}, got {depth_data.size}")

        # Reshape to image dimensions
        depth_image = depth_data.reshape((height, width))
        '''
        depth_image = np.load(depth_filename)
        #print (f"depth main data {depth_image}")

        return depth_image
    


    def load_lidar_points_for_frame(self, path, frame):
        """
        Loads 2D LiDAR points for a given frame.

        Args:
            path (str): Path to the sequence folder.
            frame (str or int): Frame index or name.

        Returns:
            np.ndarray: Array of shape (N, 2) containing [x, y] LiDAR points.
        """
        # Convert frame number to string if it's int
        frame_str = str(frame).zfill(3)  # e.g., 3 → "003"
        
        # Construct file path (adjust the filename format if needed)
        lidar_file_npy = os.path.join(path, f"lidar_frame_{frame_str}.npy")
        lidar_file_txt = os.path.join(path, f"lidar_frame_{frame_str}.txt")

        if os.path.exists(lidar_file_npy):
            lidar_data = np.load(lidar_file_npy)  # shape: (N, 2)
        elif os.path.exists(lidar_file_txt):
            lidar_data = np.loadtxt(lidar_file_txt, delimiter=',')  # adjust delimiter if needed
        else:
            raise FileNotFoundError(f"No LiDAR data found for frame {frame_str} at {lidar_file_npy} or {lidar_file_txt}")

        # Optional: make sure it's 2D (x, y)
        if lidar_data.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2) for 2D LiDAR, got {lidar_data.shape}")

        return lidar_data

            
            
from scipy.spatial import cKDTree
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

class RealExtractor2:
    def __init__(self, depth_scale=1.0, camera_intrinsics=None, camera_offset=(2.5, 0, 1)):
        # Load Detectron2 model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

        self.depth_scale = depth_scale  # Scale from raw depth image to meters
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 600.0,
            'fy': 600.0,
            'cx': 320.0,
            'cy': 240.0
        }
        
        # Offsets for camera and lidar positions
        self.camera_offset = np.array(camera_offset)  # (x, y, z)
        #self.lidar_offset = np.array(lidar_offset)    # (x, y, z)

    def load_depth(self, path):
        depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {path}")
        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]  # use single channel
        return depth_img.astype(np.float32) * self.depth_scale

    def project_pixel_to_3d(self, u, v, depth):
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    def get_lidar_pointcloud(self, lidar_path):
        try:
            return np.load(lidar_path)[:, :3]  # shape (N, 3)
        except:
            print(f"[WARNING] Could not load lidar file: {lidar_path}")
            return np.empty((0, 3))

    def estimate_object_depth(self, bbox, depth_img):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        roi = depth_img[y1:y2, x1:x2]
        valid = roi[roi > 0]
        return np.median(valid) if valid.size else 0.0

    def get_nearest_lidar_distance(self, object_3d_pos, lidar_points):
        if lidar_points.size == 0:
            return None
        tree = cKDTree(lidar_points)
        distance, _ = tree.query(object_3d_pos)
        return distance

    def apply_transformation(self, lidar_points, object_pos_3d):
        # Apply the transformation based on LiDAR and camera offsets
        # Transform LiDAR points
        lidar_points_transformed = lidar_points - self.camera_offset
        # Transform object position based on camera offset
        object_pos_3d_transformed = object_pos_3d
        return lidar_points_transformed, object_pos_3d_transformed

    def extract_features(self, rgb_img_path, depth_img_path, lidar_path):
        img = cv2.imread(rgb_img_path)
        depth_img = self.load_depth(depth_img_path)
        lidar_points = self.get_lidar_pointcloud(lidar_path)

        outputs = self.predictor(img)
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        features = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            class_id = int(classes[i])
            score = float(scores[i])

            # Get center pixel of bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            # Get depth from RGBD image
            object_depth = self.estimate_object_depth(bbox, depth_img)

            # Convert to 3D position
            object_pos_3d = self.project_pixel_to_3d(u, v, object_depth)

            # Apply the transformations based on camera and LiDAR offsets
            lidar_points_transformed, object_pos_3d_transformed = self.apply_transformation(lidar_points, object_pos_3d)

            # Refine with LiDAR
            lidar_distance = self.get_nearest_lidar_distance(object_pos_3d_transformed, lidar_points_transformed)
            if lidar_distance and abs(lidar_distance - object_depth) > 0.3:
                object_pos_3d_transformed = object_pos_3d_transformed * (lidar_distance / (np.linalg.norm(object_pos_3d_transformed) + 1e-5))

            features.append({
                "class": class_id,
                "score": score,
                "bbox": bbox.tolist(),
                "position": object_pos_3d_transformed.tolist(),  # (x, y, z)
            })

        return features
