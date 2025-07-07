import os
from io import BytesIO
from pathlib import Path
from pprint import pprint

from glob import glob
import json

import networkx as nx
import cv2
from PIL import Image
from networkx.drawing import nx_agraph, nx_pydot





from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction.image_extractor import RealExtractor
from roadscene2vec.scene_graph.extraction.carla_extractor import CarlaExtractor
from roadscene2vec.data.dataset import RawImageDataset
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

def elapsed_time(func, *args, **kwargs):
  start = timer()
  output = func(*args, **kwargs)
  end = timer()
  print(f'{end - start} seconds elapsed.')
  return output

def elapsed_time(func, *args, **kwargs):
  start = timer()
  output = func(*args, **kwargs)
  end = timer()
  print(f'{end - start} seconds elapsed.')
  return output

# Utilities
def get_extractor(config):
  return RealExtractor(config)
  
def get_carla_extractor(config):
  return CarlaExtractor(config)

def get_data(extractor):
  temp = RawImageDataset()
  temp.dataset_save_path = extractor.input_path
  return temp.load().data

def get_bev(extractor):
  return extractor.bev

def get_bbox(extractor, frame):
  return extractor.get_bounding_boxes(frame)

def get_scenegraph(extractor, bbox, bev):
  scenegraph = SceneGraph(extractor.relation_extractor,   
                          bounding_boxes=bbox, 
                          bev=bev,
                          coco_class_names=extractor.coco_class_names, 
                          platform=extractor.dataset_type)
  return scenegraph.g
  

def get_carla_scenegraph(extractor, frame_dict, frame):
  scenegraph = SceneGraph(extractor.relation_extractor, 
                          framedict = frame_dict, 
                          framenum = frame, 
                          platform = extractor.dataset_type)
  return scenegraph.g

def inspect_nodes(sg):
  for node in sg.nodes: print(node.name, end=' '); pprint(node.attr);

def inspect_relations(sg):
  for edge in sg.edges(data=True, keys=True): pprint(edge);

def yield_data(data):
  for sequence in data:
      for frame in data[sequence]:
        yield data[sequence][frame]

# Visualization
def cv2_color(frame):
  return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def draw_bbox(extractor, frame):
  return extractor.get_bounding_box_annotated_image(frame)

def draw_bev(bev, frame):
  img = bev.offset_image_height(frame)
  return bev.warpPerspective(img)
  
def draw_scenegraph_agraph(sg):
  # Not sure why this function cannot draw multi-edge connections to same node
  A = nx_agraph.to_agraph(sg) 
  A.layout('dot') 
  img = A.draw(format='png')
  return Image.open(BytesIO(img))

def draw_scenegraph_pydot(sg):
  
  sg = sg.copy()  # Don't modify the original graph
  
  # Rename nodes
  mapping = {}
  for n in sg.nodes:
      if isinstance(n, str) and ":" in n:
          new_name = n.replace(":", "_")
          mapping[n] = new_name
  if mapping:
      nx.relabel_nodes(sg, mapping, copy=False)

  # Fix node labels
  for n, data in sg.nodes(data=True):
      if "label" in data and ":" in data["label"]:
          data["label"] = data["label"].replace(":", "_")

  # Fix edge labels
  for u, v, key, data in sg.edges(keys=True, data=True):
      if "label" in data and ":" in data["label"]:
          data["label"] = data["label"].replace(":", "_")
  A = nx_pydot.to_pydot(sg)
  img = A.create_png()
  return Image.open(BytesIO(img))

def draw(extractor, frame, bbox, bev, sg, save_path=None):

#  frame = frame.transpose(1,2,0) #must do this for cv functionality due to change in real preprocessor
#  img = frame
  plt.subplot(2, 3, 1)
  plt.imshow(cv2_color(frame))
  plt.title("Raw Image")
  plt.axis('off')
  
  bbox_img = draw_bbox(extractor, frame)
  plt.subplot(2, 3, 2)
  plt.imshow(cv2_color(bbox_img))
  plt.title("Object Detection Image")
  plt.axis('off')
  
  bev_img = draw_bev(bev, frame)
  plt.subplot(2, 3, 3)
  plt.imshow(cv2_color(bev_img))
  plt.title("Bird's Eye Image")
  plt.axis('off')

  sg_img = draw_scenegraph_pydot(sg)
  plt.subplot(2, 1, 2)
  plt.imshow(sg_img)
  plt.title("SceneGraph Image")
  plt.axis('off')

  # This call is slow!
  if save_path is not None: 
    plt.savefig(save_path, dpi=600)

  plt.show()
  

def draw_carla(sg, image = None, save_path = None):

  sg_img = draw_scenegraph_pydot(sg)
  plt.subplot(1, 2, 1)
  plt.imshow(sg_img)
  plt.title("Scenegraph")
  plt.axis('off')
  
  if image is not None:
    plt.subplot(1, 2, 2)
    img = Image.open(image)
    plt.imshow(img)
    plt.title("Simulation Image")
    plt.axis('off')
  else:
    plt.subplot(1, 2, 2)
    img = Image.new(mode = "RGB", size = (200, 200),
                           color = (0, 0, 0))
    plt.imshow(img)
    plt.title("No Associated Simulation Image")
  plt.show()
 
 
  if save_path is not None: 
    plt.savefig(save_path, dpi=600)
  
def visualize(extraction_config):
  if extraction_config.dataset_type == "image":
    #visualize_real_image(extraction_config)
    visualize_camera_images_with_graphviz(extraction_config)
    #visualize_camera_images_networkx(extraction_config)
  elif extraction_config.dataset_type == "carla":
    visualize_carla(extraction_config)
  else:
    raise ValueError("Extraction dataset type not recognized")

def visualize_real_image(extraction_config):
  extractor = get_extractor(extraction_config)
  dataset_dir = extractor.conf.location_data["input_path"]
  if not os.path.exists(dataset_dir):
      raise FileNotFoundError(dataset_dir)
  all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
  all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0])) 
  print("test") 
  for path in tqdm(all_sequence_dirs):
      sequence = extractor.load_images(path)
      print ("path iteration")
      for frame in sorted(sequence.keys()):
          bbox = get_bbox(extractor, sequence[frame])
          bev = get_bev(extractor)
          sg = get_scenegraph(extractor, bbox, bev)
          print ("gonna save now")
          
          
    
          draw(extractor, sequence[frame], bbox, bev, sg, save_path='C:/Users/riahi/carla/CARLA_0.9.15/carla_latest/PythonAPI/examples/roadscene2vec/examples/output.png')
          #draw(extractor, sequence[frame], bbox, bev, sg, save_path='C:/Users/riahi/carla/CARLA_0.9.15/carla_latest/PythonAPI/examples/sequences/output.png')
  print('- finished')
  
  
def visualize_carla(extraction_config):
  extractor = get_carla_extractor(extraction_config)
  dataset_dir = extractor.conf.location_data["input_path"]
  if not os.path.exists(dataset_dir):
      raise FileNotFoundError(dataset_dir)
  all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
  all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0])) 
  for path in tqdm(all_sequence_dirs):
    txt_path = sorted(list(glob("%s/**/*.json" % str(path/"scene_raw"), recursive=True)))[0]
    raw_images_path = Path(path/"raw_images")
    raw_image_names = [str(i) for i in raw_images_path.iterdir() if i.is_file()]
    with open(txt_path, 'r') as scene_dict_f:
        try:
            framedict = json.loads(scene_dict_f.read()) 
            image_frames = list(framedict.keys()) #this is the list of frame names
            image_frames = sorted(image_frames)
            #### filling the gap between lane change where some of ego node might miss the invading lane information. ####
            start_frame_number = 0; end_frame_number = 0; invading_lane_idx = None
            
            for idx, frame_number in enumerate(image_frames):
                if "invading_lane" in framedict[str(frame_number)]['ego']:
                    start_frame_number = idx
                    invading_lane_idx = framedict[str(frame_number)]['ego']['invading_lane']
                    break
  
            for frame_number in image_frames[::-1]:
                if "invading_lane" in framedict[str(frame_number)]['ego']:
                    end_frame_number = image_frames.index(frame_number)
                    break
        
            for idx in range(start_frame_number, end_frame_number):
                framedict[str(image_frames[idx])]['ego']['invading_lane'] = invading_lane_idx
            
            for frame, frame_dict in framedict.items():
                if str(frame) in image_frames: 
                    sg = get_carla_scenegraph(extractor, frame_dict, frame)
                    image_file = [image_name for image_name in raw_image_names if str(frame) in image_name] #some frames do not have corresponding simulation images
                    if len(image_file) > 0:
                      image = Path(raw_images_path/image_file[0])
                    else:
                      image = None
                    draw_carla(sg, image, save_path='output.png')
        except:
          print("Issue visualizing carla scenegraphs")
  print('- finished')
  
'''
def visualize_camera_images_networkx(extraction_config):
    """
    Visualize camera images using NetworkX for the scene graph, saving each plot to disk.

    This function:
      - Creates a RealExtractor from `extraction_config`
      - Iterates over subfolders in `input_path`
      - For each image frame, builds a scene graph
      - Draws the raw image, bounding boxes, BEV, and
        the scene graph with NetworkX
      - Saves each figure as a .png (no interactive window)
    """

    extractor = RealExtractor(extraction_config)
    dataset_dir = extractor.conf.location_data["input_path"]
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Input path not found: {dataset_dir}")

    output_dir = Path("networkx_output")
    output_dir.mkdir(exist_ok=True)

    all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
    all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))

    print("[INFO] Starting NetworkX visualization...")

    for seq_path in tqdm(all_sequence_dirs, desc="Sequence Folders"):
        frames_dict = extractor.load_images(seq_path)
        print(f"[INFO] Processing {len(frames_dict)} frames in folder: {seq_path}")

        for frame_id in sorted(frames_dict.keys()):
            frame = frames_dict[frame_id]

            bbox = extractor.get_bounding_boxes(frame)
            print(f"[DEBUG] bbox type: {type(bbox)}")

            bbox = unify_vehicle_labels(bbox)
            bbox = suppress_duplicate_detections(bbox)

            bev = extractor.bev

            scene_graph = SceneGraph(
                extractor.relation_extractor,
                bounding_boxes=bbox,
                bev=bev,
                coco_class_names=extractor.coco_class_names,
                platform=extractor.dataset_type
            ).g

            plt.figure(figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(_cv2_color(frame))
            plt.title("Raw Image (NetworkX)")
            plt.axis('off')

            bbox_img = extractor.get_bounding_box_annotated_image(frame)
            plt.subplot(2, 2, 2)
            plt.imshow(_cv2_color(bbox_img))
            plt.title("Object Detection Image")
            plt.axis('off')

            bev_img = _draw_bev(bev, frame)
            plt.subplot(2, 2, 3)
            plt.imshow(_cv2_color(bev_img))
            plt.title("Bird’s Eye Image")
            plt.axis('off')

            plt.subplot(2, 2, 4)
            node_labels = nx.get_node_attributes(scene_graph, "label") 
            pos = nx.shell_layout(scene_graph)
            nx.draw(
                scene_graph, pos,
                with_labels=True,
                labels=node_labels,
                node_color='lightblue',
                edge_color='gray',
                node_size=800,
                font_size=8
            )

            G_simple = nx.DiGraph()
            for u, v, data in scene_graph.edges(data=True):
                if not G_simple.has_edge(u, v):
                    G_simple.add_edge(u, v, label=data.get('label', ''))

            edge_labels = nx.get_edge_attributes(G_simple, "label")
            nx.draw_networkx_edge_labels(
                scene_graph, pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=6,
                label_pos=0.5
            )
            plt.title("Scene Graph (NetworkX)")
            plt.axis('off')

            out_name = f"{seq_path.stem}_{frame_id}_networkx.png"
            out_path = output_dir / out_name
            plt.savefig(out_path, dpi=600)
            plt.close()

    print("[INFO] Visualization complete. Images saved to:", output_dir)


def _cv2_color(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _draw_bev(bev_obj, frame):
    img = bev_obj.offset_image_height(frame)
    return bev_obj.warpPerspective(img)

def unify_vehicle_labels(bboxes):
    vehicle_classes = {"truck", "motorcycle", "boat", "bus", "train", "car"}
    for box in bboxes:
        if box["label"].lower() in vehicle_classes:
            box["label"] = "car"
    return bboxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def suppress_duplicate_detections(detections, iou_threshold=0.6):
    filtered = []
    detections = sorted(detections, key=lambda d: d.get("score", 0), reverse=True)

    for i, det in enumerate(detections):
        keep = True
        for f in filtered:
            if compute_iou(det["bbox"], f["bbox"]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(det)

    return filtered


'''
def visualize_camera_images_networkx(extraction_config):
    """
    Visualize camera images using NetworkX for the scene graph, saving each plot to disk.

    This function:
      - Creates a RealExtractor from `extraction_config`
      - Iterates over subfolders in `input_path`
      - For each image frame, builds a scene graph
      - Draws the raw image, bounding boxes, BEV, and
        the scene graph with NetworkX
      - Saves each figure as a .png (no interactive window)
    """

    extractor = RealExtractor(extraction_config)
    dataset_dir = extractor.conf.location_data["input_path"]
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Input path not found: {dataset_dir}")

    # We'll store all generated images in a subfolder "networkx_output"
    output_dir = Path("networkx_output")
    output_dir.mkdir(exist_ok=True)

    # We expect subfolders like raw_images/0, raw_images/1, etc.
    all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
    all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))
    
    print("[INFO] Starting NetworkX visualization...")

    for seq_path in tqdm(all_sequence_dirs, desc="Sequence Folders"):
        frames_dict = extractor.load_images(seq_path)
        print(f"[INFO] Processing {len(frames_dict)} frames in folder: {seq_path}")

        for frame_id in sorted(frames_dict.keys()):
            frame = frames_dict[frame_id]

            # Extract bounding boxes
            bbox = extractor.get_bounding_boxes(frame)
            
            # Grab the BEV object
            bev = extractor.bev

            # Build the scene graph (NetworkX graph)
            scene_graph = SceneGraph(
                extractor.relation_extractor,
                bounding_boxes=bbox,
                bev=bev,
                coco_class_names=extractor.coco_class_names,
                platform=extractor.dataset_type
            ).g  # => networkx.MultiDiGraph

            # Create a figure
            plt.figure(figsize=(10, 8))

            # (A) Raw Image
            plt.subplot(2, 2, 1)
            plt.imshow(_cv2_color(frame))
            plt.title("Raw Image (NetworkX)")
            plt.axis('off')

            # (B) BBox overlay
            bbox_img = extractor.get_bounding_box_annotated_image(frame)
            plt.subplot(2, 2, 2)
            plt.imshow(_cv2_color(bbox_img))
            plt.title("Object Detection Image")
            plt.axis('off')

            # (C) Bird’s-Eye View
            bev_img = _draw_bev(bev, frame)
            plt.subplot(2, 2, 3)
            plt.imshow(_cv2_color(bev_img))
            plt.title("Bird’s Eye Image")
            plt.axis('off')

            # (D) Scene Graph (NetworkX)
            plt.subplot(2, 2, 4)
            node_labels = nx.get_node_attributes(scene_graph, "label") 
            pos = nx.circular_layout(scene_graph)
            #pos = nx.spring_layout(scene_graph, k=0.5, iterations=20)
            nx.draw(
                scene_graph, pos,
                with_labels=True,
                labels=node_labels,
                node_color='lightblue',
                edge_color='gray',
                node_size=800,
                font_size=8
            )
            
            #G_simple = nx.DiGraph()
            #for u, v, data in scene_graph.edges(data=True):
            #  if not G_simple.has_edge(u, v):
            #      G_simple.add_edge(u, v, label=data.get('label', ''))
                  
                  
            G_simple = nx.DiGraph()
            for u, v, data in scene_graph.edges(data=True):
                label = data.get('label', '')
                if G_simple.has_edge(u, v):
                    G_simple[u][v]['label'] += f", {label}"
                else:
                    G_simple.add_edge(u, v, label=label)




            edge_labels = nx.get_edge_attributes(G_simple, "label")
            nx.draw_networkx_edge_labels(
                scene_graph, pos,
                #G_simple, pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=6,
                label_pos=0.5
            )
            plt.title("Scene Graph (NetworkX)")
            plt.axis('off')

            # -------------------------------------
            # Save instead of show
            # -------------------------------------
            # We'll create a unique filename for each frame.
            out_name = f"{seq_path.stem}_{frame_id}_networkx.png"
            out_path = output_dir / out_name
            plt.savefig(out_path, dpi=600)
            plt.close()  # close figure to save memory

    print("[INFO] Visualization complete. Images saved to:", output_dir)


# ------------------------------------------------
#  Internal helper functions used above
# ------------------------------------------------

def _cv2_color(frame):
    """
    Convert BGR to RGB for proper display in Matplotlib.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _draw_bev(bev_obj, frame):
    """
    Apply the offset + warpPerspective transform 
    to get the bird’s-eye view image.
    """
    img = bev_obj.offset_image_height(frame)
    return bev_obj.warpPerspective(img)



import os
from pathlib import Path
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import graphviz

# … your other imports (RealExtractor, SceneGraph, _cv2_color, _draw_bev) …
def visualize_camera_images_with_graphviz(extraction_config):
    import graphviz
    import matplotlib.pyplot as plt
    import cv2
    from pathlib import Path
    from tqdm import tqdm

    extractor = RealExtractor(extraction_config)
    dataset_dir = extractor.conf.location_data["input_path"]
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Input path not found: {dataset_dir}")

    output_dir = Path("graphviz_output")
    output_dir.mkdir(exist_ok=True)

    all_seq_dirs = sorted(
        (d for d in Path(dataset_dir).iterdir() if d.is_dir()),
        key=lambda x: int(x.stem.split('_')[0])
    )

    print("[INFO] Starting Graphviz visualization...")
    for seq_path in tqdm(all_seq_dirs, desc="Sequence Folders"):
        frames = extractor.load_images(seq_path)
        print(f"[INFO] Processing {len(frames)} frames in {seq_path}")

        for frame_id in sorted(frames):
            frame = frames[frame_id]
            bbox = extractor.get_bounding_boxes(frame)
            bev = extractor.bev

            scene_graph = SceneGraph(
                extractor.relation_extractor,
                bounding_boxes=bbox,
                bev=bev,
                coco_class_names=extractor.coco_class_names,
                platform=extractor.dataset_type
            ).g  # This is a networkx.MultiDiGraph

            # --- Subplots ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            axes[0].imshow(_cv2_color(frame))
            axes[0].set_title("Raw Image")
            axes[0].axis('off')

            bbox_img = extractor.get_bounding_box_annotated_image(frame)
            axes[1].imshow(_cv2_color(bbox_img))
            axes[1].set_title("Detection")
            axes[1].axis('off')

            bev_img = _draw_bev(bev, frame)
            axes[2].imshow(_cv2_color(bev_img))
            axes[2].set_title("Bird’s-Eye View")
            axes[2].axis('off')

            # --- Graphviz scene graph (multi-edge, stable engine) ---
            dot = graphviz.Digraph(format='png', engine='neato', strict=False)
            dot.attr(dpi='300')
            dot.attr('graph', rankdir='LR', splines='true', overlap='false', ranksep='1.4')
            dot.attr('node', shape='ellipse', style='filled', fontname='Helvetica', fontsize='10')
            dot.attr('edge', fontname='Helvetica', fontsize='9', arrowsize='0.7',
                     labeldistance='1.5', labelfloat='true')  # improves label spacing

            # Add nodes
            for n, data in scene_graph.nodes(data=True):
                label = data.get('label', str(n)).replace(":", "_")
                color = 'lightblue' if 'car' in label.lower() else 'lightgreen' if 'person' in label.lower() else 'white'
                dot.node(str(n).replace(":", "_"), label=label, fillcolor=color)

            # Add multi-edges with labels
            for u, v, key, data in scene_graph.edges(keys=True, data=True):
                lbl = data.get('label', '')
                dot.edge(str(u).replace(":", "_"),
                         str(v).replace(":", "_"),
                         label=lbl.replace(":", "_"),
                         decorate='true')  # helps center the label

            # Render and display
            gv_basename = f"{seq_path.stem}_{frame_id}_scenegraph"
            gv_path = output_dir / gv_basename
            dot.render(filename=str(gv_path), cleanup=True)

            graph_img = cv2.imread(str(gv_path) + ".png")
            graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB)
            axes[3].imshow(graph_img)
            axes[3].set_title("Scene Graph (Graphviz)")
            axes[3].axis('off')

            out_png = output_dir / f"{seq_path.stem}_{frame_id}_combined.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=300)
            plt.close(fig)

    print(f"[INFO] Done! Check your PNGs in {output_dir}")
