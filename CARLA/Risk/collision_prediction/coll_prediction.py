"""
Collision Prediction using RoadScene2Vec Scene Graphs

This script processes visual scene data from real or CARLA-based driving scenarios
and uses a pretrained RoadScene2Vec model to assess collision risk. It involves:
- Extracting scene graphs from image sequences
- Formatting the data for graph-based deep learning
- Running inference through a pretrained GNN-based model 
- Outputting a binary classification: collision or no collision
- takes config files as input for relevent features and paths

Modules used:
- `roadscene2vec`: Scene graph extraction and learning pipeline
- `torch_geometric`: For handling GNN data structures
- `torch`: For model inference and output interpretation

Usage:
    Run as a script to get a risk prediction:
    $ python coll_prediction.py
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(sys.path[0]))
from Risk_assessment import roadscene2vec
from Risk_assessment.roadscene2vec.util.config_parser import configuration
import Risk_assessment.roadscene2vec.scene_graph.extraction.image_extractor as RealEx
from Risk_assessment.roadscene2vec.learning.util.scenegraph_trainer import Scenegraph_Trainer
from torch_geometric.data import Data, DataLoader

sys.modules['util'] = roadscene2vec.util



def format_use_case_model_input(sequence, trainer):
        if trainer.config.training_configuration["scenegraph_dataset_type"] == "carla":
            for seq in sequence.scene_graphs:
                data = {"sequence":trainer.scene_graph_dataset.process_carla_graph_sequences(sequence.scene_graphs[seq], feature_list = trainer.feature_list, folder_name = sequence.folder_names[0]) , "label":None, "folder_name": sequence.folder_names[0]}
        elif trainer.config.training_configuration["scenegraph_dataset_type"] == "real":
            for seq in sequence.scene_graphs:
                data = {"sequence":trainer.scene_graph_dataset.process_real_image_graph_sequences(sequence.scene_graphs[seq], feature_list = trainer.feature_list, folder_name = sequence.folder_names[0]) , "label":None, "folder_name": sequence.folder_names[0]}
        else:
            raise ValueError('output():scenegraph_dataset_type unrecognized')
        data = data['sequence']
        graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]  
        train_loader = DataLoader(graph_list, batch_size=len(graph_list))
        sequence = next(iter(train_loader)).to(trainer.config.model_configuration["device"])
        
        return (sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)    

def extract_seq(scenegraph_extraction_config):                                                                                          
    sg_extraction_object = RealEx.RealExtractor(scenegraph_extraction_config) #creating Real Image Preprocessor using config
    sg_extraction_object.load() #preprocesses sequences by extracting frame data for each sequence
    scenegraph_dataset = sg_extraction_object.getDataSet() #returned scenegraphs from extraction
    scenegraph_dataset.save() #save ScenegraphDataset
    return scenegraph_dataset #return ScenegraphDataset
    
def collison_pred():
    scenegraph_extraction_config = configuration(r"use_case_3_scenegraph_extraction_config.yaml",from_function = True) #create scenegraph extraction config object
    extracted_scenegraphs = extract_seq(scenegraph_extraction_config) #extracted scenegraphs for each frame for the given sequence  
    training_config = configuration(r"use_case_3_learning_config.yaml",from_function = True)  #create training config object                                                                                                 
    trainer = Scenegraph_Trainer(training_config) #create trainer object using config
    trainer.build_model()
    trainer.load_model() #load the proper model using the trainer
    #trainer.split_dataset() #split ScenegraphDataset specified in learning config into training, testing data
    #trainer.build_model() #build model specified in learning config
    #trainer.learn()
    model_input = format_use_case_model_input(extracted_scenegraphs, trainer) #turn extracted original sequence's extracted ScenegraphDataset into model input
    output, _ = trainer.model.forward(*model_input) #output risk assessment for the original sequence 
    return output   
    

if __name__ == "__main__":
    output = collison_pred()

    # Compute probabilities and prediction
    probs = torch.exp(output)
    confidence, predicted_class = torch.max(probs, dim=0)

    # Interpret result
    label_map = {0: "SAFE", 1: "UNSAFE"}  # You can extend this if needed
    prediction = label_map.get(predicted_class.item(), "UNKNOWN")
    confidence_percent = confidence.item() * 100

    # Print result
    print("\n[🧠 COLLISION PREDICTION]")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence_percent:.2f}%")
    
    
    
