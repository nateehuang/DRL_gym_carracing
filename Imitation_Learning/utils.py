import numpy as np
from torchvision import transforms

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if np.allclose(a, [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif np.allclose(a, [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif np.allclose(a, [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif np.allclose(a, [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0

def id_to_action(id):
    if id == LEFT: return [-1.0, 0.0, 0.0]
    elif id == RIGHT: return [1.0, 0.0, 0.0]              
    elif id == ACCELERATE: return [0.0, 1.0, 0.0]      
    elif id == BRAKE: return [0.0, 0.0, 0.2]             
    else:       
        return [0.0, 0.0, 0.0]           

def data_preprocess(states):
    """
    Remove score and convert rgb to grayscale 
    """
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Pad((12, 12, 12, 0)),
        transforms.CenterCrop(84),
        transforms.ToTensor()
    ])

    transf_np = transf(states).numpy()
    return transf_np