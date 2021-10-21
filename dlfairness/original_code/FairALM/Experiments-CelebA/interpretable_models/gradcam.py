from PIL import Image
from matplotlib.pyplot import imshow, imsave
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import pdb
from misc_functions import get_example_params, save_class_activation_images
from resnet import resnet18
import os
import torch
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imname', type=str)
parser.add_argument('--modelname', type=str)
args = parser.parse_args()

imname = args.imname
if imname is None:
    print('No image name provided')
    exit(1)

class_idx = 1
dirname="./images/"
image = Image.open(dirname+imname + '.jpg').convert('RGB')
imname = 't1_s1_'+imname
modelname = args.modelname
if modelname is None:
    print('No model name')
    exit(1)

NUM_CLASSES = 2
GRAYSCALE = False


def deleteitems(checkpoint):
    known = ["bn1.num_batches_tracked", "layer1.0.bn1.num_batches_tracked", "layer1.0.bn2.num_batches_tracked", "layer1.1.bn1.num_batches_tracked", "layer1.1.bn2.num_batches_tracked", "layer2.0.bn1.num_batches_tracked", "layer2.0.bn2.num_batches_tracked", "layer2.0.downsample.1.num_batches_tracked", "layer2.1.bn1.num_batches_tracked", "layer2.1.bn2.num_batches_tracked", "layer3.0.bn1.num_batches_tracked", "layer3.0.bn2.num_batches_tracked", "layer3.0.downsample.1.num_batches_tracked", "layer3.1.bn1.num_batches_tracked", "layer3.1.bn2.num_batches_tracked", "layer4.0.bn1.num_batches_tracked", "layer4.0.bn2.num_batches_tracked", "layer4.0.downsample.1.num_batches_tracked", "layer4.1.bn1.num_batches_tracked", "layer4.1.bn2.num_batches_tracked"]
    for k in known:
        del checkpoint['net'][k]

def missingkeys(checkpoint):
    #pdb.set_trace()
    key_list = list(checkpoint['net'].keys())
    for k in key_list:
        new_k = '.'.join(k.split('.')[1:])
        checkpoint['net'][new_k] = checkpoint['net'][k].clone()
        del checkpoint['net'][k]
        
        
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor()])
    
tensor = custom_transform(image)

print('image_shape', tensor.shape)

prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

''' 
LOAD PRETRAINED MODEL
'''
model = resnet18(NUM_CLASSES, GRAYSCALE)

if modelname == 'N':#'NO_CONSTRAINTS':
    print('==> NO_CONSTRAINTS from checkpoint..')
    assert os.path.isdir('./model_store/savemodel_unconstrained/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./model_store/savemodel_unconstrained/ckpt.t7')
    model.load_state_dict(checkpoint['net'])
    print('Epoch: %d | train Acc:%.3f%% | train Fnr:%.3f%%' % (checkpoint['epoch'],
                                                               checkpoint['train_acc']['acc'],
                                                               checkpoint['train_acc']['fnr']))
    print('          | valid Acc:%.3f%% | valid Fnr:%.3f%%' % (checkpoint['val_acc']['acc'],
                                                               checkpoint['val_acc']['fnr']))
elif modelname == 'F': #'FAIR_ALM':
    #pdb.set_trace()
    print('==> FAIR_ALM from checkpoint..')
    assert os.path.isdir('./model_store/savemodel_fairalm/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./model_store/savemodel_fairalm/ckpt.t7')
    #deleteitems(checkpoint)
    model.load_state_dict(checkpoint['net'])
    print('Epoch: %d | train Acc:%.3f%% | train Fnr:%.3f%%' % (checkpoint['epoch'],
                                                               checkpoint['train_acc']['acc'],
                                                               checkpoint['train_acc']['fnr']))
    print('          | valid Acc:%.3f%% | valid Fnr:%.3f%%' % (checkpoint['val_acc']['acc'],
                                                               checkpoint['val_acc']['fnr']))
elif modelname == 'G':# 'GENDER_CLASSIFY':
    print('==> GENDER_CLASSIFY from checkpoint..')
    assert os.path.isdir('./model_store/savemodel_genderclassify/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./model_store/savemodel_genderclassify/ckpt.t7')
    missingkeys(checkpoint)
    model.load_state_dict(checkpoint['net'])
    print('Epoch: %d | train Acc:%.3f%%' % (checkpoint['epoch'], checkpoint['train_acc']))
    print('          | valid Acc:%.3f%%' % (checkpoint['val_acc']))
else:
    print('==> RESNET PRETRAINED MODEL')
    model = torchvision.models.resnet18(pretrained=True)
    
#pdb.set_trace()
model = model.cuda()
model.eval()

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

prediction, pred_probabilities = model(prediction_var)
pred_class_idx = topk(pred_probabilities,1)[1].int()
print('CLASS_IDX', class_idx)
#pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    #pdb.set_trace()
    _, nc, h, w = feature_conv.shape
    fc_fea = weight_fc[class_idx]
    fc_fea_sm = np.zeros(nc)
    j = -1
    for i in range(len(fc_fea)):
        if j % 4 == 0:
            j += 1
        fc_fea_sm[j] += fc_fea[i]
    #pdb.set_trace()
    cam = fc_fea_sm.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

overlay = getCAM(activated_features.features, weight_softmax, class_idx )

imnew = skimage.transform.resize(overlay[0], tensor.shape[1:3])

image = image.resize((tensor.shape[1], tensor.shape[2]), Image.ANTIALIAS)
print('imshape', image.size, 'imnewshape', imnew.shape)

imname = 'camon'+str(class_idx) + '_' + 'p' + str(pred_class_idx.item()) + '_' + imname
print('predicted_class', pred_class_idx.item())

save_class_activation_images(image, imnew, modelname + '_' + imname)    
