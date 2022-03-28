import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms          
import torchvision.models as models
import torch 
import torch.nn as nn
from PIL import Image
import json
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.models as models


parser = argparse.ArgumentParser(description='Neural correlation')
parser.add_argument('--session', type=str)
parser.add_argument('--model_list',nargs="+")
parser.add_argument('--neuro_wise')
parser.add_argument('--model_name')
args = parser.parse_args()
session_name=args.session
neuro_wise=args.neuro_wise
model_type_list=args.model_list
model_name=args.model_name
print(args.neuro_wise)

device='cpu'
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
      
        yield iterable[ndx:min(ndx + n, l)]
        
        
        
#get activation for natural images
import json
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.models as models

for model_type in model_type_list:
  if model_type=="simclr":
    # load checkpoint for simclr
    checkpoint = torch.load('/content/gdrive/MyDrive/resnet50-1x.pth')
    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(checkpoint['state_dict'])
    # preprocess images for simclr
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
    ])

  if model_type=="simclr_v2_0":
    # load checkpoint for simclr
    checkpoint = torch.load('/content/gdrive/MyDrive/r50_1x_sk0.pth')
    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(checkpoint['resnet'])
    # preprocess images for simclr
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
    ])

  if model_type=="moco":
    # load checkpoints of moco
    state_dict = torch.load('/content/gdrive/MyDrive/moco/moco_v1_200ep_pretrain.pth.tar',map_location=torch.device('cpu'))['state_dict']
    resnet = models.resnet50(pretrained=False)
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') :
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #preprocess for moco
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type.split('_')[0]=="moco101":
    # load checkpoints of moco
    epoch_num=model_type.split('_')[1]
    state_dict = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/moco101/moco_{epoch_num}.pth.tar',map_location=torch.device('cpu'))['state_dict']
    resnet = models.resnet50(pretrained=False)
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') :
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #preprocess for moco
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="mocov2":
    # load checkpoints of mocov2
    state_dict = torch.load('/content/gdrive/MyDrive/moco/moco_v2_200ep_pretrain.pth.tar',map_location=torch.device('cpu'))['state_dict']
    resnet = models.resnet50(pretrained=False)
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #preprocess for mocov2
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="InsDis":
    # load checkpoints for instance recoginition resnet
    resnet=models.resnet50(pretrained=False)
    state_dict = torch.load('/content/gdrive/MyDrive/moco/lemniscate_resnet50_update.pth',map_location=torch.device('cpu') )['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module') and not k.startswith('module.fc'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #preprocess for instance recoginition resnet
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="place365_rn50":
    # load checkpoints for place365 resnet
    resnet=models.resnet50(pretrained=False)
    state_dict = torch.load('/content/gdrive/MyDrive/resnet50_places365.pth.tar',map_location=torch.device('cpu') )['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module') and not k.startswith('module.fc'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #preprocess for place365-resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="resnext101":
    #load ResNeXt 101_32x8 imagenet trained model
    resnet=models.resnext101_32x8d(pretrained=True)
    #preprocess for resnext101
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="untrained_resnet50":
    # load checkpoint for st resnet
    resnet=models.resnet50(pretrained=False)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="untrained_resnet101":
    # load checkpoint for st resnet
    resnet=models.resnet101(pretrained=False)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="untrained_wrn50":
    # load checkpoint for st resnet
    resnet=models.wide_resnet50_2(pretrained=False)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="untrained_wrn101":
    # load checkpoint for st resnet
    resnet=models.wide_resnet101_2(pretrained=False)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="wsl_resnext101":
    # load wsl resnext101
    resnet= models.resnext101_32x8d(pretrained=False)
    checkpoint = torch.load("/content/gdrive/MyDrive/resent_wsl/ig_resnext101_32x8-c38310e5.pth")
    resnet.load_state_dict(checkpoint)
    #preprocess for wsl resnext101
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="st_resnet":
    # load checkpoint for st resnet
    resnet=models.resnet50(pretrained=True)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="resnet101":
    # load checkpoint for st resnet
    resnet=models.resnet101(pretrained=True)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="wide_resnet50":
    # load checkpoint for st resnet
    resnet=models.wide_resnet50_2(pretrained=True)
    #preprocess for st_resnet50
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="wide_resnet101":
    # load checkpoint for st resnet
    resnet=models.wide_resnet101_2(pretrained=True)
    #preprocess for st_resnet101
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="alexnet":
    # load checkpoint for st alexnet
    alexnet=models.alexnet(pretrained=True)
    #preprocess for alexnet
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

  if model_type=="clip":
    # pip install git+https://github.com/openai/CLIP.git
    import clip
    resnet, preprocess = clip.load("RN50")

  if model_type=='linf_8':
    # pip install robustness
    resnet = torch.load('/content/gdrive/MyDrive/imagenet_linf_8_model.pt') # https://drive.google.com/file/d/1DRkIcM_671KQNhz1BIXMK6PQmHmrYy_-/view?usp=sharing
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])


  if model_type=='linf_4':
    # pip install robustness
    resnet = torch.load('/content/gdrive/MyDrive/robust_resnet.pt')#https://drive.google.com/file/d/1_tOhMBqaBpfOojcueSnYQRw_QgXdPVS6/view?usp=sharing
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])


  if model_type=='l2_3':
    # pip install robustness
    resnet = torch.load('/content/gdrive/MyDrive/imagenet_l2_3_0_model.pt') # https://drive.google.com/file/d/1SM9wnNr_WnkEIo8se3qd3Di50SUT9apn/view?usp=sharing 
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1'or model_type=='resnet50_l2_eps0.05':
    # pip install git+https://github.com/HelenR6/robustness
    from robustness.datasets import CIFAR,ImageNet
    from robustness.model_utils import make_and_restore_model
    ds = ImageNet('/tmp')
    resnet, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                resume_path=f'/content/gdrive/MyDrive/model_checkpoints/{model_type}.ckpt')
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="resnet_30"  or model_type=="resnet_60" or model_type=="resnet_90" or  model_type=="resnet_0" or  model_type=="resnet_10" or  model_type=="resnet_20" or  model_type=="resnet_40" or  model_type=="resnet_50" or  model_type=="resnet_60" or  model_type=="resnet_70" or  model_type=="resnet_80" or  model_type=="resnet_90":
    resnet=models.resnet50(pretrained=False)
    model_epoch=model_type.split('_')[1]
    checkpoint = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/{model_epoch}_model_best.pth.tar',map_location=torch.device('cpu') )
    state_dict=checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.') :

            state_dict[k[len('module.'):]] = state_dict[k]
        del state_dict[k]
    resnet.load_state_dict(state_dict)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
  if model_type=="v_resnet_60" or model_type=="v_resnet_0" or  model_type=="v_resnet_30"  or  model_type=="v_resnet_90" or  model_type=="v_resnet_10" or  model_type=="v_resnet_20" or  model_type=="v_resnet_40" or  model_type=="v_resnet_50" or  model_type=="v_resnet_60" or  model_type=="v_resnet_70" or  model_type=="v_resnet_80":
    resnet=models.resnet50(pretrained=False)
    epoch_num=model_type.split('_')[2]
#     if model_type=="v_resnet_90":
#       checkpoint = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/model_best.pth.tar',map_location=torch.device('cpu') )
#     else:
    checkpoint = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/model_epoch{epoch_num}.pth.tar',map_location=torch.device('cpu') )
    state_dict=checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.') :

            state_dict[k[len('module.'):]] = state_dict[k]
        del state_dict[k]
    resnet.load_state_dict(state_dict)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
#   if model_type=="v_resnet_0"  :
#     resnet=models.resnet50(pretrained=False)
#     preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225])
#     ])
    
    
    

  from PIL import Image
  session_path=args.session.replace('_','/')
  final_path=session_path[:-1]+'_'+session_path[-1:]
  f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
  natural_data = f['images/naturalistic'][:]
  synth_data=f['images/synthetic/monkey_'+final_path][:]
  print(natural_data.shape)

  x = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in natural_data])
  natural_perm_tensor=torch.tensor(x)
  print(natural_perm_tensor.shape)

  x = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in synth_data])
  synth_perm_tensor=torch.tensor(x)
  print(synth_perm_tensor.shape)

  n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
  target=np.mean(n1, axis=0)
  print(target.shape)
  n2=f.get('neural/synthetic/monkey_'+final_path)[:]
  neuron_target=np.mean(n2, axis=0)
  print(neuron_target.shape)
  neuro_wise=args.neuro_wise
  print(neuro_wise)
  if neuro_wise == 'True':
    print("!!!!!!!!!!!!!!!!!!!!!")
    with open(f'/content/gdrive/MyDrive/V4/{session_name}/moco101_200_natural_mean.json') as json_file:
      layerlist=[]
      load_data = json.load(json_file)
      json_acceptable_string = load_data.replace("'", "\"")
      d = json.loads(json_acceptable_string)
      max_natural_layer=max(d, key=d.get)
      layerlist.append(max_natural_layer)
  elif model_type=="clip":
    layerlist=['avgpool','relu','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','attnpool']
  elif model_type=="wsl_resnext101" or model_type== "resnext101" or model_type== "resnet101" or model_type== "wide_resnet101":
    layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer3[6]', 'layer3[7]', 'layer3[8]', 'layer3[9]', 'layer3[10]', 'layer3[11]', 'layer3[12]', 'layer3[13]', 'layer3[14]', 'layer3[15]', 'layer3[16]', 'layer3[17]', 'layer3[18]', 'layer3[19]', 'layer3[20]', 'layer3[21]', 'layer3[22]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
  elif model_type=="alexnet":
    layerlist=['features[0]','features[2]','features[3]','features[5]','features[6]','features[8]','features[10]','features[12]','classifier[1]','classifier[4]','classifier[6]']

  else:
    #layer list for resnet 50
    layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
      # layer list for neuro_wise histogram
    # layerlist=['layer3[0]']
  if model_name=="resnet":
    x1=natural_perm_tensor
    model=resnet
  if model_name=="alexnet":
    x1=natural_perm_tensor
    model=alexnet
  activation={}
  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
  for layer in layerlist:
    # #used for robust resnet only!!!!!!!!!!!!!!!!!!!!!
    #exec(f"{model_name}.model.{layer}.register_forward_hook(get_activation('{layer}'))")
    if model_type=="clip":
      exec(f"{model_name}.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
    elif model_type=="linf_8" or model_type=="linf_4" or model_type=="l2_3" or model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      exec(f"{model_name}.model.{layer}.register_forward_hook(get_activation('{layer}'))")
    else:
      exec(f"{model_name}.{layer}.register_forward_hook(get_activation('{layer}'))")
    # used for fast training 
    #exec(f"{model_name}.module.{layer}.register_forward_hook(get_activation('{layer}'))")
  counter=0
  for  minibatch in batch(x1,64):
    print(counter)
    if model_type=="clip":
      output=exec(f"{model_name}.visual(minibatch.to(device))")
    else:
    # output=resnet(minibatch.to(device).divide(255))
      output=exec(f"{model_name}(minibatch.to(device))")
    if counter==0:
      with h5py.File(f'{model_type}_natural_layer_activation.hdf5','w')as f:
        for layer in layerlist:
          dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
    else:
      with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r+')as f:
        for k,v in activation.items():
          print(k)
          data = f[k]
          a=data[...]
          del f[k]
          dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
    counter=counter+1

  # get activation for synthetic images
  if model_name=="resnet":
    x1=synth_perm_tensor
    model=resnet
  if model_name=="alexnet":
    x1=synth_perm_tensor
    model=alexnet
  activation={}
  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
  for layer in layerlist:
    if model_type=="clip":
      exec(f"{model_name}.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
    elif model_type=="linf_8" or model_type=="linf_4" or model_type=="l2_3" or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      exec(f"{model_name}.model.{layer}.register_forward_hook(get_activation('{layer}'))")
    else:
      exec(f"{model_name}.{layer}.register_forward_hook(get_activation('{layer}'))")
  counter=0
  # for  minibatch in batch(x1,64):
  print(counter)
  #output=resnet(minibatch.to(device).divide(255))
  if model_type=="clip":
    output=exec(f"{model_name}.visual(x1.to(device))")
  else:
    output=exec(f"{model_name}(x1.to(device))")
  if counter==0:
    with h5py.File(f'{model_type}_synth_layer_activation.hdf5','w')as f:
      for layer in layerlist:
        dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
  else:
    with h5py.File(f'{model_type}_synth_layer_activation.hdf5','r+')as f:
      for k,v in activation.items():
        print(k)
        data = f[k]
        a=data[...]
        del f[k]
        dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
  counter=counter+1
  # import pycuda.autoinit
  # import pycuda.gpuarray as gpuarray
  # import skcuda.linalg as linalg
  # from skcuda.linalg import PCA as cuPCA
  import numpy as np
  from sklearn.model_selection import KFold
  from sklearn.linear_model import Ridge
  from scipy.stats.stats import pearsonr
  from sklearn.decomposition import PCA
  import torch.nn.functional as F

  natural_score_dict={}
  synth_score_dict={}
  # random_list=[2,5,667,89,43]
  random_list=[2,10,32,89,43]
  #layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
  for key in layerlist:
    natural_score_dict[key]=None
    synth_score_dict[key]=None
  total_synth_corr=[]
  total_natural_corr=[]
  cc=0
  with h5py.File(f'{model_type}_synth_layer_activation.hdf5','r')as s:
    with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r')as f:
      for seed in random_list:
        for k in layerlist:
          print(k)
          natural_data = f[k]
          synth_data=s[k]
          a=natural_data[...]
          b=synth_data[...]
          print('a.shape')
          print(a.shape)
          # pca=cuPCA(n_components=640)
          pca=PCA(random_state=seed)
          # array=np.asfortranarray(a.reshape(640,-1))
          # X_gpu=gpuarray.to_gpu(array)
          # natural_x_pca = pca.fit_transform(X_gpu)
          natural_x_pca = pca.fit_transform(torch.tensor(a).cpu().detach().reshape(640,-1))
          print('natural_x_pca.shape')
          print(natural_x_pca.shape)
          # array=np.asfortranarray(b.reshape(50,-1))
          # X_gpu=gpuarray.to_gpu(array)
          synth_x_pca = pca.transform(torch.tensor(b).cpu().detach().reshape(neuron_target.shape[0],-1))
          kfold = KFold(n_splits=5, shuffle=True,random_state=seed)
          num_neuron=n1.shape[2]
          natural_prediction= np.empty((640,target.shape[1]), dtype=object)
          print('natural_prediction.shape')
          print(natural_prediction.shape)
          synth_prediction=np.empty((neuron_target.shape[0],neuron_target.shape[1]), dtype=object)
          for fold, (train_ids, test_ids) in enumerate(kfold.split(natural_x_pca)):
            clf = Ridge(random_state=seed)
            clf.fit((natural_x_pca)[train_ids],target[train_ids])
            start=fold*10
            end=((fold+1)*10)
            natural_prediction[test_ids]=clf.predict((natural_x_pca)[test_ids])
            # synth_prediction[start:end]=clf.predict((synth_x_pca)[start:end])
            if fold==0:
              synth_prediction=clf.predict((synth_x_pca))
            else:
              synth_prediction=synth_prediction+clf.predict((synth_x_pca))
            if fold==4:
              synth_prediction=synth_prediction/5

          if natural_score_dict[k] is None:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            total_natural_corr=natural_corr_array
            natural_score_dict[k] = np.median(natural_corr_array)
            cc=cc+1
          else:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            total_natural_corr=np.vstack([total_natural_corr,natural_corr_array])
            natural_score=np.median(natural_corr_array)
            natural_score_dict[k] =np.append(natural_score_dict[k],natural_score)
            cc=cc+1
          if synth_score_dict[k] is None:
            synth_corr_array=np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_synth_corr=synth_corr_array
            synth_score_dict[k] = np.median(synth_corr_array)
          else:
            synth_corr_array=np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_synth_corr=np.vstack([total_synth_corr,synth_corr_array])
            synth_score=np.median(synth_corr_array)
            synth_score_dict[k] =np.append(synth_score_dict[k],synth_score)
          # natural_score_dict[k] = np.median(np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])]))
          # synth_score_dict[k] = np.median(np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])]))

          print(natural_score_dict[k])
          print(synth_score_dict[k]) 
      print(cc)
      if neuro_wise=='True':
        np.save(f'gdrive/MyDrive/V4/{session_name}/{model_type}_synth_neuron_corr.npy',total_synth_corr)
        np.save(f'gdrive/MyDrive/V4/{session_name}/{model_type}_natural_neuron_corr.npy',total_natural_corr)


      else:


        from statistics import mean
        new_natural_score_dict = {k:  v.tolist() for k, v in natural_score_dict.items()}
        new_synth_score_dict = {k:  v.tolist() for k, v in synth_score_dict.items()}
        import json
        # Serializing json  
        synth_json = json.dumps(new_synth_score_dict, indent = 4) 
        natural_json = json.dumps(new_natural_score_dict, indent = 4) 
        print(natural_json)
        print(synth_json)

        with open(f"gdrive/MyDrive/V4/{session_name}/{model_type}_natural.json", 'w') as f:
          json.dump(natural_json, f)
        with open(f"gdrive/MyDrive/V4/{session_name}/{model_type}_synth.json", 'w') as f:
          json.dump(synth_json, f)

        natural_mean_dict = {k:  mean(v) for k, v in natural_score_dict.items()}
        synth_mean_dict = {k:  mean(v) for k, v in synth_score_dict.items()}
        json_object = json.dumps(natural_mean_dict, indent = 4) 
        print(json_object)
        with open(f"gdrive/MyDrive/V4/{session_name}/{model_type}_natural_mean.json", 'w') as f:
          json.dump(json_object, f)

        json_object = json.dumps(synth_mean_dict, indent = 4) 
        print(json_object)
        with open(f"gdrive/MyDrive/V4/{session_name}/{model_type}_synth_mean.json", 'w') as f:
          json.dump(json_object, f)


