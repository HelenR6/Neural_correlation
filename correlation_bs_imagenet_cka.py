import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms          
import torchvision.models as models
import torch 
import torch.nn as nn
import os
from PIL import Image
import json
import torch
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
from load_model import load_model
# from cka import *
from cka import *


parser = argparse.ArgumentParser(description='Neural correlation')
parser.add_argument('--session', type=str)
parser.add_argument('--model_list',nargs="+")
parser.add_argument('--neuro_wise')
parser.add_argument('--per_figure', type=str)
# parser.add_argument('--model_name')
parser.add_argument('--base_model')
args = parser.parse_args()
session_name=args.session
neuro_wise=args.neuro_wise
model_type_list=args.model_list
# model_name=args.model_name
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

num_classes = 1000
num_images=1000
num_images_per_class = (
    num_images- 1) // num_classes
base_indices = np.arange(num_images_per_class).astype(int)
indices = []
for i in range(num_classes):
    indices.extend(50 * i + base_indices)
for i in range((num_images - 1) % num_classes + 1):
    indices.extend(50 * i + np.array([num_images_per_class]).astype(int))
filepaths = []

imagenet_dir="/content/gdrive/MyDrive/bs_imagenet"
# with h5py.File(imagenet_filepath, 'r') as f:
for index in indices:
    imagepath = os.path.join(imagenet_dir, f"{index}.png")
    # if not os.path.isfile(imagepath):
    #     image = np.array(f['val/images'][index])
    #     Image.fromarray(image).save(imagepath)
    filepaths.append(imagepath)
#IN_image_tensor=torch.tensor(np.array([np.array(preprocess(Image.open(image_filepath).copy())) for image_filepath in filepaths]))
for model_type in model_type_list:
  print(model_type)

  if not os.path.exists(f'/content/gdrive/MyDrive/V4/{session_name}/pls_IN_pca_{model_type}_natural_cka_score.npy'):
      print("not exists")
#     continue
      resnet,preprocess=load_model(model_type)
      IN_image_tensor=torch.tensor(np.array([np.array(preprocess(Image.open(image_filepath).copy())) for image_filepath in filepaths]))
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
        if (args.base_model):
          with open(f'/content/gdrive/MyDrive/V4/{session_name}/{args.base_model}_natural_mean.json') as json_file:
            layerlist=[]
            load_data = json.load(json_file)
            json_acceptable_string = load_data.replace("'", "\"")
            d = json.loads(json_acceptable_string)
            max_natural_layer=max(d, key=d.get)
            layerlist.append(max_natural_layer)
        else:
          with open(f'/content/gdrive/MyDrive/V4/{session_name}/{model_type}_natural_mean.json') as json_file:
            layerlist=[]
            load_data = json.load(json_file)
            json_acceptable_string = load_data.replace("'", "\"")
            d = json.loads(json_acceptable_string)
            max_natural_layer=max(d, key=d.get)
            layerlist.append(max_natural_layer)
      elif model_type=="clip":
        layerlist=['avgpool','relu','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','attnpool']
      elif model_type=="wsl_resnext101" or model_type== "resnext101" or model_type== "resnet101" or model_type== "wide_resnet101" or '101' in model_type:
        layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer3[6]', 'layer3[7]', 'layer3[8]', 'layer3[9]', 'layer3[10]', 'layer3[11]', 'layer3[12]', 'layer3[13]', 'layer3[14]', 'layer3[15]', 'layer3[16]', 'layer3[17]', 'layer3[18]', 'layer3[19]', 'layer3[20]', 'layer3[21]', 'layer3[22]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
      elif model_type=="alexnet":
        layerlist=['features[0]','features[2]','features[3]','features[5]','features[6]','features[8]','features[10]','features[12]','classifier[1]','classifier[4]','classifier[6]']
      elif '18' in model_type:
        layerlist=['maxpool','layer1[0]','layer1[1]','layer2[0]','layer2[1]','layer3[0]','layer3[1]','layer4[0]','layer4[1]','avgpool','fc']
      else:
        #layer list for resnet 50
        layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
      model=resnet
      # get activation for natural images

      activation={}
      def get_activation(name):
          def hook(model, input, output):
              activation[name] = output.detach()
          return hook
      for layer in layerlist:
        if model_type=="clip":
          exec(f"model.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
        elif model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
          exec(f"model.model.{layer}.register_forward_hook(get_activation('{layer}'))")
        else:
          exec(f"model.{layer}.register_forward_hook(get_activation('{layer}'))")
      counter=0
      for  minibatch in batch(natural_perm_tensor,64):
        print(counter)
        if model_type=="clip":
          output=exec(f"model.visual(minibatch.to(device))")
        else:
          output=exec(f"model(minibatch.to(device))")
        if counter==0:
          with h5py.File(f'{args.neuro_wise}_{model_type}_natural_layer_activation.hdf5','w')as f:
            for layer in layerlist:
              dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
        else:
          with h5py.File(f'{args.neuro_wise}_{model_type}_natural_layer_activation.hdf5','r+')as f:
            for k,v in activation.items():
              print(k)
              data = f[k]
              a=data[...]
              del f[k]
              dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
        counter=counter+1

      # get activation for imagenet images
      activation={}
      def get_activation(name):
          def hook(model, input, output):
              activation[name] = output.detach()
          return hook
      for layer in layerlist:
        if model_type=="clip":
          exec(f"model.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
        elif model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
          exec(f"model.model.{layer}.register_forward_hook(get_activation('{layer}'))")
        else:
          exec(f"model.{layer}.register_forward_hook(get_activation('{layer}'))")
      counter=0
      for  minibatch in batch(IN_image_tensor,64):
        print(counter)
        if model_type=="clip":
          output=exec(f"model.visual(minibatch.to(device))")
        else:
          output=exec(f"model(minibatch.to(device))")
        if counter==0:
          with h5py.File(f'{args.neuro_wise}_{model_type}_imagenet_layer_activation.hdf5','w')as f:
            for layer in layerlist:
              dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
        else:
          with h5py.File(f'{args.neuro_wise}_{model_type}_imagenet_layer_activation.hdf5','r+')as f:
            for k,v in activation.items():
              print(k)
              data = f[k]
              a=data[...]
              del f[k]
              dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
        counter=counter+1

      # get activation for synthetic images

      activation={}
      def get_activation(name):
          def hook(model, input, output):
              activation[name] = output.detach()
          return hook
      for layer in layerlist:
        if model_type=="clip":
          exec(f"model.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
        elif model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
          exec(f"model.model.{layer}.register_forward_hook(get_activation('{layer}'))")
        else:
          exec(f"model.{layer}.register_forward_hook(get_activation('{layer}'))")
      counter=0
      if model_type=="clip":
        output=exec(f"model.visual(synth_perm_tensor.to(device))")
      else:
        output=exec(f"model(synth_perm_tensor.to(device))")
      if counter==0:
        with h5py.File(f'{args.neuro_wise}_{model_type}_synth_layer_activation.hdf5','w')as f:
          for layer in layerlist:
            dset=f.create_dataset(layer,data=activation[layer].cpu().detach().numpy())
      else:
        with h5py.File(f'{args.neuro_wise}_{model_type}_synth_layer_activation.hdf5','r+')as f:
          for k,v in activation.items():
            print(k)
            data = f[k]
            a=data[...]
            del f[k]
            dset=f.create_dataset(k,data=np.concatenate((a,activation[k].cpu().detach().numpy()),axis=0))
      counter=counter+1


      natural_score_dict={}
      synth_score_dict={}
      random_list=[2,10,32,89,43]
      #layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
      for key in layerlist:
        natural_score_dict[key]=None
        synth_score_dict[key]=None
      total_synth_corr=[]
      total_natural_corr=[]
      cc=0
      with h5py.File(f'{args.neuro_wise}_{model_type}_imagenet_layer_activation.hdf5','r')as m:
        with h5py.File(f'{args.neuro_wise}_{model_type}_synth_layer_activation.hdf5','r')as s:
            with h5py.File(f'{args.neuro_wise}_{model_type}_natural_layer_activation.hdf5','r')as f:
                #for seed in random_list:

                    for k in layerlist:
                        print(k)
                        imagenet_data=m[k]
                        natural_data = f[k]
                        synth_data=s[k]
                        a=natural_data[...]
                        b=synth_data[...]
                        c=imagenet_data[...]
                        pca=PCA()
                        pca.fit(torch.tensor(c).cpu().detach().reshape(1000,-1))
                        natural_x_pca = pca.transform(torch.tensor(a).cpu().detach().reshape(640,-1))
                        synth_x_pca = pca.transform(torch.tensor(b).cpu().detach().reshape(neuron_target.shape[0],-1))
                        natural_score = cka(gram_rbf(natural_x_pca, 0.5), gram_rbf(target, 0.5))
                        synth_score = cka(gram_rbf(synth_x_pca, 0.5), gram_rbf(neuron_target, 0.5))
                        natural_score_dict[k] = natural_score
                        synth_score_dict[k] = synth_score
                    max_natural_score=max(natural_score_dict.values())
                    max_synth_score=max(synth_score_dict.values())
                    np.save(f'gdrive/MyDrive/V4/{session_name}/pls_IN_pca_{model_type}_natural_cka_score.npy',max_natural_score)
                    np.save(f'gdrive/MyDrive/V4/{session_name}/pls_IN_pca_{model_type}_synth_cka_score.npy',max_synth_score)
      os.remove(f'{args.neuro_wise}_{model_type}_natural_layer_activation.hdf5')
      os.remove(f'{args.neuro_wise}_{model_type}_synth_layer_activation.hdf5')
      os.remove(f'{args.neuro_wise}_{model_type}_imagenet_layer_activation.hdf5')          
                #         kfold = KFold(n_splits=10, shuffle=True,random_state=seed)
                #         num_neuron=n1.shape[2]
                #         natural_prediction= np.empty((640,target.shape[1]), dtype=object)
                #         synth_prediction=np.empty((neuron_target.shape[0],neuron_target.shape[1]), dtype=object)
                #         for fold, (train_ids, test_ids) in enumerate(kfold.split(natural_x_pca)):
                # #             clf = Ridge(random_state=seed)
                #             pls=PLSRegression(n_components=25,scale=False)
                #             pls.fit((natural_x_pca)[train_ids],target[train_ids])
                #             start=fold*10
                #             end=((fold+1)*10)
                #             natural_prediction[test_ids]=pls.predict((natural_x_pca)[test_ids])
                #             # synth_prediction[start:end]=clf.predict((synth_x_pca)[start:end])
                #             if fold==0:
                #                 synth_prediction=pls.predict((synth_x_pca))
                #             else:
                #                 synth_prediction=synth_prediction+pls.predict((synth_x_pca))
                #             if fold==9:
                #                 synth_prediction=synth_prediction/10


