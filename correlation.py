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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F
from load_model import load_model


parser = argparse.ArgumentParser(description='Neural correlation')
parser.add_argument('--session', type=str)
parser.add_argument('--model_list',nargs="+")
parser.add_argument('--neuro_wise')
# parser.add_argument('--model_name')
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
        



for model_type in model_type_list:
  # load model
  resnet,preprocess=load_model(model_type)
  # load images data
  session_path=args.session.replace('_','/')
  final_path=session_path[:-1]+'_'+session_path[-1:]
  f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
  natural_data = f['images/naturalistic'][:]
  synth_data=f['images/synthetic/monkey_'+final_path][:]
#   print(natural_data.shape)

 
  natural_image_tensor=torch.tensor(np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in natural_data]))
#   print(natural_image_tensor.shape)

  synth_image_tensor=torch.tensor(np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in synth_data]))
#   print(synth_image_tensor.shape)

  # load neural activation data
  n1 = f.get('neural/naturalistic/monkey_'+final_path)[:]
  natural_neuron_target=np.mean(n1, axis=0)
  print(natural_neuron_target.shape)
  n2=f.get('neural/synthetic/monkey_'+final_path)[:]
  synth_neuron_target=np.mean(n2, axis=0)
  print(synth_neuron_target.shape)
  neuro_wise=args.neuro_wise
  print(neuro_wise)

  
  if neuro_wise == 'True':
    with open(f'/content/gdrive/MyDrive/V4/{session_name}/{model_type}_natural_mean.json') as json_file:
      layerlist=[]
      load_data = json.load(json_file)
      json_acceptable_string = load_data.replace("'", "\"")
      d = json.loads(json_acceptable_string)
      # get the best layer
      max_natural_layer=max(d, key=d.get)
      layerlist.append(max_natural_layer)
  else:
    if model_type=="clip":
        layerlist=['avgpool','relu','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','attnpool']
    elif '101' in model_type:
        layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer3[6]', 'layer3[7]', 'layer3[8]', 'layer3[9]', 'layer3[10]', 'layer3[11]', 'layer3[12]', 'layer3[13]', 'layer3[14]', 'layer3[15]', 'layer3[16]', 'layer3[17]', 'layer3[18]', 'layer3[19]', 'layer3[20]', 'layer3[21]', 'layer3[22]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
    elif model_type=="alexnet":
        layerlist=['features[0]','features[2]','features[3]','features[5]','features[6]','features[8]','features[10]','features[12]','classifier[1]','classifier[4]','classifier[6]']
    elif '18' in model_type:
        layerlist=['maxpool','layer1[0]','layer1[1]','layer2[0]','layer2[1]','layer3[0]','layer3[1]','layer4[0]','layer4[1]','avgpool','fc']
    else:
        #layer list for resnet 50
        layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
  model=resnet
  activation={}
  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
  # get activation
  for layer in layerlist:
    if model_type=="clip":
      exec(f"model.visual.{layer}.register_forward_hook(get_activation('{layer}'))")
    elif model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      exec(f"model.model.{layer}.register_forward_hook(get_activation('{layer}'))")
    else:
      exec(f"model.{layer}.register_forward_hook(get_activation('{layer}'))")
  counter=0
  for  minibatch in batch(natural_image_tensor,64):
    print(counter)
    if model_type=="clip":
      output=exec(f"model.visual(minibatch.to(device))")
    elif model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      output=exec(f"model.model(minibatch.to(device))")
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
    output=exec(f"model.visual(synth_image_tensor.to(device))")
  elif model_type=='resnet50_l2_eps0.1' or model_type=='resnet50_l2_eps0.01' or model_type=='resnet50_l2_eps0.03' or model_type=='resnet50_l2_eps0.5' or model_type=='resnet50_l2_eps0.25' or model_type=='resnet50_l2_eps3' or model_type=='resnet50_l2_eps5' or model_type=='resnet50_l2_eps1' or model_type=='resnet50_l2_eps0.05':
      output=exec(f"model.model(synth_image_tensor.to(device))")
  else:
    output=exec(f"model(synth_image_tensor.to(device))")
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
  
  for key in layerlist:
    natural_score_dict[key]=None
    synth_score_dict[key]=None
  total_synth_corr=[]
  total_natural_corr=[]
  cc=0
  with h5py.File(f'{args.neuro_wise}_{model_type}_synth_layer_activation.hdf5','r')as s:
    with h5py.File(f'{args.neuro_wise}_{model_type}_natural_layer_activation.hdf5','r')as f:
      for seed in random_list:
        for k in layerlist:
          print(k)
          natural_data = f[k]
          synth_data=s[k]
        #   a=natural_data[...]
        #   b=synth_data[...]
          pca=PCA(random_state=seed)
          natural_x_pca = pca.fit_transform(torch.tensor(natural_data[...]).cpu().detach().reshape(natural_neuron_target.shape[0],-1))
          synth_x_pca = pca.transform(torch.tensor(synth_data[...]).cpu().detach().reshape(synth_neuron_target.shape[0],-1))
          kfold = KFold(n_splits=5, shuffle=True,random_state=seed)
          natural_prediction= np.empty((natural_neuron_target.shape[0],natural_neuron_target.shape[1]), dtype=object)
          synth_prediction=np.empty((synth_neuron_target.shape[0],synth_neuron_target.shape[1]), dtype=object)
          for fold, (train_ids, test_ids) in enumerate(kfold.split(natural_x_pca)):
            clf = Ridge(random_state=seed)
            clf.fit((natural_x_pca)[train_ids],natural_neuron_target[train_ids])
            natural_prediction[test_ids]=clf.predict((natural_x_pca)[test_ids])
            if fold==0:
              synth_prediction=clf.predict((synth_x_pca))
            else:
              synth_prediction=synth_prediction+clf.predict((synth_x_pca))
            if fold==4:
              synth_prediction=synth_prediction/5

          if natural_score_dict[k] is None:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], natural_neuron_target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            total_natural_corr=natural_corr_array
            natural_score_dict[k] = np.median(natural_corr_array)
            cc=cc+1
          else:
            natural_corr_array= np.array([pearsonr(natural_prediction[:, i], natural_neuron_target[:, i])[0] for i in range(natural_prediction.shape[-1])])
            total_natural_corr=np.vstack([total_natural_corr,natural_corr_array])
            natural_score=np.median(natural_corr_array)
            natural_score_dict[k] =np.append(natural_score_dict[k],natural_score)
            cc=cc+1
          if synth_score_dict[k] is None:
            synth_corr_array=np.array([pearsonr(synth_prediction[:, i], synth_neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_synth_corr=synth_corr_array
            synth_score_dict[k] = np.median(synth_corr_array)
          else:
            synth_corr_array=np.array([pearsonr(synth_prediction[:, i], synth_neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
            total_synth_corr=np.vstack([total_synth_corr,synth_corr_array])
            synth_score=np.median(synth_corr_array)
            synth_score_dict[k] =np.append(synth_score_dict[k],synth_score)
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


