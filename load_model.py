import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms 
from load_SIN_model import load_SIN_model

def load_model(model_type):
  if model_type=="resnet50_trained_on_SIN" or model_type=="resnet50_trained_on_SIN_and_IN" or model_type=="resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN":
    state_dict = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/{model_type}.pth.tar',map_location="cpu")['state_dict']
    resnet=models.resnet50(pretrained=False)
    for k in list(state_dict.keys()):
        if k.startswith('module'):
            state_dict[k[len("module."):]] = state_dict[k]
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
    resnet = models.resnet101(pretrained=False)
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
  if model_type.split('_')[0]=="moco50":
    # load checkpoints of moco
    epoch_num=model_type.split('_')[1]
    state_dict = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/moco50/moco_{epoch_num}.pth.tar',map_location=torch.device('cpu'))['state_dict']
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
  if model_type.split('_')[0]=="rn101":
    resnet=models.resnet101(pretrained=False)
    model_epoch=model_type.split('_')[1]
    # load checkpoints of moco
    state_dict = torch.load(f'/content/gdrive/MyDrive/model_checkpoints/rn101/rn101_epoch{model_epoch}.pth.tar',map_location=torch.device('cpu'))['state_dict']
    
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') :
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msg = resnet.load_state_dict(state_dict, strict=False)
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
  return resnet,preprocess
