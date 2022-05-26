import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser(description='download from gdrive')
parser.add_argument('--model_list',nargs="+")
model_list=args.model_list
directory = "natural_mean"
# Parent Directory path
parent_dir = "/content/"
# Path
path = os.path.join(parent_dir, directory)
os.mkdir(path)
parent_dir = "/content/natural_mean"
session_list=['m_ohp_session1','m_ohp_session2','m_stretch_session1','m_stretch_session2','n_stretch_session1','s_ohp_session1','s_stretch_session1']
for s in session_list:
  directory=s
  path = os.path.join(parent_dir, directory)
  os.mkdir(path)
for s in session_list:
  for m in model_list:
    with open(f'/content/gdrive/MyDrive/V4/{s}/{m}_natural_mean.json') as json_file:
      layerlist=[]
      load_data = json.load(json_file)
      with open(f"/content/natural_mean/{s}/{m}_natural_mean.json", 'w') as f:
        json.dump(load_data, f)
shutil.make_archive("natural_mean", 'zip', parent_dir)
