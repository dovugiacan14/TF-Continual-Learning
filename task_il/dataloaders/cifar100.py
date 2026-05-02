import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=42,pc_valid=0, inc=5):
    assert 100%inc == 0
    task_num=100//inc
    data={}
    taskcla=[]
    size=[3,32,32]

    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir points to task_il/dataloaders/, go up one level to task_il/
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'dat', 'binary_cifar_inc%d' % inc)
    original_data_dir = os.path.join(project_dir, 'data')

    # Check if dataset needs to be downloaded/processed
    # Use a lock file to prevent race conditions in parallel processing
    lock_file = os.path.join(data_dir, '.download_complete.lock')
    need_download = False

    if os.path.exists(lock_file):
        # Dataset already downloaded and verified, skip all checks
        need_download = False
    elif not os.path.isdir(data_dir):
        need_download = True
    else:
        # Check if expected files exist (task_num * 4 files: trainx, trainy, testx, testy)
        expected_files = task_num * 4
        files = [f for f in os.listdir(data_dir) if f.endswith('.bin')] if os.path.exists(data_dir) else []
        if len(files) < expected_files:
            need_download = True
            print(f"Dataset incomplete ({len(files)}/{expected_files} files). Re-downloading...")

    if need_download:
        # Create lock file BEFORE download to prevent race conditions
        os.makedirs(data_dir, exist_ok=True)
        with open(lock_file, 'w') as f:
            f.write('downloading\n')

        mean=[0.5071, 0.4867, 0.4408]
        std=[0.2675, 0.2565, 0.2761]

        # CIFAR100 - with retry logic for HTTP 503 errors
        max_retries = 3
        dat = {}
        for attempt in range(max_retries):
            try:
                print(f'Downloading CIFAR100 (attempt {attempt + 1}/{max_retries})...')
                dat['train']=datasets.CIFAR100(original_data_dir,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.CIFAR100(original_data_dir,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                print('Download successful!')
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'Download failed (attempt {attempt + 1}): {e}')
                    print('Retrying in 5 seconds...')
                    import time
                    time.sleep(5)
                else:
                    print(f'Download failed after {max_retries} attempts: {e}')
                    raise
        for n in range(task_num):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=0
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                nn=(n//inc)
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n%inc)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(data_dir, 'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(data_dir, 'data'+str(t)+s+'y.bin'))
        print('Dataset processing completed!')

        # Mark download as complete
        with open(lock_file, 'w') as f:
            f.write('download_complete\n')

    # Load binary files
    data={}
    ids=list(shuffle(np.arange(task_num),random_state=seed))
    print('Task order =',ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(data_dir, 'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(data_dir, 'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']=str(ids[i])

    # Validation
    for t in data.keys():
        if pc_valid>0 and pc_valid<1:
            r=np.arange(data[t]['train']['x'].size(0))
            r=np.array(shuffle(r,random_state=seed),dtype=int)
            nvalid=int(pc_valid*len(r))
            ivalid=torch.LongTensor(r[:nvalid])
            itrain=torch.LongTensor(r[nvalid:])
            data[t]['valid']={}
            data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
            data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
            data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
            data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
        else:
            data[t]['valid']={}
            data[t]['valid']['x']=data[t]['test']['x'].clone()
            data[t]['valid']['y']=data[t]['test']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
