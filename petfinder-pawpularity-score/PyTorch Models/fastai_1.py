import sys 
import os
import math
from utils import colorstr, LOGGER
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
from timm.data.mixup import Mixup
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

set_seed(1, reproducible=True)




train_df = pd.read_csv('train.csv')
train_df['path'] = train_df['Id'].map(lambda x:'train/'+ str(x) +'.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
print(f"There are {len(train_df)} images")
train_df['Pawpularity'].hist(figsize = (10, 5))
print(f"The mean Pawpularity score is {train_df['Pawpularity'].mean()}")
print(f"The median Pawpularity score is {train_df['Pawpularity'].median()}")
print(f"The standard deviation of the Pawpularity score is {train_df['Pawpularity'].std()}")
print(f"There are {len(train_df['Pawpularity'].unique())} unique values of Pawpularity score")
train_df['norm_score'] = train_df['Pawpularity']/100



seed=12
set_seed(seed, reproducible=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True



#Sturges' rule
num_bins = int(np.floor(1+np.log2(len(train_df))))
print(num_bins)

train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)
train_df['bins'].hist()



def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))

def get_data(fold):
    train_df_f = train_df.copy()
    # add is_valid for validation fold
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)
    splitter = RandomSplitter(0.2)
    # Change RandomSplitter to IndexSplitter
    splitter = IndexSplitter(splitter(range(len(train_df)))[1])
    dls = DataBlock(blocks=(ImageBlock, RegressionBlock),
                get_x=ColReader('path'),
                get_y=ColReader('norm_score'),
                splitter=splitter,
                item_tfms=Resize(224), #pass in item_tfms
                batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
               )
    
    paw_dls = dls.dataloaders(train_df_f, 
                          bs=BATCH_SIZE,
                          num_workers=8,
                          seed=seed)
    
    return paw_dls, splitter


def get_learner(fold_num):
    data, splitter = get_data(fold_num)
    
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=data.c)
    learn = Learner(data, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse, cbs=[MixUp(0.2)]).to_fp16()
    
    return learn, splitter



train_df['fold'] = -1
N_FOLDS = 10
BATCH_SIZE = 4

strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
    train_df.iloc[train_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')

    
    
learn, splitter = get_learner(fold_num=0)
learn.lr_find()
'''
learn.fit_one_cycle(5, 
                    2e-5, 
                    cbs=[SaveModelCallback(), 
                         EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=2)]) 
'''