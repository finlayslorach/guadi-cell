{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ethical-senegal",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastai import * \n",
    "from fastai.data.all import *\n",
    "from fastai.vision.data import * \n",
    "from fastai.vision.core import *\n",
    "from fastai.vision.all import *\n",
    "from torchvision import transforms\n",
    "from torch_lr_finder import LRFinder\n",
    "import torch.optim.lr_scheduler as lr_scheduler\n",
    "import torch.utils.data as data\n",
    "from torch import nn, optim\n",
    "from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder\n",
    "import re\n",
    "import os\n",
    "import glob\n",
    "from skimage import io, color, img_as_float32, img_as_uint\n",
    "import random\n",
    "import numpy as np\n",
    "import PIL\n",
    "import glob\n",
    "import gc\n",
    "from torch.optim.lr_scheduler import _LRScheduler\n",
    "from sklearn.metrics import fbeta_score\n",
    "import sys\n",
    "import builtins\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "vietnamese-hierarchy",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "----------------------running script---------------\n"
     ]
    }
   ],
   "source": [
    "print('----------------------running script---------------')\n",
    "fnames = get_image_files('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells')\n",
    "path_glob = glob.glob('/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells/*/*/*/*')\n",
    "path_img = '/hpc/scratch/hdd2/fs541623/CellProfilerFeatureExtraction/CP_Cropped_Cells'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cooked-comedy",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_func(fname):\n",
    "    return (re.match(r'.*(Time_\\d+hrs).*(Well_\\d+).*', fname.name).groups())\n",
    "labels = np.unique(list(zip(*(set(fnames.map(label_func))))))\n",
    "label_n = len(np.unique(labels))\n",
    "classes=len(labels)\n",
    "labels_encoder = {metadata:l for l, metadata in enumerate(labels)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "eligible-chicago",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_encoder(fname):\n",
    "    time, well = re.match(r'.*(Time_\\d+hrs).*(Well_\\d+).*', fname.name).groups()\n",
    "    return labels_encoder[time], labels_encoder[well]\n",
    "indxs = np.random.permutation(range(int(len(fnames))))\n",
    "dset_cut = int(len(fnames)*0.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "unnecessary-portal",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Split into train & val\n",
    "train_files = fnames[indxs[:dset_cut]]\n",
    "valid_files = fnames[indxs[dset_cut:]]\n",
    "\n",
    "## Get labels for shuffled files\n",
    "Y = fnames.map(label_encoder)\n",
    "valid_y = valid_files.map(label_encoder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "directed-compiler",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Dataset \n",
    "class Dataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, x, y):\n",
    "        self.x = x\n",
    "        self.y = y\n",
    "        self.transform = transforms.Resize(224)\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.x)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        img = img_as_float32(Image.open(self.x[idx]))\n",
    "        out = np.zeros((1,14), int) # TO DO : refactor \n",
    "        out[[0,0], np.array(self.y[idx])] = 1\n",
    "        return (self.transform(torch.tensor((img[None]))), torch.tensor(out, dtype=float).squeeze())\n",
    "    \n",
    "## Create Datasets \n",
    "train_ds = Dataset(train_files, train_y) \n",
    "valid_ds = Dataset(valid_files, valid_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "particular-turkish",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_decoder(labels):\n",
    "    label_array=np.array(list(labels_encoder))\n",
    "    idx = np.array(labels).astype(int) > 0 \n",
    "    return label_array[idx]\n",
    "\n",
    "#dataloaders\n",
    "train_iterator = data.DataLoader(train_ds, batch_size=32,shuffle=False, pin_memory=True, num_workers=32)\n",
    "valid_iterator = data.DataLoader(valid_ds, batch_size=32,shuffle=False, pin_memory=True, num_workers=32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "vietnamese-shark",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "---------------Getting model-----------\n",
      "input channels to next layer: 64\n",
      "residual conv layer:(64, 64, 256)\n",
      "identity layer:(256, 64, 256)\n",
      "identity layer:(256, 64, 256)\n",
      "outchannels: 256\n",
      "input channels to next layer: 128\n",
      "residual conv layer:(256, 128, 512)\n",
      "identity layer:(512, 128, 512)\n",
      "identity layer:(512, 128, 512)\n",
      "identity layer:(512, 128, 512)\n",
      "outchannels: 512\n",
      "input channels to next layer: 256\n",
      "residual conv layer:(512, 256, 1024)\n",
      "identity layer:(1024, 256, 1024)\n",
      "identity layer:(1024, 256, 1024)\n",
      "identity layer:(1024, 256, 1024)\n",
      "identity layer:(1024, 256, 1024)\n",
      "identity layer:(1024, 256, 1024)\n",
      "outchannels: 1024\n",
      "input channels to next layer: 512\n",
      "residual conv layer:(1024, 512, 2048)\n",
      "identity layer:(2048, 512, 2048)\n",
      "identity layer:(2048, 512, 2048)\n",
      "outchannels: 2048\n"
     ]
    }
   ],
   "source": [
    "print('---------------Getting model-----------')\n",
    "# model \n",
    "def conv_layer(inputs, outputs, ks, stride, padding, use_activation=None):\n",
    "    layers=[nn.Conv2d(inputs, outputs, ks, stride, padding, bias=False), nn.BatchNorm2d(outputs)]\n",
    "    if use_activation: layers.append(nn.ReLU(inplace=True))\n",
    "    return nn.Sequential(*layers)\n",
    "\n",
    "# Residual Block \n",
    "class residual(nn.Module):\n",
    "    def __init__(self, input_channels, out_channels, stride=1, activation: torch.nn.Module = nn.ReLU(inplace=True)):\n",
    "        super().__init__()\n",
    "    \n",
    "        # 64 -> 256 ; always first block in resnetlayer\n",
    "        self.convs = nn.Sequential(*[conv_layer(input_channels, out_channels, 1, 1, 0, use_activation=True),\n",
    "                                     conv_layer(out_channels, out_channels, 3, stride, 1, use_activation=True),\n",
    "                                     conv_layer(out_channels, out_channels*4, 1, 1, 0, use_activation=True)])\n",
    "        \n",
    "        # if 256 == 4*64 (256) e.g. for other blocks of resnet layer \n",
    "        if input_channels == out_channels*4: \n",
    "            self.conv4 = nn.Identity()\n",
    "            print(f'identity layer:{input_channels, out_channels, out_channels*4}')\n",
    "        else: \n",
    "            # if 64 != 256 ( 4*64) -> do convolutional layer\n",
    "            print(f'residual conv layer:{input_channels, out_channels, out_channels*4}')\n",
    "            self.conv4 = conv_layer(input_channels, out_channels*4, 1, stride, 0)\n",
    "        \n",
    "        self.activation = activation\n",
    "        \n",
    "    def forward(self, X):\n",
    "        return self.activation((self.convs(X) + self.conv4(X)))\n",
    "\n",
    "## Need to refactor \n",
    "class resnetmodel(nn.Module):\n",
    "    def __init__(self, channels, n_blocks, classes=classes):\n",
    "        super().__init__()\n",
    "        self.in_channels = channels[0] # 64\n",
    "        \n",
    "        ## to work with 1 channel images\n",
    "        self.model_stem = nn.Sequential(*[conv_layer(1, self.in_channels, ks=7, stride=2, padding=3, use_activation=True), \n",
    "                                     nn.MaxPool2d(3, stride=2, padding=1)])\n",
    "        self.res_layer1 = self._make_res(residual, channels[0], n_blocks[0])\n",
    "        self.res_layer2 = self._make_res(residual, channels[1], n_blocks[1], stride=2)\n",
    "        self.res_layer3 = self._make_res(residual,channels[2], n_blocks[2], stride=2)\n",
    "        self.res_layer4 = self._make_res(residual, channels[3], n_blocks[3], stride=2)\n",
    "        \n",
    "        # inchannels = 2048??\n",
    "        self.adpool = nn.AdaptiveAvgPool2d((1,1))\n",
    "        self.flatten = nn.Flatten()\n",
    "        self.linear = nn.Linear(self.in_channels, classes)\n",
    "        \n",
    "    \n",
    "    def _make_res(self, residual, channels, n_blocks, stride=1):\n",
    "        # 1st reslayer doesnt have stride == 2 \n",
    "        layers = []\n",
    "        \n",
    "        # 1st block of each res layer always has stride == 1\n",
    "        # e.g. inchannels = 64, channels = 64 --> ends up outputting channels 4*64 = 256\n",
    "        \n",
    "        print(f'input channels to next layer: {channels}')\n",
    "        \n",
    "        # convolution block\n",
    "        layers.append(residual(self.in_channels, channels)) # 256 -> 128 (128 * 4 = 512)\n",
    "        \n",
    "        # identity blocks\n",
    "        for i in range(1, n_blocks):\n",
    "            # input channels = 256 -> 64 \n",
    "            layers.append(residual(channels*4, channels)) # 128*4 = 512 -> 512 (128 * 4)\n",
    "        self.in_channels = 4*channels # set in_channels for next convolution block\n",
    "        print(f'outchannels: {self.in_channels}')\n",
    "        return nn.Sequential(*layers)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = self.model_stem(x)\n",
    "        x = self.res_layer1(x) \n",
    "        x = self.res_layer2(x)\n",
    "        x = self.res_layer3(x)\n",
    "        x = self.res_layer4(x)\n",
    "        x = self.adpool(x)\n",
    "        x = self.flatten(x)\n",
    "        x = self.linear(x)\n",
    "        return x\n",
    "model = resnetmodel(channels=[64,128,256,512], n_blocks=[3,4,6,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "intended-rachel",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr, epochs, bs = 3e-5, 10, 8\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "# Scheduler\n",
    "params = [\n",
    "        {'params':model.model_stem.parameters(), 'lr': lr/10},\n",
    "        {'params':model.res_layer1.parameters(), 'lr': lr/8},\n",
    "        {'params':model.res_layer2.parameters(), 'lr': lr/6},\n",
    "        {'params':model.res_layer3.parameters(), 'lr': lr/4},\n",
    "        {'params':model.res_layer4.parameters(), 'lr': lr/2},\n",
    "        {'params':model.linear.parameters()}]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "patient-valentine",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = optim.Adam(params,lr=lr)\n",
    "total_steps = epochs * len(train_iterator) # epochs * number of batches\n",
    "max_lr = [p['lr'] for p in optimizer.param_groups]\n",
    "scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps)\n",
    "loss_func = nn.BCEWithLogitsLoss()\n",
    "model = model.to(device)\n",
    "loss_func = loss_func.to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "norman-values",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_predictions(model, loader):\n",
    "    model.eval()\n",
    "    images = []\n",
    "    labels = []\n",
    "    preds = []\n",
    "    with torch.no_grad():\n",
    "        for (x, y) in iter(loader):\n",
    "            x = x.to(device)\n",
    "            y_pred = model(x)\n",
    "            y_pred = torch.sigmoid(y_pred) > 0.5\n",
    "            images.append(x.cpu())\n",
    "            labels.append(y.cpu())\n",
    "            preds.append(y_pred.cpu())\n",
    "\n",
    "    images = torch.cat(images, dim = 0)\n",
    "    labels = torch.cat(labels, dim = 0)\n",
    "    preds = torch.cat(preds, dim=0)\n",
    "    return images, labels, preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "economic-ranch",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.load_state_dict(torch.load('/hpc/scratch/hdd2/fs541623/Bash_scripts/resnet50-scratch-p5.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "statutory-burlington",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-26-472df556f8dc>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mimages\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpreds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_predictions\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalid_iterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-25-e333202d7616>\u001b[0m in \u001b[0;36mget_predictions\u001b[0;34m(model, loader)\u001b[0m\n\u001b[1;32m      9\u001b[0m             \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m             \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msigmoid\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m0.5\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m             \u001b[0mimages\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m             \u001b[0mlabels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m             \u001b[0mpreds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "images, labels, preds = get_predictions(model, valid_iterator)\n",
    "with open('predictions.pkl', 'w') as f:  \n",
    "    pickle.dump([images, labels, preds], f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "satellite-numbers",
   "metadata": {},
   "outputs": [],
   "source": [
    "# confusion matrix \n",
    "def plot_confusion_matrix(labels, pred_labels, classes):\n",
    "    \n",
    "    fig = plt.figure(figsize = (50, 50));\n",
    "    ax = fig.add_subplot(1, 1, 1);\n",
    "    cm = confusion_matrix(labels, pred_labels);\n",
    "    cm = ConfusionMatrixDisplay(cm, display_labels = classes);\n",
    "    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)\n",
    "    fig.delaxes(fig.axes[1]) #delete colorbar\n",
    "    plt.xticks(rotation = 90)\n",
    "    plt.xlabel('Predicted Label', fontsize = 50)\n",
    "    plt.ylabel('True Label', fontsize = 50)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Environment (conda_fastai)",
   "language": "python",
   "name": "conda_fastai"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
