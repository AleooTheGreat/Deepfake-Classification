import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torchvision as tv
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.data import DataLoader, TensorDataset

# initializam placa video daca e disponibil
rtx = th.device('cuda' if th.cuda.is_available() else 'cpu')

# citim fisierele CSV cu datele
pre_date = pd.read_csv('train.csv')
pre_test = pd.read_csv('test.csv')
pre_val = pd.read_csv('validation.csv')

# extragem imaginile si labelurile
date = pre_date['image_id']
categorie = pre_date['label']

# liste pentru stocare
# imaginile originale pentru calcularea mean/std
# imaginile augmentate pentru antrenament, validare si test
original = []  
train = []     
labels_train = []  

val = []
test = []

# DataLoader are nevoie de un dataset custom pentru a incarca imaginile si labelurile
class Date(Dataset):
    def __init__(self, img, lb):
        self.img = img
        self.lb = lb
    
    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.lb[i]

# modelul arhitectural pentru reteaua neuronala convolutionala
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.block_counter = 0
        self.blocks = nn.ModuleList()

    # creeaza un bloc convolutionar: Conv-BatchNorm-ReLU-Conv-BatchNorm-ReLU-MaxPool-Dropout
    def strat(self, drop , ker , pad , pol , v):
        block = nn.Sequential(
            nn.Conv2d(v[0], v[1], kernel_size=ker, padding=pad),
            nn.BatchNorm2d(v[1]),
            nn.ReLU(),
            nn.Conv2d(v[1], v[2], kernel_size=ker, padding=pad),
            nn.BatchNorm2d(v[2]),
            nn.ReLU(),
            nn.MaxPool2d(pol, pol),
            nn.Dropout(drop)
        )
        self.blocks.append(block)
        self.block_counter += 1
        return self

    # partea de clasificare: Flatten + Dense layers
    def smooth(self, v, size, dropout):
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # v[1]*size*size = dimensiunea dupa conv layers
            nn.Linear(v[1] * size * size, v[0]),  
            nn.BatchNorm1d(v[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(v[0], self.num_classes)
        )
        return self

    # forward parcurge toate blocurile, iar apoi clasificatorul
    def forward(self, model):
        for block in self.blocks:
            model = block(model)
        model = self.classifier(model)
        return model

# citim imaginile originale si le normalizam la 0-1
for i in date:
    image = read_image('train/'+i+'.png').float()/255.0
    original.append(image)

# calculam mean si std pe toate imaginile pentru normalizare
reduce = th.cat([img.view(3, -1) for img in original], dim=1)
media = reduce.mean(dim=1)
std = reduce.std(dim=1)

# 5 tipuri de augmentare pentru a mari diversitatea datelor
transformari = [
    # Doar normalizare
    tv.transforms.Compose([
        tv.transforms.Normalize(mean=media, std=std)
    ]),

    # Rotatie + transformari geometrice + culori
    tv.transforms.Compose([
        tv.transforms.RandomRotation(degrees=30), 
        tv.transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        tv.transforms.Normalize(mean=media, std=std)
    ]),
    
    # Resize + crop + flip-uri
    tv.transforms.Compose([
        tv.transforms.Resize((120, 120)),
        tv.transforms.RandomCrop(100),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomVerticalFlip(p=0.3),
        tv.transforms.Normalize(mean=media, std=std)
    ]),
    
    # Rotatii fixe (0, 90, 180, 270 grade)
    tv.transforms.Compose([
        tv.transforms.RandomChoice([
            tv.transforms.Lambda(lambda img: img),
            tv.transforms.Lambda(lambda img: tv.transforms.functional.rotate(img, 90)),
            tv.transforms.Lambda(lambda img: tv.transforms.functional.rotate(img, 180)),
            tv.transforms.Lambda(lambda img: tv.transforms.functional.rotate(img, 270))
        ]),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        tv.transforms.Normalize(mean=media, std=std)
    ]),
    
    # FiveCrop si alegem unul aleatoriu
    tv.transforms.Compose([
        tv.transforms.Resize((110, 110)),
        tv.transforms.FiveCrop(100),
        tv.transforms.Lambda(lambda crops: crops[np.random.randint(0, 5)]),
        tv.transforms.Normalize(mean=media, std=std)
    ])
]

# aplicam fiecare transformare pe toate imaginile
for trans in transformari:
    mod = []
    for image in original:
        mod.append(trans(image))
    
    train.append(th.stack(mod))
    # repetam labelurile pentru fiecare set augmentat
    labels_train.extend(categorie.values)  

# concatenam toate seturile augmentate
train = th.cat(train, dim=0)
lbl_train = th.tensor(labels_train)

# pentru validare si test folosim doar normalizarea (transformari[0])
for i in pre_val['image_id']:
    img = read_image('validation/'+i+'.png').float()/255.0
    val.append(transformari[0](img))

val = th.stack(val)
val_cls = th.tensor(pre_val['label'].values)

for i in pre_test['image_id']:
    img = read_image('test/'+i+'.png').float()/255.0
    test.append(transformari[0](img))

test = th.stack(test)

# construim modelul cu 4 blocuri convolutionale
nemesis = CNN(5)

nemesis.strat(drop=0.25, ker=5, pad=2, pol=2, v=[3, 64, 64])
nemesis.strat(drop=0.3, ker=3, pad=1, pol=2, v=[64, 128, 128]) 
nemesis.strat(drop=0.4, ker=3, pad=1, pol=2, v=[128, 256, 256])
nemesis.strat(drop=0.5, ker=3, pad=1, pol=2, v=[256, 384, 384])

# clasificatorul
nemesis.smooth(v=[256, 384], size=6, dropout=0.6)

nemesis = nemesis.to(rtx)

# dataLoader pentru batch processing
baus = DataLoader(Date(train, labels_train), batch_size=64, shuffle=True, num_workers=8)
odo = DataLoader(Date(val, val_cls), batch_size=64, shuffle=False, num_workers=8)

# loss function cu label smoothing pentru generalizare mai buna
cel = nn.CrossEntropyLoss(label_smoothing = 0.1)

# adamW optimizer cu weight decay pentru regularizare
opt = optim.AdamW(nemesis.parameters(), lr=0.001, weight_decay=1e-4, eps=1e-8)

# scheduler pentru scaderea learning rate-ului pe o curba cosinus 
# conform cu numarul de epoch-uri si batch-uri
EPOCHS = 200
steps_per_epoch = len(baus)
total_steps = EPOCHS * steps_per_epoch

scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

best_val_acc = 0.0
best_models = []

# antrenam modelul
for epoch in range(EPOCHS):
    nemesis.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # antrenament pe batch-uri
    for (img, lb) in baus:
        img = img.to(rtx)
        lb = lb.to(rtx)
        
        # resetam gradientii ca sa nu se acumuleze
        opt.zero_grad()

        # forward pass
        out = nemesis(img)

        # calculam loss-ul 
        loss = cel(out, lb)
        
        # calculam gradientii in raport cu loss-ul (backpropagation)
        loss.backward()
        
        # actualizam modelul cu gradientii noi
        opt.step()

        # actualizeaza learning rate-ul
        scheduler.step() 
        
        # statistici
        running_loss += loss.item()
        pred = out.argmax(1)
        total += lb.size(0)
        correct += (pred == lb).sum().item()
    
    train_acc = 100 * correct / total
    
    # evaluare pe validation set
    nemesis.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

     # nu calculam gradienti pentru evaluare
    with th.no_grad(): 
        for img, lb in odo:
            img = img.to(rtx)
            lb = lb.to(rtx)
            out = nemesis(img)
            loss = cel(out, lb)
            
            val_loss += loss.item()
            pred = out.argmax(1)
            val_total += lb.size(0)
            val_correct += (pred == lb).sum().item()
    
    val_acc = 100 * val_correct / val_total

    print(f'{epoch+1:3d}: loss: {running_loss/len(baus):.4f} | acc: {train_acc:.2f}%  | loss: {val_loss/len(odo):.4f} | acc: {val_acc:.2f}% |', end='\n')
    
    # salvam doar modelele peste un prag dorit de acuratete de evaluare
    if val_acc >= 93.3:
        model_name = f'best/model_acc_{val_acc:.2f}_ep{epoch+1}.pth'
        th.save(nemesis.state_dict(), model_name)
        best_models.append((val_acc, model_name))    

# combinam predictiile modelelor salvate prin a le supune la vot
model_files = []
for filename in os.listdir('best'):
    if filename.endswith('.pth'):
        model_files.append(f'best/{filename}')

# respectam oridinea de predictie a imaginilor din sample_submission
pre_sample = pd.read_csv('sample_submission.csv')
image_ids = pre_sample['image_id'].values
n = len(image_ids)

# matricea de voturi pentru fiecare imagine si clasa
democracy = np.zeros((n, 5))

# votam
for i, model_path in enumerate(model_files):
    nemesis.load_state_dict(th.load(model_path,map_location = rtx))
    nemesis.eval()
    
    predictions = []
    
    # procesam testul in batch-uri de 64
    for j in range(0, len(test), 64):  
        testin = test[j:j+64].to(rtx)
        out = nemesis(testin)
        predicted = out.argmax(1)
        predictions.extend(predicted.cpu().numpy())
    
    # adunam voturile pentru fiecare clasa
    for img_idx, pred_class in enumerate(predictions):
        democracy[img_idx, pred_class] += 1

# clasa cu cele mai multe voturi castiga
predictions = []
for i in range(n):
    predictions.append(np.argmax(democracy[i]))

# salvam predictiile finale
result = pd.DataFrame({
    'image_id': image_ids,
    'label': predictions
})

result.to_csv('predictions.csv', index=False)
