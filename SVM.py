import torch as th
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from torchvision.io import read_image
import matplotlib.pyplot as plt
import seaborn as sns

# initializam placa video
rtx = th.device('cuda' if th.cuda.is_available() else 'cpu')

# pregatim cele 3 fisiere respectiv: antrenament, validare si test
pre_date = pd.read_csv('train.csv')
pre_test = pd.read_csv('test.csv')
pre_val = pd.read_csv('validation.csv')

# extragem imaginile si etichetele pentru antrenament
date = pre_date['image_id']
categorie = pre_date['label']

# initializam parametrii de lumina si contrast ca in CNN
brightness = 0.3
contrast = 1.3

# initializam vectorii pentru datele de antrenament
img_data = []

for i in date:
    # tranformam imaginile in valori cuprinse intre 0 si 1
    image = read_image('train/'+i+'.png').float()/255.0

    # aplicam contrastul si luminozitatea
    image = image * contrast + brightness

    # ne asiguram ca valorile raman intre 0, 1
    image = th.clip(image, 0.0, 1.0)
    img_data.append(image.reshape(-1).numpy())

# convertim vectorul la numpy
img_data = np.array(img_data)
# extragem etichetele de antrenament
img_lbl = categorie.values

# repetam procesul de mai sus pentru datele de validare, respectiv datele de test
val = []
for i in pre_val['image_id']:
    image = read_image('validation/'+i+'.png').float()/255.0
    image = image * contrast + brightness
    image = th.clip(image, 0.0, 1.0)

    flat = image.reshape(-1).numpy()
    val.append(flat)

val = np.array(val)
val_lbl = pre_val['label'].values

test = []
for i in pre_test['image_id']:
    image = read_image('test/'+i+'.png').float()/255.0
    image = image * contrast + brightness
    image = th.clip(image, 0.0, 1.0)

    flat = image.reshape(-1).numpy()
    test.append(flat)

test = np.array(test)

# normalizam datele cu StandardScaler
scaler = StandardScaler()
img_data_scaled = scaler.fit_transform(img_data)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# incercam mai multi parametri pentru PCA si SVM
pca_try = [200, 300, 400, 500]
c_try = [0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000]

# ne pregatim o variabila pentru cea mai buna acuratete de vlaidare
# si pentru predictiile finale
best_val_acc = 0
best_val_pred = None

for pca_components in pca_try:
    
    # aplicam PCA pentru a reduce dimensiunea si a extrage caracteristici principale
    pca = PCA(n_components=pca_components)
    img_data_pca = pca.fit_transform(img_data)
    val_pca = pca.transform(val)
    test_pca = pca.transform(test)
    
    
    for c in c_try:
        # initializam SVM cu kernel-ul RBF si incercam diferite valori pentru parametrul C
        svm = SVC(kernel='rbf', C=c)
        svm.fit(img_data_pca, img_lbl)
        train_pred = svm.predict(img_data_pca)
        val_pred = svm.predict(val_pca)
        
        # monitorizam acuratetea pe setul de antrenament si validare
        train_acc = accuracy_score(img_lbl, train_pred) * 100
        val_acc = accuracy_score(val_lbl, val_pred) * 100
        
        print(f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        # salvam cea mai buna varianta de validare
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_predictions = svm.predict(test_pca)

# evaluam si pregatim csv-ul pentru submit pe Kaggle in formatul corect
pre_sample = pd.read_csv('sample_submission.csv')
result = pd.DataFrame({
    'image_id': pre_sample['image_id'].values,
    'label': best_predictions
})

result.to_csv('s_predictions.csv', index=False)
