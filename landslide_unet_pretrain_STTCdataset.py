import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T
import cv2
import matplotlib.pyplot as plt
from unet import UNet
import time
from tqdm import tqdm
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, Dice, BinaryPrecision, BinaryRecall

class LandslideDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        img = cv2.imread(img_path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.where(mask>1,1,mask)
        if self.transforms is not None:
            img=self.transforms(img)
            mask=self.transforms(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)

def get_prediction(img_path, model, device, quantile_thre):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    orig = img.copy()
    img = img.astype("float32") / 255.0
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(device)
    with torch.no_grad():
        pred = model(img).squeeze()
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()
    threshold=np.quantile(pred, quantile_thre)
    predMask = (pred>threshold).astype(int)
    return orig, pred, predMask
def evaluate(model, imagePath, device, quantile_thre, mode):
    maskname=os.path.splitext(imagePath)[0]
    imagePath_mask=maskname+'.png'
    maskPath=os.path.join('./landslides_'+mode+'set/Masks',imagePath_mask)
    gtMask = Image.open(maskPath)
    gtMask = np.array(gtMask)
    gtMask = np.where(gtMask>1,1,gtMask)
    gtMask = cv2.resize(gtMask,(256,256))
    iou=np.zeros_like(quantile_thre)
    f1_score=np.zeros_like(quantile_thre)
    dice_score=np.zeros_like(quantile_thre)
    precision=np.zeros_like(quantile_thre)
    recall=np.zeros_like(quantile_thre)
    for i in range(len(quantile_thre)):
        orig, pred, predMask = get_prediction(os.path.join('./landslides_'+mode+'set/Images',imagePath), 
                                              model, device, quantile_thre[i])
        thresh=np.quantile(pred, quantile_thre[i])
        metric1=BinaryJaccardIndex(threshold=thresh)
        metric2=BinaryF1Score(threshold=thresh)
        metric3=Dice(threshold=thresh)
        metric4=BinaryPrecision(threshold=quantile_thre[i])
        metric5=BinaryRecall(threshold=quantile_thre[i])
        iou[i]=metric1(torch.tensor(pred),torch.tensor(gtMask)).numpy()
        f1_score[i]=metric2(torch.tensor(pred),torch.tensor(gtMask)).numpy()
        dice_score[i]=metric3(torch.tensor(pred),torch.tensor(gtMask)).numpy()
        precision[i]=metric4(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
        recall[i]=metric5(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
    return iou, f1_score, dice_score, precision, recall
def fix_colour_masks(image,i):
    # RGB Colors: 255, 0, 0 is red. 0, 255, 0 is green. 0, 0, 255 is blue
    colours = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[i]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('The model will be running on', device, 'device')

num_classes = 1
transforms_t = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])
dataset = LandslideDataset('landslides_dataset', transforms_t)
dataset_val = LandslideDataset('landslides_valset', transforms_t)
dataset_test = LandslideDataset('landslides_testset', transforms_t)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=4)
    
model = UNet(n_channels=3, n_classes=num_classes)
model.to(device)

lossFunc = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
print("[INFO] training the network...")
NUM_EPOCHS = 20
H = {"train_loss": [], "val_loss": []}
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	model.train()
	totalTrainLoss = 0
	totalValLoss = 0
	for x, y in data_loader:
		x, y = x.to(device), y.to(device) 
		pred = model(x)
		loss = lossFunc(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		totalTrainLoss += loss
	with torch.no_grad():
		model.eval()
		for x, y in data_loader_val:
			x, y = x.to(device), y.to(device)
			pred = model(x)
			totalValLoss += lossFunc(pred, y)
	H["train_loss"].append(totalTrainLoss.cpu().detach().numpy())
	H["val_loss"].append(totalValLoss.cpu().detach().numpy())
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
		totalTrainLoss, totalValLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}minutes".format(
	(endTime - startTime)/60.0))

epoch_optimal=np.argmin(H["val_loss"])+1
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="validation_loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('outputs/training_validation_loss_unet')
torch.save(model, 'outputs/landslides_unet_model_pretrain.pth')           