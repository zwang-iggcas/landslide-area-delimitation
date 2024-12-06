import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T
import torchvision
import utils
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import matplotlib.pyplot as plt
import random
import os.path as osp
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, Dice, BinaryPrecision, BinaryRecall
import time

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
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id=osp.basename(img_path).split('.')[0]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    return model

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'landslide']
transforms_t = T.Compose([T.ToPILImage(), T.ToTensor()])
def compute_iou(mask1, mask2, threshold):
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Parameters:
    - mask1: Numpy array representing the first binary mask.
    - mask2: Numpy array representing the second binary mask.

    Returns:
    - iou: IoU value between the two masks.
    """
    mask1=np.where(mask1>threshold,1,0)
    mask2=np.where(mask2>threshold,1,0)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou
def mask_to_keep(masks,quantile_thre,iou_thre=0.01):
    if len(masks)==1:
        keep=masks
    else:
        keep=masks[0][np.newaxis]
        for k in range(1,len(masks)):
            current_mask=masks[k]
            is_overlapping=False
            for selected_mask in keep:
                iou=compute_iou(current_mask, selected_mask, quantile_thre)
                if iou>iou_thre:
                    is_overlapping=True
                    break
            if not is_overlapping:
                keep=np.concatenate((keep, current_mask[np.newaxis]))
    return keep
def get_prediction(img_path, model, device, threshold, num_landslide_max=3):
    img = Image.open(img_path).convert('RGB')
    img = transforms_t(img)
    img = img.to(device)
    with torch.no_grad():
        pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t=min(len(pred_score)-1,num_landslide_max-1)
    masks = (pred[0]['masks']).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    keep=mask_to_keep(masks,quantile_thre=threshold)
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return keep, pred_boxes, pred_class
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask
def fix_colour_masks(image,i):
    # RGB Colors: 255, 0, 0 is red. 0, 255, 0 is green. 0, 0, 255 is blue
    colours = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[i]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask
def plot_prediction(img_path, model, device, threshold=0.1, num_landslide_max=3, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img_path, model, device, threshold, num_landslide_max)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks=np.where(masks>threshold,1,0)
    for i in range(len(masks)):
      rgb_mask = fix_colour_masks(masks[i],i)
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('outputs/landslides_maskrcnn_segmentation_mask_'+img_path.split(os.path.sep)[-1])
def scale_array_to_0_1(array):
    """
    Scale a numpy array to the range [0, 1].

    Parameters:
    - array: Numpy array to be scaled.

    Returns:
    - scaled_array: Numpy array scaled to the range [0, 1].
    """
    min_value = np.min(array)
    max_value = np.max(array)

    scaled_array = (array - min_value) / (max_value - min_value)

    return scaled_array
def evaluate_test(model, imagePath, device, quantile_thre, mode):
    maskname=os.path.splitext(imagePath)[0]
    imagePath_mask=maskname+'.png'
    maskPath=os.path.join('./landslides_'+mode+'set/Masks',imagePath_mask)
    mask = Image.open(maskPath)
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    gtMask = mask == obj_ids[:, None, None]
    img = Image.open(os.path.join('./landslides_'+mode+'set/Images',imagePath)).convert('RGB')
    img = transforms_t(img)
    img = img.to(device)
    with torch.no_grad():
        pred = model([img])
    
    gtMask=np.sum(gtMask,axis=0)
    gtMask = np.where(gtMask>1,1,gtMask)
    iou=np.zeros_like(quantile_thre)
    f1_score=np.zeros_like(quantile_thre)
    dice_score=np.zeros_like(quantile_thre)
    precision=np.zeros_like(quantile_thre)
    recall=np.zeros_like(quantile_thre)
    for i in range(len(quantile_thre)):
        masks = pred[0]['masks'].squeeze(axis=1).detach().cpu().numpy()
        masks=mask_to_keep(masks,quantile_thre=quantile_thre[i])
        predMask=np.sum(masks,axis=0)
        predMask=scale_array_to_0_1(predMask)
        metric1=BinaryJaccardIndex(threshold=quantile_thre[i])
        metric2=BinaryF1Score(threshold=quantile_thre[i])
        metric3=Dice(threshold=quantile_thre[i])
        metric4=BinaryPrecision(threshold=quantile_thre[i])
        metric5=BinaryRecall(threshold=quantile_thre[i])
        iou[i]=metric1(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
        f1_score[i]=metric2(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
        dice_score[i]=metric3(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
        precision[i]=metric4(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
        recall[i]=metric5(torch.tensor(predMask),torch.tensor(gtMask)).numpy()
    return iou, f1_score, dice_score, precision, recall

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('The model will be running on', device, 'device')

num_classes = 2
dataset = LandslideDataset('landslides_dataset', transforms_t)
dataset_val = LandslideDataset('landslides_valset', transforms_t)
dataset_test = LandslideDataset('landslides_testset', transforms_t)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

model = get_model_instance_segmentation(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

num_epochs = 20
losses_classifier=[]
losses_box_reg=[]
losses_mask=[]
losses_objectness=[]
losses_rpn_box_reg=[]
max_Fscore=0
precision_ave=[]
recall_ave=[]
Fscore_ave=[]
img_path='./nonvalid_images/2014-cut1.jpg'
startTime = time.time()
for epoch in range(num_epochs):
    loss_train=train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    train_dict=loss_train[1]
    for key, value in train_dict.items():
        train_dict[key] = value.detach().cpu().numpy()
    losses_classifier.append(train_dict['loss_classifier'])
    losses_box_reg.append(train_dict['loss_box_reg'])
    losses_mask.append(train_dict['loss_mask'])
    losses_objectness.append(train_dict['loss_objectness'])
    losses_rpn_box_reg.append(train_dict['loss_rpn_box_reg'])
    evaluator=evaluate(model, data_loader_val, device=device)
    precision=evaluator[1][1]
    recall=evaluator[1][8]
    F1_score=2*precision*recall/(precision+recall)
    if max_Fscore < F1_score:
        max_Fscore=F1_score
        model.eval()
        plot_prediction(img_path,model,device)
    else:
        print('Validation F-score decreased! Epoch#'+str(epoch))
    precision_ave.append(evaluator[1][1])
    recall_ave.append(evaluator[1][8])
    Fscore_ave.append(F1_score)
    lr_scheduler.step()

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}minutes".format(
	(endTime - startTime)/60.0))
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 
para_total=count_parameters(model)
print('para_total='+str(para_total))
plot_prediction(img_path,model,device,num_landslide_max=2)
torch.save(model, 'outputs/landslides_maskrcnn_model_pretrain.pth')
plt.figure()    
plt.plot(losses_classifier, label='loss_classifier')
plt.plot(losses_box_reg, label='loss_box_reg')
plt.plot(losses_mask, label='loss_mask')
plt.plot(losses_objectness, label='loss_objectness')
plt.plot(losses_rpn_box_reg, label='loss_rpn_box_reg')
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.savefig('outputs/training_loss_maskrcnn')
plt.figure()
plt.plot(precision_ave, label='Average precision')
plt.plot(recall_ave, label='Average recall')
plt.xlabel("Epoch #")
plt.ylabel("Precision/recall")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.savefig('outputs/validation_results_maskrcnn')
