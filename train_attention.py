#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
import scipy.spatial.distance as ssd
from get_input import loadData
from models import GCNAttention, TripletAttentionNet
from loss import TripletLoss
from torch.optim import lr_scheduler
from config import *
import tqdm
import numpy as np
import random
import time
import os


def loadDataDirect(shuffleList, imageList, batchIndex, epoch):
    """Load citation network dataset (cora only for now)"""
    '''
            image_list: images captured according to bounding boxes
            label_list: labels of image_list
            category_list: the center coordinates and the number of bounding boxes
            '''
    shuffleNum = imageList[(len(shuffleList) * epoch + batchIndex) % len(imageList)]
    batchIndex = shuffleList[batchIndex]
    image_list_arc, label_list_arc, bbox_list_arc, total_image_arc, adj_arc, corr_arc, area_arc, category_list_arc = loadData(
        os.path.join(sketchVPath, str(batchIndex) + ".csv"),
        os.path.join(sketchEPath, str(batchIndex) + ".csv"),
        os.path.join(sketchImgTrainPath, str(batchIndex).zfill(12) + ".png"))

    image_list_pos, label_list_pos, bbox_list_pos, total_image_pos, adj_pos, corr_pos, area_pos, category_list_pos = loadData(
        os.path.join(imageVPath, str(batchIndex) + ".csv"),
        os.path.join(imageEPath, str(batchIndex) + ".csv"),
        os.path.join(imageImgPath, str(batchIndex).zfill(12) + ".jpg"))

    while shuffleNum == batchIndex:
        shuffleNum = imageList[random.randint(0, len(imageList) - 1)]

    image_list_neg, label_list_neg, bbox_list_neg, total_image_neg, adj_neg, corr_neg, area_neg, category_list_neg = loadData(
        os.path.join(imageVPath, str(shuffleNum) + ".csv"),
        os.path.join(imageEPath, str(shuffleNum) + ".csv"),
        os.path.join(imageImgPath, str(shuffleNum).zfill(12) + ".jpg"))

    return image_list_arc, label_list_arc, bbox_list_arc, total_image_arc, adj_arc, corr_arc, image_list_pos, label_list_pos, bbox_list_pos, total_image_pos, adj_pos, corr_pos, image_list_neg, label_list_neg, bbox_list_neg, total_image_neg, adj_neg, corr_neg, area_arc, area_pos, area_neg

def outputHtml(sketchindex,indexList,mode="test"):

    imageNameList=testList
    sketchPath=sketchImgTestPath
    imgPath=imageImgTestPath


    tmpLine = "<tr>"

    tmpLine+="<td><image src='%s' width=256 /></td>"%(os.path.join(sketchPath,str(shuffleListTest[sketchindex]).zfill(12)+".png"))
    for i in indexList:
        if i!=sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (os.path.join(imgPath,str(imageNameList[i]).zfill(12)+".jpg"))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))

    return tmpLine +"</tr>"

def loadDataDirectTest(mode, shuffleList_new, batchIndex):
    """Load citation network dataset (cora only for now)"""

    batchIndex = shuffleList_new[batchIndex]
    if mode == "sketch":
        image_list, label_list, bbox_list, total_image, adj, corr, area, category_list = loadData(
            os.path.join(sketchVPathTest, str(batchIndex) + ".csv"),
            os.path.join(sketchEPathTest, str(batchIndex) + ".csv"),
            os.path.join(sketchImgTestPath, str(batchIndex).zfill(12) + ".png"))
    else:
        image_list, label_list, bbox_list, total_image, adj, corr, area, category_list = loadData(
            os.path.join(imageVPathTest, str(batchIndex) + ".csv"),
            os.path.join(imageEPathTest, str(batchIndex) + ".csv"),
            os.path.join(imageImgTestPath, str(batchIndex).zfill(12) + ".jpg"))

    return image_list, label_list, bbox_list, total_image, adj, corr, category_list


NUM_VIEWS = 1


def compute_view_specific_distance(sketch_feats, image_feats):
    multi_view_dists = ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')
    return multi_view_dists


def calculate_accuracy(dist, epoch_name):
    top1 = 0
    top5 = 0
    top10 = 0
    tmpLine=""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        tmpLine += outputHtml(i, rank[:10]) + "\n"
    num = dist.shape[0]


    htmlContent = """
           <html>
           <head></head>
           <body>
           <table>%s</table>
           </body>
           </html>""" % (tmpLine)
    with open(r"html_result/furniture_"+epoch_name+".html", 'w+')as f:
        f.write(htmlContent)
    num = dist.shape[0]
    print(epoch_name + ' top1: ' + str(top1 / float(num)))
    print(epoch_name + ' top5: ' + str(top5 / float(num)))
    print(epoch_name + 'top10: ' + str(top10 / float(num)))
    return top1 / float(num), top5 / float(num), top10 / float(num)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
margin = 1.
embedding_net = GCNAttention(gcn_input_shape=featureNumDict[featureType], gcn_output_shape=outPutDim)

model = TripletAttentionNet(embedding_net)
if args.cuda:
    model.cuda()
loss_fn = TripletLoss(margin)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

Result_Step = 1

t_total = time.time()
if args.cuda:
    model.cuda()

maxModel = "0"

for epoch in range(args.epochs + 1):
    model.train()
    random.shuffle(shuffleList)
    running_loss = 0.0
    for batch in range(batches):
        t = time.time()

        image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc, corr_arc, image_list_pos, label_list_pos, category_list_pos, total_image_pos, adj_pos, corr_pos, image_list_neg, label_list_neg, category_list_neg, total_image_neg, adj_neg, corr_neg, area_arc, area_pos, area_neg = loadDataDirect(
            shuffleList, trainList, batch, epoch)

        output1, output2, output3 = model(image_list_arc, label_list_arc, category_list_arc, total_image_arc, adj_arc,
                                          corr_arc, image_list_pos, label_list_pos, category_list_pos, total_image_pos,
                                          adj_pos, corr_pos, image_list_neg, label_list_neg, category_list_neg,
                                          total_image_neg, adj_neg, corr_neg)

        #
        loss_train = loss_fn(output1, output2, output3)

        # 2.1 loss regularization
        loss = loss_train
        # 2.2 back propagation
        optimizer.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()  # update parameters of net
        running_loss += loss.item()
        # 3. update parameters of net
        if (batch % batch_size) == 0 or (batch + 1) == batches:
            # optimizer the net
            print('Epoch: {:04d}'.format(epoch + 1), 'Batch: {:04d}'.format(batch + 1),
                  'loss_train: {:.4f}'.format(running_loss / batch_size),
                  'time: {:.4f}s'.format(time.time() - t))
            running_loss = 0.0

        torch.save(model.state_dict(), "model/model_" + str(epoch) + ".pth")
    # -----------------evaluate---------------------------
    epoch_name =  str(epoch)

    aList = []
    pList = []
    sketch_category_lists = []
    image_category_lists = []
    model.eval()
    with torch.no_grad():

        for batchIndex in tqdm.tqdm(range(batchesTest)):
            image_list, label_list, bbox_list, total_image, adj, corr, category_list = loadDataDirectTest("sketch",
                                                                                                          shuffleListTest,
                                                                                                          batchIndex)
            a = model.get_embedding(image_list, label_list, bbox_list, total_image, adj, corr)

            aList.append(a.cpu().numpy()[0])
            sketch_category_lists.append(category_list)
        aList = np.array(aList)
        for batchIndex in tqdm.tqdm(range(batchImgTest)):
            image_list, label_list, bbox_list, total_image, adj, corr, category_list = loadDataDirectTest("image",
                                                                                                          testList,
                                                                                                          batchIndex)
            p = model.get_embedding(image_list, label_list, bbox_list, total_image, adj, corr)
            image_category_lists.append(category_list)
            pList.append(p.cpu().numpy()[0])
        pList = np.array(pList)
        dis = compute_view_specific_distance(aList, pList)

        top1, top5, top10 = calculate_accuracy(dis, str(epoch))
        print("top1, top5, top10:",top1, top5, top10)




print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
