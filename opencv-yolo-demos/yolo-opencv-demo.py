#encoding:utf-8
import cv2
import numpy as np 
from cv2 import dnn

confThreshold = 0.5


net = dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")

def getOutputsNames(net):
    # 获取 net 中所有的层的名字
    layersNames = net.getLayerNames()

    print("layersNames:",layersNames)
    # 获取没有向后连接的层的名字，最后一层就是 unconnectedoutlayers
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out size",out.shape)
        for detection in out:
            # 不同的数据集训练下的 label 数量不一样，yolov3 是在 coco 数据集上训练的，所以支持 80 种类别，输出层代表多个 box 的信息
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
               # x,y,width,height 都是相对于输入图片的比例，所以需要乘以相应的宽高进行复原
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
 
    # 利用 NMS 算法消除多余的框，有些框会叠加在一块，留下置信度最高的框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.5)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        print(box)
        cv2.rectangle(frame,(left,top),(left+width,top+height),(0,0,255))
        return frame

def detect(img):

    # 1. 图像缩放到规整的 608*608 分辨率
    img = cv2.resize(img,(608,608))
    w = img.shape[1]
    h = img.shape[0]

    # 2. 从图像处创建 blob 对象
    blob = dnn.blobFromImage(img,1/255.0)

    # 3. 将图像输入给神经网络
    net.setInput(blob)

    layername = getOutputsNames(net)
    print("layername-->",layername)
  
    # 4. 神经网络进行前向推断预测
    detections = net.forward(layername)

    # 5. 推断的结果进行后处理优化
    img = postprocess(img,detections)


    return img 

def main():

    # 读取图片
    img = cv2.imread("yolotest.jpg")

    # 目标检测
    img = detect(img)

    # 绘制图片
    cv2.imshow("test",img)
    cv2.waitKey()
    cv2.imwrite("yoloresult.jpg",img)
    cv2.destroyAllWindows()
    
    pass

if __name__ == "__main__":
    main()