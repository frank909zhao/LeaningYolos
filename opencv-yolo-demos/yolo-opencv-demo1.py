#encoding:utf-8
import cv2
import numpy as np 
from cv2 import dnn

confThreshold = 0.2


net = dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
# net = dnn.readNetFromDarknet("yolov3-tiny.cfg","yolov3-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(DNN_TARGET_CPU)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out size",out.shape)
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                print("detect:",detection[0:5])
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                print("detect:",[left,top,width,height])
 
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

    img = cv2.resize(img,(416,416))
    print(img.shape)
    w = img.shape[1]
    h = img.shape[0]

    blob = dnn.blobFromImage(img,1/255.0)

    net.setInput(blob)

    layername = getOutputsNames(net)
    print("layername:",layername)

    detections = net.forward(layername)

    print("detections.shape:",len(detections))

    img = postprocess(img,detections)


    return img 

def main():

    # 加载视频
    cap = cv2.VideoCapture("bottle_test.mp4")
    # 加载默认的摄像头视频流
    #cap = cv2.VideoCapture(0)

    while cap.isOpened():


        ret,img = cap.read()

        img = detect(img)

        if img is not None:

            cv2.imshow("test",img)

        # 按 ESC 键结束
        if cv2.waitKey(1) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()