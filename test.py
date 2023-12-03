import cv2
import numpy as np

def detectByCamera():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3-tiny.cfg")
    layer_names = net.getUnconnectedOutLayersNames()
    output_layers = [net.getUnconnectedOutLayers()[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    video = cv2.VideoCapture(0)  # 0 corresponds to the default camera
    print('Detecting people in real-time with YOLO...')

    while True:
        check, frame = video.read()

        if check:
            frame = detect(frame, net, layer_names, output_layers)

            cv2.imshow('Real-time Human Detection', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectByCamera()
