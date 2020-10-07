import struct
import numpy as np
import cv2
if __name__ == "__main__":
    file = "./gan_train.bin"

    with open(file, 'rb') as file:
        xyz = []
        uv = []
        for i in range(42*3):
            xyz.append(float(struct.unpack('f', file.read(4))[0]))

        for i in range(42*2):
            uv.append(float(struct.unpack('f', file.read(4))[0]))
        k =[]
        for i in range(3*3):
            k.append(float(struct.unpack('f', file.read(4))[0]))
        image = []
        for i in range(256*256*3):
            image.append(int(struct.unpack('B', file.read(1))[0]))
        mask = []
        for i in range(256*256):
            mask.append(int(struct.unpack('B', file.read(1))[0]))
        # print(xyz)
        # print(len(xyz))
        xyz = np.array(xyz)
        uv = np.array(uv)
        k = np.array(k)
        k = k.reshape((3,3))
        image = np.array(image).reshape((256, 256, 3))
        mask = np.array(mask).reshape((256, 256))
        print(xyz.reshape((42,3)))
        print(uv.reshape((42,2)))
        print(k)
        print(image)
        print(mask)
        cv2.imwrite("test.png", image)
        cv2.imwrite("mask.png", mask)
        # print(xyz.resize((42,3)))
        # print(uv.resize((42,2)))
