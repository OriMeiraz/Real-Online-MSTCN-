import cv2
import os
import tqdm

fps = 30
size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
videoWriter = cv2.VideoWriter(
    os.path.join('video', 'final.wmv'), fourcc,
    fps, size)

for i in tqdm.tqdm(range(3910)):
    img = cv2.imread(f'video/img_{str(i+1).zfill(5)}.jpg')
    videoWriter.write(img)

videoWriter.release()
