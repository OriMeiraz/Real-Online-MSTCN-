from traceback import print_tb
import cv2
from load_model import load_model, args
import torch
from project import prepare_models, extraction_examples, device, shape
from FrameGenrator import FrameGenerator
from DataStructures import ModelRecreate
import time
import tqdm
from try2 import to_vid

gestures = {0.0: "no gesture",
            1.0: "needle passing",
            2.0: "pull the suture",
            3.0: "Instrument tie",
            4.0: "Lay the knot",
            5.0: "Cut the suture"}

tool_usage = {0.0: "no tool in hand",
              1.0: "needle driver",
              2.0: "forceps",
              3.0: "scissors"}


def get_label_arr(label):
    f, s, g = label.split(' ')
    f = int(f)
    s = int(s)
    g = gestures[float(g[1])]
    return (s-f+1)*[g]


labels = """0 409 G0
410 527 G1
528 581 G2
582 743 G3
744 818 G4
819 878 G3
879 929 G4
930 1016 G3
1017 1061 G4
1062 1121 G3
1122 1175 G4
1176 1215 G0
1216 1270 G5
1271 1504 G0
1505 1693 G1
1694 1819 G2
1820 1873 G3
1874 1942 G4
1943 1990 G3
1991 2038 G4
2039 2086 G3
2087 2125 G4
2126 2176 G3
2177 2212 G4
2213 2254 G3
2255 2326 G4
2327 2467 G0
2468 2527 G5
2528 2641 G0
2642 2920 G1
2921 3010 G2
3011 3048 G3
3049 3060 G0
3061 3123 G3
3124 3144 G0
3145 3241 G3
3242 3355 G4
3356 3454 G3
3455 3490 G4
3491 3550 G3
3551 3583 G4
3584 3691 G3
3692 3739 G4
3740 3784 G0
3785 3826 G5
3827 3910 G0"""

arr = []
for label in labels.split('\n'):
    arr += get_label_arr(label)


def text(image, y, line):
    return cv2.putText(
        image, line,
        (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)


video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side/"


def main():
    model, extractor = prepare_models()
    mean, std = extractor.input_mean, extractor.input_std
    extractor = extraction_examples(extractor, shape=shape)
    frame_gen = FrameGenerator(
        "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side/")
    mr = ModelRecreate(model.to(device), 0,
                       frame_gen, extractor.to(device), mean, std)

    image = cv2.imread(video_path+"img_00001.jpg")
    t0 = time.time()
    if args.task == 'gestures':
        for i in tqdm.tqdm(range(3910)):
            image = cv2.imread(f'{video_path}img_{str(i+1).zfill(5)}.jpg')
            preds = mr.next()
            pred_gesture = torch.argmax(preds[0]).item()
            image = text(image, 400, f'prediction: {gestures[pred_gesture]}')
            image = text(image, 450, f'label: {arr[i]}')
            cv2.imwrite(f'video/img_{str(i+1).zfill(5)}.jpg', image)
    print(time.time() - t0)
    to_vid()


if __name__ == '__main__':
    main()
