import cv2
import os


def to_vid(image_folder='video', video_name='video.avi', start=1000):

    images = [f'img_{str(i+1).zfill(5)}.jpg' for i in range(start, 3910)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    to_vid()
