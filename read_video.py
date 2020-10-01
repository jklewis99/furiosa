import cv2
import numpy as np

def extract_frames(path):
    video = cv2.VideoCapture(path)
    height, width = get_dimensions(path)
    num_frames = count_frames(path)

    # create numpy array for all the frames (convert video to array)
    video = numpy_video(path, num_frames, height, width)

    return video

def numpy_video(path, num_frames, height, width, num_channels=3):
    video = cv2.VideoCapture(path)
    video_array = np.empty((num_frames, height, width, num_channels), np.dtype('uint8')) # fails on high res videos
    frame_idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            video_array[frame_idx] = frame
        else:
            print("1 ERROR: Error reading video", frame_idx)
            break
        frame_idx += 1
    video.release()
    return video_array

def get_dimensions(path):
    video = cv2.VideoCapture(path)
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return h, w

def count_frames(path):
    video = cv2.VideoCapture(path)
    num_frames = 0
    
    try:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        num_frames = manual_count_frames(video)
    video.release()
    return num_frames

def manual_count_frames(video):
    num_frames = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("2 ERROR: Error reading video.")
        num_frames += 1

    return num_frames

def main():
    path = "C:/Users/jklew/Videos/flip.mp4"
    numpy_of_video = extract_frames(path)
    print(numpy_of_video[19])
    cv2.imshow("Frame 20", numpy_of_video[10])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()