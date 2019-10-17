
def save_frame(frame_index):
    from multiagent.rendering import pyglet
    pyglet.image.get_buffer_manager().get_color_buffer().save('../../frames/frame%06d.bmp' % frame_index)

def combine_frames_to_video(video_fname):
    import cv2
    import numpy as np
    import os
    from os.path import isfile, join

    pathIn = '../../frames/'
    pathOut = video_fname
    fps = 1.0/0.05
    frame_array = []
    files = sorted([f for f in os.listdir(pathIn) if isfile(join(pathIn, f))])
    # for sorting the file names properly

    frame_array = []
    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

if __name__ == "__main__":
    combine_frames_to_video("../../videos/test_video.mp4")