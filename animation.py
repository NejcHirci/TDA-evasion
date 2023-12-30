import numpy as np
import cv2

def createVideo(space):
    video = space.astype(float)

    n_frames, height, width = np.shape(video)
    newWidth = 800
    newHeight = int(height/width*newWidth)

    newVideo = []
    for f in range(n_frames):
        newImage = np.reshape(video[f], (height, width, 1))
        newImage = cv2.resize(newImage, (newWidth, newHeight))
        newImage[newImage>0] = 1
        newImage[0,:] = 1
        newImage[-1,:] = 1
        newImage[:,0] = 1
        newImage[:,-1] = 1
        newImage *= 255
        
        newVideo.append(newImage)
    newVideo = np.array(newVideo, dtype=np.uint8)
    newShape = np.shape(newVideo)
    video_file = 'output_video.avi'
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24.0
    resolution = (newShape[2], newShape[1])

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(video_file, codec, fps, resolution)

    videoSlower = 5
    # Generate frames (you can replace this with your own frames)
    for i in range(videoSlower*len(newVideo)):
        #frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
        # cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frame = np.repeat(newVideo[i//videoSlower][:, :, np.newaxis], 3, axis=2)
        # print(np.shape(frame1))
        # Write the frame to the video file
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video created: {video_file}")