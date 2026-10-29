import cv2
import imageio
from matplotlib import gridspec, pyplot as plt
from IPython.display import HTML, display
from PIL import Image
import numpy as np


def to_gif(images, duration, filename):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave(filename, images, duration=duration)
    print(f"{filename} saved!")

def to_img_sequence(images, filename, cols):
    """Converts image sequence (4D numpy array) to a sequence of frame plots."""
    num_frames = len(images)
    rows = int(np.ceil(num_frames / cols))

    aspect_ratio = images[0].shape[1] / images[0].shape[0]
    fig_width = 10
    fig_height = ((fig_width // cols // aspect_ratio) + 1) * rows

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(rows, cols, wspace=0, hspace=0.1)

    for idx in range(num_frames):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(images[idx])
        ax.set_title(f"Frame {idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.suptitle(filename, fontsize=12)
    plt.show()

def progress(value, max=100):
    '''Displays progress bar'''
    return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

def mp4_to_gif(video_file):
    '''Converts .mp4 file to .gif file of same duration.'''

    cap = cv2.VideoCapture(video_file)

    # Get the frame rate of the video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a GIF writer
    gif_path = video_file[:-4]+'.gif'
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Convert frames to a GIF using PIL
    frames_pil = [Image.fromarray(cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    frames_pil[0].save(gif_path, save_all=True, append_images=frames_pil[1:],
                       duration=int(1000 / frame_rate), loop=0)

    cap.release()
    print(f"Conversion of {video_file} complete.")

def compute_euclidian_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_vector(p1, p2):
    return np.array(p2) - np.array(p1)

def compute_angle(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    cosine_angle = np.dot(unit_vector1, unit_vector2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compute_velocity(position):
    '''
    input
        position: np array shape (n_frames, 2) where position[i] = (x,y) in pixels
    output
        velocity: np array shape (n_frames, 2) where velocity[i] = (x,y) in pixels/frame
    '''
    position_float = position.astype(float)
    displacement = np.diff(position_float, axis=0)
    velocity = displacement / 1.0  # Assuming each frame is one time unit
    return velocity