#@title Create EditVideo class
#@markdown The codes in this cell is responsible for video editing. This includes:

#@markdown - edit FPS for  original and skeleton video to be insync with the GIF
#@markdown - resize original and skeleton video by inputed width and height
#@markdown - Trim the original video by start and end frame
#@markdown - Trim the original video by start and end Time in seconds
#@markdown - Perform super-impose on original+skeleton and gif+skeleton
#@markdown - Preview the edited video, or inputed video if user inputed a video path
#@markdown - Place edited original and edited gif video side by side and save it
#@markdown - Save the edited original video


import cv2
from moviepy.editor import VideoFileClip, ImageSequenceClip, clips_array, concatenate_videoclips
from IPython.display import display, HTML
import time

class EditVideo:
  def __init__(self, original_path, skeleton_path=None, gif_path=None):
    """
    Initialise variable and load video

    Parameters:
    - original_path (string): Path to the original video
    - skeleton_path (string): Path to the skeleton video
    - gif_path (string, optional): Path to the inference gif
    """
    self.__original_path = original_path
    self.__original_clip = VideoFileClip(self.__original_path)

    self.__skeleton_path = skeleton_path
    self.__skeleton_clip = VideoFileClip(self.__skeleton_path) if self.__skeleton_path is not None else None

    self.__gif_path = gif_path
    self.__gif_clip = VideoFileClip(self.__gif_path) if self.__gif_path is not None else None


  def resize(self, width, height): 
    """
    Resize original and skeleton video by height and crop the video from center

    Parameters:
    - width (int): The expected width.
    - height (int): The expected height.
    """
    print(f"Original Before: width:{self.__original_clip.size[0]} height:{self.__original_clip.size[1]}")
    if (self.__skeleton_clip!=None):
      print(f"Skeleton Before: width:{self.__skeleton_clip.size[0]} height:{self.__skeleton_clip.size[1]}")

    #Resize original and skeleton to input hight
    self.__original_clip = self.__original_clip.resize(height=height)

    # Calculate the left and right boundaries for center cropping
    left_bound = (self.__original_clip.size[0] - width) // 2
    right_bound = left_bound + width

    # Crop the clips to the desired width
    self.__original_clip = self.__original_clip.crop(x1=left_bound, x2=right_bound)

    if (self.__skeleton_clip!=None):
      self.__skeleton_clip = self.__skeleton_clip.resize(height=height)

      # Calculate the left and right boundaries for center cropping
      left_bound = (self.__skeleton_clip.size[0] - width) // 2
      right_bound = left_bound + width

      self.__skeleton_clip = self.__skeleton_clip.crop(x1=left_bound, x2=right_bound)


    print(f"Original After: width:{self.__original_clip.size[0]} height:{self.__original_clip.size[1]}")
    if (self.__skeleton_clip!=None):
      print(f"Skeleton After: width:{self.__skeleton_clip.size[0]} height:{self.__skeleton_clip.size[1]}")
  
  def trimByFrame(self, start_frame, end_frame):
    """
    Trim the video clip based on frame indices.

    Parameters:
    - start_frame (int): Starting frame index for trimming.
    - end_frame (int): Ending frame index for trimming.
    """

    self.__original_clip = self.__original_clip.subclip(max(0, start_frame/self.__original_clip.fps), min(end_frame/self.__original_clip.fps, self.__original_clip.duration * self.__original_clip.fps))

  def trimBySeconds(self, start_time, end_time):
    """
    Trim the video clip based on time in seconds.

    Parameters:
    - start_time (float): Starting time in seconds for trimming.
    - end_time (float): Ending time in seconds for trimming.
    """

    # Trim the video clip
    self.__original_clip = self.__original_clip.subclip(max(0, start_time), min(self.__original_clip.duration, end_time))

  def sync_fps(self): 
    """
    Function to synchronise the fps for original and skeleton to the GIF
    """

    if (self.__gif_path == None or self.__skeleton_path == None):
      print("Unable to do sync fps as gif/skeleton is not given")
      return

    oldfps_o = self.__original_clip.fps
    oldfps_s = self.__skeleton_clip.fps
    oldfps_g = self.__gif_clip.fps

    print(f"Set Original Video fps:{oldfps_o} -> {oldfps_g}")
    self.__original_clip = self.__original_clip.set_fps(oldfps_g)
    print(f"Original Video fps successfully set to {self.__original_clip.fps}")

    print(f"Set Skeleton Video fps:{oldfps_s} -> {oldfps_g}")
    self.__skeleton_clip = self.__skeleton_clip.set_fps(oldfps_g)
    print(f"Skeleton Video fps successfully set to {self.__skeleton_clip.fps}")


  def preview(self, video_path=None):
    """
    Function to display the current original video if no video path is given, 
    otherwise, display the video of the given video path.

    Parameters:
    - video_path (str, optional): Path to the video file.
    """
    if video_path==None:
      display(self.__original_clip.ipython_display(width=800))
    else:
      display(VideoFileClip(video_path).ipython_display(width=800))

  def perform_superimpose(self):
    """
    Function to overlay skeleton onto the original video and gif
    """

    if (self.__gif_path == None or self.__skeleton_path == None):
      print("Unable to do perform superimpose as gif/skeleton is not given")
      return

    #turn all fps to gif fps
    self.sync_fps()

    # Get video properties
    width, height = self.__gif_clip.size
    fps_g = self.__gif_clip.fps

    processed_frames = []
    processed_frames2 = []
    for frame_o, frame_s, frame_g in zip(self.__original_clip.iter_frames(), self.__skeleton_clip.iter_frames(), self.__gif_clip.iter_frames()):
        # Resize skeleton frame to match the gif video dimensions
        frame_s = cv2.resize(frame_s, (width, height))
        frame_o = cv2.resize(frame_o, (width, height))

        # Combine the original and skeleton frames
        result = cv2.addWeighted(frame_o, 1, frame_s, 2, 0)
        # Combine the gif and skeleton frames
        result2 = cv2.addWeighted(frame_g, 1, frame_s, 2, 0)

        # Append the processed frame to the list
        processed_frames.append(result)
        processed_frames2.append(result2)

    # Create a new video clip from the processed frames
    self.__original_clip = ImageSequenceClip(processed_frames, fps=fps_g)
    self.__gif_clip = ImageSequenceClip(processed_frames2, fps=fps_g)
  
  def combine_and_save_videos(self, output_path):
    """
    Function to place original video and gif side by side and save it.

    Parameters:
    - output_path (str): Path to the save the merged video.
    """

    if (self.__gif_path == None):
      print("Unable to do combine as gif is not given")
      return

    #Resize original video to fix the size of gif
    self.resize(self.__gif_clip.size[0], self.__gif_clip.size[1])

    # Combine videos side by side
    combined_clip = clips_array([[self.__original_clip, self.__gif_clip]])

    # Write the combined video to the output path
    combined_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp_audio.m4a', remove_temp=True)

  def save(self, output_path):
    """
    Function to save the edited original video.

    Parameters:
    - output_path (str): Path to the save the edited original video.
    """

    self.__original_clip.write_videofile(output_path, fps=self.__original_clip.fps, remove_temp=True)



