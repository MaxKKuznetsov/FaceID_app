import time
import os

class SetSettings():
    def __init__(self):
        self.screen_settings()
        self.video_settings()

    def screen_settings(self):
        self.display_width, self.display_height = 640, 480

    def video_settings(self):
        self.out_video_file = os.path.join('BestFrameSelection', 'saved_video', 'output.avi')
        self.out_video_fps = 20
        self.out_video_width, self.out_video_height = 640, 480



class Timer():
    def __init__(self):
        self.t_start = 0
        self.dt = 0

    def start_timer(self):
        self.t_start = time.time()

    def stop_timer(self):
        self.t_start = 0

    def return_time(self):
        return time.time() - self.t_start
