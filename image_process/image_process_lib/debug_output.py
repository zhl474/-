"""中文路径调试图片与视频输出。"""

import os

import cv2
import rospy


def save_image_to_path(image_path, image):
    if image is None:
        rospy.logwarn("调试图像为空，无法保存: %s", image_path)
        return False
    try:
        image_dir = os.path.dirname(image_path)
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
        extension = os.path.splitext(image_path)[1] or ".jpg"
        success, encoded_image = cv2.imencode(extension, image)
        if not success:
            raise RuntimeError("OpenCV 图像编码失败")
        encoded_image.tofile(image_path)
        return True
    except Exception as exc:
        rospy.logwarn("调试图像保存失败 %s: %s", image_path, exc)
        return False


class DebugVideoRecorder:
    def __init__(self, video_path, fps=10.0, enabled=True):
        self.requested_video_path = video_path
        self.video_path = video_path
        self.fps = max(0.1, float(fps))
        self.enabled = bool(enabled)
        self.writer = None
        self.frame_size = None
        self.codec_name = None
        self.open_failed = False

    def _writer_candidates(self):
        if not self.requested_video_path:
            return []
        base_path, extension = os.path.splitext(self.requested_video_path)
        extension = extension.lower()
        if extension in (".mp4", ".m4v", ".mov"):
            candidates = [(base_path + ".avi", "MJPG"), (self.requested_video_path, "mp4v")]
        elif extension == ".avi":
            candidates = [(self.requested_video_path, "MJPG"), (self.requested_video_path, "XVID")]
        else:
            candidates = [(self.requested_video_path + ".avi", "MJPG")]
        return list(dict.fromkeys(candidates))

    def _open(self, frame_size):
        if not self.enabled or self.open_failed:
            return False
        for video_path, codec_name in self._writer_candidates():
            try:
                video_dir = os.path.dirname(video_path)
                if video_dir:
                    os.makedirs(video_dir, exist_ok=True)
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*codec_name),
                    self.fps,
                    frame_size,
                )
                if writer.isOpened():
                    self.writer = writer
                    self.video_path = video_path
                    self.frame_size = frame_size
                    self.codec_name = codec_name
                    return True
                writer.release()
            except Exception as exc:
                rospy.logwarn("调试视频初始化失败 %s: %s", video_path, exc)
        self.open_failed = True
        return False

    def write(self, image):
        if not self.enabled or self.open_failed or image is None:
            return False
        frame = image
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self.writer is None and not self._open(frame_size):
            return False
        if self.frame_size != frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        return True

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
