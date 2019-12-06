import cv2 as cv
import dlib


class Camera:

    def __init__(self, camera_code):
        self.camera = camera_code
        self.capture = cv.VideoCapture(camera_code)
        if not self.capture.isOpened():
            print('不能访问摄像头或者无法访问路径！')
            exit(0)
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        self.ROI = []
        self.frame_counts = self.capture.get(cv.CAP_PROP_FRAME_COUNT)

    def read(self):
        detector = dlib.get_frontal_face_detector()
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                continue

            dects = detector(frame)
            for face in dects:
                left = face.left()
                right = face.right()
                top = face.top()
                bottom = face.bottom()

                h = bottom - top
                w = right - left
                roi = frame[top + h // 10:top + h // 10 * 3, left + w // 9 * 3:left + w // 9 * 6]
                cv.rectangle(frame, (left + w // 9 * 3, top + h // 10), (left + w // 9 * 6, top + h // 10 * 3),
                             color=(0, 0, 255))
                cv.rectangle(frame, (left, top), (left + w, top + h), color=(0, 0, 255))
                self.ROI.append(roi)
            cv.imshow('frame', frame)
            if (cv.waitKey(1) & 0xFF == ord('q')) or len(self.ROI) == 300:
                break
        self.capture.release()
        cv.destroyAllWindows()
        return self.ROI, self.fps

    def show_rio(self):
        for roi in self.ROI:
            cv.imshow('roi', roi)
            cv.waitKey(100)


if __name__ == '__main__':
    video_path = 'videos/rohin_active.mov'
    cascade_path = 'haarcascades/face_detect.xml'
    camera = Camera(video_path, cascade_path)
    camera.read()

    camera.show_rio()
