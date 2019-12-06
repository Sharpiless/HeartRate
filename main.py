from signal_handler import Handler
import numpy as np
import fft_filter
import dlib
import cv2 as cv
import hr_calculator

freqs_min = 0.8
freqs_max = 1.8


def get_hr(ROI, fps):
    signal_handler = Handler(ROI)
    blue, green, red = signal_handler.get_channel_signal()
    matrix = np.array([blue, green, red])
    component = signal_handler.ICA(matrix, 3)
    fft, freqs = fft_filter.fft_filter(component[0], freqs_min, freqs_max, fps)
    heartrate_1 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[1], freqs_min, freqs_max, fps)
    heartrate_2 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[2], freqs_min, freqs_max, fps)
    heartrate_3 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    return (heartrate_1 + heartrate_2 + heartrate_3) / 3


if __name__ == '__main__':
    # video_path = 'videos/rohin_face.mov'
    ROI = []
    heartrate = 0
    camera_code = 0
    capture = cv.VideoCapture(camera_code)
    fps = capture.get(cv.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    while capture.isOpened():
        ret, frame = capture.read()
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
            roi = frame[top + h // 10 * 2:top + h // 10 * 7, left + w // 9 * 2:left + w // 9 * 8]
            cv.rectangle(frame, (left + w // 9 * 2, top + h // 10 * 3), (left + w // 9 * 8, top + h // 10 * 7),
                         color=(0, 0, 255))
            cv.rectangle(frame, (left, top), (left + w, top + h), color=(0, 0, 255))
            ROI.append(roi)
            if len(ROI) == 300:
                heartrate = get_hr(ROI, fps)
                for i in range(30):
                    ROI.pop(0)
        cv.putText(frame, '{:.1f}bps'.format(heartrate), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
