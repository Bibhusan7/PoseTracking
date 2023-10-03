import cv2
import mediapipe as mp

class PoseTracking():
    def __init__(self,  mode=False, model_comp=1,smoothLms=True,enableSegmentation=False,smoothSegmentation=True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.model_comp = model_comp
        self.smoothLms = smoothLms
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRgb)
        # print(results.pose_landmarks)

        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
        return img


def main():
    cap = cv2.VideoCapture(1)
    tracker = PoseTracking()
    while True:
        success, img = cap.read()
        img = tracker.findPose(img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()