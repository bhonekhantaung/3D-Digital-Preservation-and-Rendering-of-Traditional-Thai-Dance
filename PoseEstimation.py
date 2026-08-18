import mediapipe as mp
import cv2
import time

# 1. ADDED THESE TWO LINES TO FIX THE MAC SSL DOWNLOAD ERROR
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. CHANGED 'results' to 'self.results' SO THE OTHER FUNCTION CAN USE IT
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

        # 3. ADDED RETURN STATEMENT SO THE LIST CAN BE PRINTED IN MAIN
        return lmList


def main():
    cap = cv2.VideoCapture('PoseVideos/thai_dance_testing.mp4')
    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()

        # 4. MOVED THIS UP: Stop immediately if the video ends, otherwise it crashes
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()