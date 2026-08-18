import json
import ssl
import cv2
import mediapipe as mp
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

class PoseDetector:

    def __init__(self, mode=False, upBody=False,smooth=True, detectionCon=0.5,trackCon=0.5,):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon,
                                     )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return lmList

    def find3DPosition(self):
        lmList3D = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(
                self.results.pose_world_landmarks.landmark
            ):
                lmList3D.append({
                    "id": id,
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                })
        return lmList3D


def calculate_bone_vector(p1, p2):
    v = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"], p2["z"] - p1["z"]])
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0]


def main():
    # Load static image
    img = cv2.imread("PoseImages/thai_dance_img_test2.jpg")

    if img is None:
        print("Error: Could not find or read the image.")
        return

    detector = PoseDetector(mode=True)
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    lmList3D = detector.find3DPosition()

    if lmList3D:
        pose_payload = {
            "metadata": {
                "source_image": "thai_dance_img_test2.jpg",
                "landmark_count": len(lmList3D),
            },
            "world_landmarks": lmList3D,
            "bone_vectors": {
                "right_upper_arm": calculate_bone_vector(
                    lmList3D[12], lmList3D[14]
                ),  # R_Shoulder -> R_Elbow
                "right_forearm": calculate_bone_vector(
                    lmList3D[14], lmList3D[16]
                ),  # R_Elbow -> R_Wrist
            },
        }

        with open("dance_pose.json", "w") as f:
            json.dump(pose_payload, f, indent=4)
            print("Exported pose data to dance_pose.json successfully!")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()