import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ArmedPersonDetector:
    def __init__(self, pose_model_path, gun_model_path, video_path, save_path, conf):
        self.pose_model = YOLO(pose_model_path)
        self.gun_model = YOLO(gun_model_path)
        self.video_path = video_path
        self.save_path = save_path
        self.conf = conf

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            pose_result = self.pose_model.predict(frame, verbose=False)[0]
            gun_result = self.gun_model.predict(frame, verbose=False, conf = self.conf)[0]

            guns_idx = torch.where(gun_result.boxes.cls == 0)[0]

            gun_boxes = gun_result.boxes.xyxy[guns_idx]

            # Compute gun centers
            gun_centers = []
            for box in gun_boxes:
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                gun_centers.append((cx, cy))

            for gun_id, (gx, gy) in enumerate(gun_centers):
                closest_wrist_box = None
                min_dist = float("inf")
                idx = None

                for i, kps in enumerate(pose_result.keypoints.xy):
                    kps_np = kps.cpu().numpy()
                    if kps_np.shape[0] < 11:
                        continue
                    right_wrist, left_wrist = kps_np[9], kps_np[10]
                    for wrist in [right_wrist, left_wrist]:
                        xw, yw = wrist
                        dist = np.linalg.norm([gx - xw, gy - yw])
                        if dist < min_dist:
                            min_dist = dist
                            idx = i
                            closest_wrist_box = pose_result.boxes.xyxy[i]

                if closest_wrist_box is None :
                    continue

                

                # Draw gun box
                gx1, gy1, gx2, gy2 = gun_boxes[gun_id].tolist()
                cv2.rectangle(frame, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 255), 2)
                cv2.putText(frame, "Gun", (int(gx1), int(gy1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw armed person box
                px1, py1, px2, py2 = closest_wrist_box.tolist()
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)


                # Blinking danger text
                if (frame_count // 15) % 2 == 0:
                    cv2.putText(frame, "!!! ARMED !!!", (int(px1), int(py1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Detection complete. Saved video to: {self.save_path}")
