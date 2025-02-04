import cv2
from ultralytics import YOLO
import os

def main():
    model_path = 'best.pt'
    input_video_path = 'videos.mp4'
    output_video_path = 'output_video.mp4'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Không thể mở video: {input_video_path}")
        return

    # Lấy thông số video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Định nghĩa codec và tạo VideoWriter để lưu video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()
        cv2.imshow('YOLOv8n Detection', annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
