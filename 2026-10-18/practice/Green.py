import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
MIN_RADIUS = 10

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # TODO: 각 프레임 이미지를 전처리
    frame_blur = cv2.GaussianBlur(frame,(7,7),0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    # TODO: Color Segmentation으로 Binary Mask 생성
    mask = cv2.inRange(frame_lab,green_lower,green_upper)

    # TODO: Morphological Operation을 적용하여 객체 영역 추출
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > MIN_RADIUS:
            center = (int(x),int(y))
            cv2.circle(frame, center, int(radius),(0,255,0),2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
            cv2.putText(frame, f"{center} r={int(radius)}", (center[0] + 10, center[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    
    cv2.imshow("Binary Mask",mask)
    cv2.imshow("Real-time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

