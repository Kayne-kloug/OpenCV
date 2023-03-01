import winsound, time, cv2, math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 이미지 합성 함수
def overlay(image, x, y, w, h, overlay_image): # 대상 영상 (3채널), 이미지 삽입좌표 (x,y), 삽입이미지 (width, height), 덮어씌울 이미지 (4채널)
    global height, width
    if (y-h > 0) and (x-w >0):
        if (y+h < height) and (x+w < width):
            alpha = overlay_image[:, :, 3] # BGRA
            mask_image_transparency = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
            # (255, 255)  ->  (1, 1)
            # (255, 0)        (1, 0)
            # 1 - mask_image ?
            # (0, 0)
            # (0, 1)
            
            for c in range(0, 3): # channel BGR
                print('c:',c)
                print(overlay_image.shape)
                print('x,y : ',x,y)
                print('h,w : ',h,w)
                print('mask_image:',mask_image_transparency.shape)
                print('image : ',image[y-h:y+h, x-w:x+w, c].shape)
                print('\n')
                image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image_transparency) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image_transparency))

#도,레,미 이미지
do = cv2.imread('img/do.png', cv2.IMREAD_UNCHANGED)
re = cv2.imread('img/re.png', cv2.IMREAD_UNCHANGED)
mi = cv2.imread('img/mi.png', cv2.IMREAD_UNCHANGED)


# 도,레,미 오디오
def freq(o, s):
    if s == '도':     return 524*2**o
    elif s == '레':   return 587*2**o
    elif s == '미':   return 659*2**o
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


## 카메라 출력
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0번째 카메라 장치 (Device ID)
if not cap.isOpened(): # 카메라가 잘 열리지 않은 경우
    exit() # 프로그램 종료

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 프레임 크기, FPS
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 정수값 처리를 위한 round함수 이용
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 저장 파일명, 코덱, FPS, 크기 (width, height)
out = cv2.VideoWriter('motion_camera.avi', fourcc, fps/10, (width, height))

#녹화 플래그
capture_video = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            x1 = hand_landmarks.landmark[4].x # 엄지끝 x좌표
            y1 = hand_landmarks.landmark[4].y # 엄지끝 y좌표
            x2 = hand_landmarks.landmark[8].x # 검지끝 x좌표
            y2 = hand_landmarks.landmark[8].y # 검지끝 y좌표
            middle_pip_y = hand_landmarks.landmark[9].y # 중지마디 y좌표
            x3 = hand_landmarks.landmark[12].x # 중지끝 y좌표
            y3 =hand_landmarks.landmark[12].y # 중지끝 y좌표
            x4 = hand_landmarks.landmark[16].x # 약지끝 x좌표
            y4 = hand_landmarks.landmark[16].y # 약지끝 y좌표
            aa = abs(x1-x2) # 엄지 검지 x좌표 거리
            bb = abs(y1-y2) # 엄지 검지 y좌표 거리
            cc = abs(x1-x3)
            dd = abs(y1-y3)
            ee = abs(x1-x4)
            ff = abs(y1-y4)
            len1 = math.sqrt((aa*aa) + (bb*bb)) # 엄지 검지 거리
            len2 = math.sqrt((cc*cc) + (dd*dd))
            len3 = math.sqrt((ee*ee) + (ff*ff))
            if middle_pip_y - y3 < -0.1 :
                color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil=Image.fromarray(color_coverted)
                draw = ImageDraw.Draw(frame_pil)
                # draw.text((10, 10),  "3!", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,255,0))
                # time.sleep(1)
                # draw.text((10, 10),  "2!", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,255,0))
                # time.sleep(1)
                # draw.text((10, 10),  "1!", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,255,0))
                # time.sleep(1)
                draw.text((10, 10),  "녹화시작!", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,255,0))
                capture_video = 1
                numpy_frame = np.array(frame_pil)
                frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
                print("녹화를 시작합니다")
            if capture_video ==1:
                out.write(frame)
                if len1 < 0.03:
                    winsound.Beep(freq(0,'도'),200) ,print("도")
                    #도 이미지 추가
                    scale_size = round(480*abs(hand_landmarks.landmark[7].y-hand_landmarks.landmark[3].y))//2*16
                    # print(scale_size)
                    resize_do = cv2.resize(do, dsize=(scale_size,scale_size), interpolation=cv2.INTER_AREA)
                    overlay(frame,round(x1*640), round(y1*480), scale_size//2,scale_size//2, resize_do )
                    # print(round(x1*640), round(y1*480))
                    # print(hand_landmarks.landmark[7].x*640, hand_landmarks.landmark[7].y*480)
                elif len2 < 0.03:
                    winsound.Beep(freq(0,'레'),200) ,print("레")
                    #레 이미지 추가
                    scale_size = round(480*abs(hand_landmarks.landmark[11].y-hand_landmarks.landmark[3].y))//2*16
                    print(scale_size)
                    resize_re = cv2.resize(re, dsize=(scale_size,scale_size), interpolation=cv2.INTER_AREA)
                    overlay(frame,round(x1*640), round(y1*480), scale_size//2,scale_size//2, resize_re )
                    # print(round(x1*640), round(y1*480))
                    print(hand_landmarks.landmark[11].x*640, hand_landmarks.landmark[7].y*480)
                elif len3 < 0.03:
                    winsound.Beep(freq(0,'미'),200) ,print("미")
                    #미 이미지 추가
                    scale_size = round(480*abs(hand_landmarks.landmark[15].y-hand_landmarks.landmark[3].y))//2*16
                    print(scale_size)
                    resize_mi = cv2.resize(mi, dsize=(scale_size,scale_size), interpolation=cv2.INTER_AREA)
                    overlay(frame,round(x1*640), round(y1*480), scale_size//2,scale_size//2, resize_mi )
                    # print(round(x1*640), round(y1*480))
                    print(hand_landmarks.landmark[15].x*640, hand_landmarks.landmark[7].y*480)
                # if x4-x3 < 0.001 and x3-x2 < 0.001:
                #     out.release()
                #     break
        # print(frame)
        # print(frame.shape)
        cv2.imshow('camera', frame)
        cv2.waitKey(1)
#out.release()
cap.release()
cv2.destroyAllWindows()