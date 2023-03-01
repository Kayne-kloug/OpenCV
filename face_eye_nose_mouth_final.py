import dlib
import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def overlay(image, x, y, w, h, overlay_image): # 대상 영상 (3채널), 이미지 삽입좌표 (x,y), 삽입이미지 (width, height), 덮어씌울 이미지 (4채널)
    global frame_height, frame_width
    if (y-h > 0) and (x-w >0):
        if (y+h < frame_height) and (x+w < frame_width):
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

def face_expression(a,b,c,d,x1,y1,x2,y2, x3,y3):
    print('입꼬리 좌표 : ',a,b,c,d)
    print('윗 속입술 3점 좌표 : ',x1,y1,x2,y2,x3,y3)
    if (y1 > ((d-b)/(c-a))*x1 + (b- a*((d-b)/(c-a)))) and   (y2 > ((d-b)/(c-a))*x2 + (b- a*((d-b)/(c-a)))) and   (y3 > ((d-b)/(c-a))*x3 + (b- a*((d-b)/(c-a)))):
        return 1
    else :
        return 0

# range는 끝값이 포함안됨   
ALL = list(range(0, 68)) 
JAWLINE = list(range(0, 17))
RIGHT_EYEBROW = list(range(17, 22))  
LEFT_EYEBROW = list(range(22, 27))  
NOSE = list(range(27, 36))  
RIGHT_EYE = list(range(36, 42))  
LEFT_EYE = list(range(42, 48))  
MOUTH_OUTLINE = list(range(48, 61))  
MOUTH_INNER = list(range(61, 68)) 

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
#cap = cv.VideoCapture('video/hataeho.mp4')
#cap = cv.VideoCapture('video/class_exodus.mp4')
#image = cv.imread('img/APINK.jpg', cv.IMREAD_COLOR)
#image = cv.imread('img/dahyun.jpg', cv.IMREAD_COLOR)
#image = cv.imread('img/IM.png', cv.IMREAD_COLOR)
#image = cv.imread('img/smile.png', cv.IMREAD_COLOR)

doggy_ear = cv.imread('img/doggy_ear.png', cv.IMREAD_UNCHANGED)
doggy_nose = cv.imread('img/doggy_nose.png', cv.IMREAD_UNCHANGED)
rabbit_ear = cv.imread('img/rabbit_ear.png', cv.IMREAD_UNCHANGED)
sunglass_1 = cv.imread('img/sunglass_1.png', cv.IMREAD_UNCHANGED)
sunglass_2 = cv.imread('img/sunglass_2.png', cv.IMREAD_UNCHANGED)

ear_emoji_list = [0,doggy_ear, rabbit_ear]
eye_emoji_list = [0, sunglass_1, sunglass_2]
nose_emoji_list = [0, doggy_nose]

ear_emoji_index = 0
eye_emoji_index = 0
nose_emoji_index = 0

index = ALL
smile_flag=1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

while True:

    ret, img_frame = cap.read()
    #img_frame = image
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)           # 얼굴 영역 좌표 저장
    # print('faces : ',faces)
    # print(type(faces))
    print('\n')
    frame_width = img_frame.shape[1]
    frame_height = img_frame.shape[0]
    print(img_frame.shape[0])
    print(img_frame.shape[1])
    for face in faces:
        shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])
        list_points = np.array(list_points)
        for i,pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (255, 255, 255), -1) # 얼굴 포인트

            if i==27 and eye_emoji_index>0:   # 미간
                #cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
                print((list_points[16][0]-list_points[0][0])//2*2)
                print((list_points[19][1]-list_points[6][1])//32*2)
                emoji_eye = cv.resize(eye_emoji_list[eye_emoji_index], dsize=((list_points[16][0]-list_points[0][0])//2*2, (list_points[6][1]-list_points[19][1])//8*2), interpolation=cv.INTER_AREA)
                (H, W), (h, w) = img_frame.shape[:2], emoji_eye.shape[:2]
                x, y = pt_pos[0]-(w//2), pt_pos[1]-(h//2)
                overlay(img_frame, pt_pos[0],pt_pos[1], emoji_eye.shape[1]//2, emoji_eye.shape[0]//2, emoji_eye)

            if i==27 and ear_emoji_index>0:     # 미간
                emoji_ear = cv.resize(ear_emoji_list[ear_emoji_index], dsize=((list_points[16][0]-list_points[0][0])//2*2, (list_points[45][0]-list_points[36][0])//2*2), interpolation=cv.INTER_AREA)
                (H, W), (h, w) = img_frame.shape[:2], emoji_ear.shape[:2]
                x, y = pt_pos[0]-(w//2), pt_pos[1]-(h//2)
                overlay(img_frame, pt_pos[0],pt_pos[1]-emoji_ear.shape[1], emoji_ear.shape[1]//2, emoji_ear.shape[0]//2, emoji_ear)
            
            if i == 30 and nose_emoji_index>0:  # 코 
                emoji_nose = cv.resize(nose_emoji_list[nose_emoji_index], dsize=((list_points[16][0]-list_points[0][0])//4*6, (list_points[33][1]-list_points[29][1])//2*4), interpolation=cv.INTER_AREA)
                (H, W), (h, w) = img_frame.shape[:2], emoji_nose.shape[:2]
                x, y = pt_pos[0]-(w//2), pt_pos[1]-(h//2)
                overlay(img_frame, round(pt_pos[0]+emoji_nose.shape[0]*0.1),pt_pos[1], emoji_nose.shape[1]//2, emoji_nose.shape[0]//2, emoji_nose)

        #cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        smile_flag = face_expression(*list_points[48],*list_points[54],*list_points[61],*list_points[62],*list_points[63])
        print(face_expression(*list_points[48],*list_points[54],*list_points[61],*list_points[62],*list_points[63]))
        print(smile_flag)




    color_coverted = cv.cvtColor(img_frame, cv.COLOR_BGR2RGB)
    img_frame_pil=Image.fromarray(color_coverted)

    # PIL 이미지에 한글 입력
    draw = ImageDraw.Draw(img_frame_pil)
    if smile_flag ==True:
        draw.text((10, 10),  "스마일", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,212, 0))
    else:
        draw.text((10, 10),  "노스마일", font=ImageFont.truetype("./malgun.ttf", 48), fill=(255,212,0))

    # PIL 이미지 -> cv2 Mat 타입으로 변경
    numpy_frame = np.array(img_frame_pil)
    cv_frame = cv.cvtColor(numpy_frame, cv.COLOR_RGB2BGR)
    
    img_frame = cv.resize(cv_frame, dsize=(1200,900), interpolation=cv.INTER_CUBIC)
    cv.imshow('result', img_frame)
    
    key = cv.waitKey(1)

    if key == 27:                   # ESC 키 누르면 중지
        break
    elif key == ord('0'):           # 이모지 끄기
        ear_emoji_index = 0
        eye_emoji_index = 0
        nose_emoji_index = 0
    elif key == ord('1'):           # 개
        ear_emoji_index = 1
        nose_emoji_index = 1
    #     index = ALL
    elif key == ord('2'):             # 토끼
        ear_emoji_index = 2
        nose_emoji_index = 1
    #     index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):                       # 1번 선글라스
        eye_emoji_index = 1
    #     index = LEFT_EYE + RIGHT_EYE
    #
    elif key == ord('4'):                       # 2번 선글라스
        eye_emoji_index = 2
    #     index = NOSE
    # elif key == ord('5'):                       # 5번 키 누르면 입술 출력
    #     index = MOUTH_OUTLINE+MOUTH_INNER
    # elif key == ord('6'):                       # 6번 키 누르면 턱선 출력
    #     index = JAWLINE



#cap.release()