import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
size = [400, 300]

actions = ['up', 'down']
seq_length = 30

model = load_model('models/model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():

    screen = pygame.display.set_mode(size)
    pygame.init()
    done = False
    #clock = pygame.time.Clock()

    screen_width = size[0]
    screen_height = size[1]

    bar_width = 9
    bar_height = 50

    bar_x = bar_start_x = 0
    bar_y = bar_start_y = (screen_height - bar_height) / 2  ## 탁구채의 왼쪽 아래 모서리가 기준

    circle_radius = 9
    circle_diameter = circle_radius * 2

    circle_x = circle_start_x = screen_width - circle_diameter
    circle_y = circle_start_y = (screen_width - circle_diameter) / 2  ## 원을 둘러싼 정사각형의 왼쪽 아래 모서리가 기준

    bar_move = 0
    speed_x, speed_y, speed_bar = 0, 0, screen_height ##-screen_width / 1.28, screen_height / 1.92,

    ret, img = cap.read() #프레임 잘 읽으면 ret = true

    img = cv2.flip(img, 1) #image flip 0 : updown 1: leftright
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR -> RGB
    result = hands.process(img) #RGB으로 처리
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #RGB -> BGR

    if result.multi_hand_landmarks is not None:
        hands_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]


            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0) #expand datas

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred)) #indice (index) of the prediction
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]: #if last 3 motions are same
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

            while not done:
                #time_passed = clock.tick(60)  ## 초당 프레임수 : 60
                #time_sec = time_passed / 1000.0
                time_sec = 1
                screen.fill(BLACK)

                circle_x += speed_x * time_sec
                circle_y += speed_y * time_sec
                ai_speed = speed_bar * time_sec

                if this_action == 'up' :
                    bar_move = -ai_speed
                elif this_action == 'down':
                    bar_move = ai_speed

                ## 탁구채 이동
                bar_y += bar_move

                ## 탁구채 범위 확인
                if bar_y >= screen_height - bar_height:
                    bar_y = screen_height - bar_height
                elif bar_y <= 0:
                    bar_y = 0

                ## 탁구공 범위 확인
                ## 1) 진행 방향을 바꾸는 행위
                ## 2) 게임이 종료되는 행위
                if circle_x < bar_width:  ## bar에 닿았을 때
                    if circle_y >= bar_y - circle_radius and circle_y <= bar_y + bar_height + circle_radius:
                        circle_x = bar_width
                        speed_x = -speed_x
                if circle_x < -circle_radius:  ## bar에 닿지 않고 좌측 벽면에 닿았을 때, 게임 종료 및 초기화
                    circle_x, circle_y = circle_start_x, circle_start_y
                    bar_x, bar_y = bar_start_x, bar_start_y
                elif circle_x > screen_width - circle_diameter:  ## 우측 벽면에 닿았을 때
                    speed_x = -speed_x
                if circle_y <= 0:  ## 위측 벽면에 닿았을때
                    speed_y = -speed_y
                    circle_y = 0
                elif circle_y >= screen_height - circle_diameter:  ## 아래 벽면에 닿았을때
                    speed_y = -speed_y
                    circle_y = screen_height - circle_diameter

                ## 탁구채
                pygame.draw.rect(screen,
                                 WHITE,
                                 (bar_x, bar_y, int(bar_width), int(bar_height)))
                ## 탁구공
                pygame.draw.circle(screen,
                                   WHITE,
                                   (int(circle_x), int(circle_y)),
                                   int(circle_radius))

                pygame.display.update()

    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break