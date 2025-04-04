import torch
import cv2
import pygame
from buttonclass import Button


def get_font(size):
    return pygame.font.Font("/Users/themagendrans/Desktop/Yolo/TitleFont.ttf", size)

model = torch.hub.load('/Users/themagendrans/Desktop/Yolo/yolov5', 'custom', 
                       path='/Users/themagendrans/Desktop/Yolo/yolov5/runs/train/exp3/weights/best.pt', 
                       source='local')
newclasses = ['glass', 'cardboard', 'metal', 'plastic', 'styrofoam']
model.names = newclasses

def vision():
    video_path = 0  
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    waittime = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    
        
        results = model(rgb_frame)

        img = results.render()[0]  
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLOv5 Video Detection', img_bgr) 

        
        if not results.pandas().xyxy[0].empty:
            waittime += 1
            if waittime == 30:
                cap.release()
                cv2.destroyAllWindows()
                return img_bgr  
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 'q'
    cap.release()
    cv2.destroyAllWindows()


def afterdetection(newframe):
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    running = True
    while running:
        MenuMousePos = pygame.mouse.get_pos()
        exit = Button(image=pygame.image.load("/Users/themagendrans/Desktop/Yolo/images.jpg"), 
                      pos=(400, 550), 
                      input="exit", 
                      font=get_font(75), 
                      baseColor="White", 
                      hoverColor="#61f255")

        exit.changeColor(MenuMousePos)
        exit.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                  
            if event.type == pygame.MOUSEBUTTONDOWN:
                if exit.checkForInput(MenuMousePos):
                    running = False 
        
        frame_rgb = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB) 
        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)  
        frame_rgb = cv2.flip(frame_rgb, 1)  
        frame_rgb = cv2.resize(frame_rgb, (500, 800))  
        frame_surface = pygame.surfarray.make_surface(frame_rgb)

        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
    pygame.quit()
    return




while True:
    frame = vision()  
    if frame is not None:
        afterdetection(frame) 
    elif frame == 'q':
        break
