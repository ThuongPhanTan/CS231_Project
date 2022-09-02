import cv2
import time
import cvzone
import sys
import os
import numpy as np
# import dlib
from math import hypot
from PIL import Image
import mediapipe as mp
from PIL import Image, ImageDraw

class Effects(object):
    def __init__(self, 
                
                record_directory_name="recorded_videos", 
                record_name="recorded_video.mp4", 
                time_capture=10, 
                import_pic_path="import_pics/dh_cntt.jpg", 
                import_video_path=None, 
                effects=None):
        
        self.record_directory_name = record_directory_name
        self.record_name = record_name
        self.time_capture = time_capture
        self.import_pic_path = import_pic_path
        self.import_video_path = import_video_path
        self.effects = effects
        self.record_screen_shape = (640, 480) # (width, height)
    def alphaBlend(self, alpha, foreground, background):
        fore = np.zeros(foreground.shape, dtype=foreground.dtype)
        fore = cv2.multiply(alpha, foreground, fore, 1 / 255.0)
        alphaPrime = np.ones(alpha.shape, dtype=alpha.dtype) * 255 - alpha
        back = np.zeros(background.shape, dtype=background.dtype)
        back = cv2.multiply(alphaPrime, background, back, 1 / 255.0)
        outImage = cv2.add(fore, back)
        return outImage
    def apply_color_to_mask(self,mask):
        # Get random lipstick color
        color = (42, 31, 192)
        b, g, r = cv2.split(mask)
        b = np.where(b > 0, color[0], 0).astype('uint8')
        g = np.where(g > 0, color[1], 0).astype('uint8')
        r = np.where(r > 0, color[2], 0).astype('uint8')
        return cv2.merge((b, g, r))

    def get_angle(self, p1, p2):
        if p2[1] <= p1[1]:
            y = np.abs(p1[1] - p2[1])
        else:
            y = p1[1] - p2[1]
        x = np.abs(p1[0] - p2[0])
        return np.rad2deg(np.arctan2(y, x))
    def record_video_capture(self):

        vid = cv2.VideoCapture(0)

        if not os.path.isdir(self.record_directory_name):
            os.mkdir(self.record_directory_name)
        video_name = os.path.join(self.record_directory_name, self.record_name)
        save_vid = cv2.VideoWriter(video_name, -1, 20.0, self.record_screen_shape)
        

        if self.effects == "time_warp_scan_horizontal":
            i = 0
            previous_frame = np.zeros((self.record_screen_shape[1], self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line = np.zeros((self.record_screen_shape[1], 1, 3), dtype="uint8")
            cyan_line[:] = (255, 255, 0)
            while (vid.isOpened() and i < self.record_screen_shape[0]):
                ret, frame = vid.read()
                if ret:
                    frame = cv2.flip(frame,1)
                    previous_frame[:, i, :] = frame[:, i, :]
                    effect_frame = np.hstack((previous_frame[:, :i, :], cyan_line, frame[:, i+1:, :]))
                    save_vid.write(effect_frame)

                    i += 1
                    try:
                        effect_frame= cv2.putText(effect_frame," Rec", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                        ret, buffer = cv2.imencode('.jpg',effect_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   

        elif self.effects == "time_warp_scan_vertical":
            i=0
            previous_frame_vertical=np.zeros((self.record_screen_shape[1], 
                                                        self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line_vertical=np.zeros((1,self.record_screen_shape[0], 3), dtype="uint8")
            cyan_line_vertical[:,:] = (255, 255, 0)
            while (vid.isOpened() and i  < self.record_screen_shape[1]):
                ret, frame = vid.read()
                if ret:
                    previous_frame_vertical[i, :, :] = frame[i, :, :]
                    effect_frame = np.vstack((previous_frame_vertical[:i,:, :],
                                                        cyan_line_vertical, frame[i+1:,:, :]))
                    save_vid.write(effect_frame)
                
                    i += 1
                    try:
                        effect_frame= cv2.putText(cv2.flip(effect_frame,1)," Rec", 
                                                        (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                        ret, buffer = cv2.imencode('.jpg',effect_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   

        elif self.effects =='lipstick':
            landmarks_bot=[  61,146,91,181,84, 17,314,405,321,375,291,324,318,402,317,14,87,178,88,95,78]
            landmarks_top = [191,185, 40,39,37,0, 267, 269, 270, 409,308, 415, 310, 311, 312,13,  82,81,80,191]
            mpFaceMesh = mp.solutions.face_mesh
            faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True,max_num_faces=4) 
            
            while  (vid.isOpened()):
                ret, frame = vid.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480)) 
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = faceMesh.process(rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            bot_lift = [None]*len(landmarks_bot)
                            top_lift = [None]*len(landmarks_top)
                            total = [None]*(len(landmarks_bot)+len(landmarks_top))
                            for lm_id, lm in enumerate(face_landmarks.landmark):
                                h, w, c = rgb.shape
                                x, y = int(lm.x * w), int(lm.y * h)
                                for i,j in enumerate(landmarks_top):
                                    if lm_id == j :
                                        top_lift[i] = (x,y)
                                        total[i] = (x,y)
                                for i,j in enumerate(landmarks_bot):
                                    if lm_id == j :
                                        bot_lift[i] = (x,y)
                                        total[len(landmarks_top)+i] = (x,y)
                            try:
                                total = np.array([total], np.int32)
                                top_lift = np.array([top_lift], np.int32)
                                mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=frame.dtype)
                                cv2.fillPoly(mask,pts=[top_lift],color=(255, 255, 255))
                                bot_lift = np.array([bot_lift], np.int32)
                                cv2.fillPoly(mask,pts=[bot_lift],color=(255, 255, 255))
                                maskHeight, maskWidth = mask.shape[0:2]
                                maskSmall = cv2.resize(mask, (600, int(maskHeight * 600.0 / maskWidth)))
                                maskSmall = cv2.dilate(maskSmall, (3, 3))
                                maskSmall = cv2.GaussianBlur(maskSmall, (5, 5), 0, 0)
                                mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
                                color_mask = self.apply_color_to_mask(mask)
                                (x, y, w, h) = cv2.boundingRect(total)
                                center = (int(x+w/2), int(y+h/2))
                                masked_lips = cv2.bitwise_and(frame, frame, mask=mask[:, :, 0])
                                output  = cv2.seamlessClone(masked_lips, color_mask, mask[:, :, 0], center, cv2.MIXED_CLONE)
                                frame = self.alphaBlend(mask, output, frame)
                            except:
                                pass
                    try:
                        frame= cv2.putText(frame," Rec", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass

        elif self.effects == "thug_life":
            thug_img = cv2.imread('static/media/thug.png',cv2.IMREAD_UNCHANGED) 
            thug_landmarks = [33,263, 267] 

            mpDraw = mp.solutions.drawing_utils
            mpFaceMesh = mp.solutions.face_mesh
            faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True,max_num_faces=4) 

            while (vid.isOpened()):
                ret, frame = vid.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = faceMesh.process(rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # get each landmark info
                            for lm_id, lm in enumerate(face_landmarks.landmark):
                                
                                # getting original value
                                h, w, c = rgb.shape
                                x, y = int(lm.x * w), int(lm.y * h)
                                
                                # calculating nose width
                                if lm_id == thug_landmarks[0]:
                                    left_eye = [x, y]
                                if lm_id == thug_landmarks[1]:
                                    right_eye = [x, y]
                                if lm_id == thug_landmarks[2]:
                                    lip = [x, y]
                                
                            width, height, _ = thug_img.shape
                            left_eye = np.array(left_eye).astype(int)
                            right_eye = np.array(right_eye).astype(int)
                            glass_img = thug_img[0:265,20:450]
                            smoke_img = thug_img[265:450,210:450]
                            left_eyeglass = np.array((100,182)).astype(int)
                            right_eyeglass = np.array((360,182)).astype(int)

                            rotation_angle = round(self.get_angle( left_eye, right_eye), 3)
                            inter_eye_distance = np.linalg.norm(left_eye - right_eye)
                            inter_lens_distance = np.linalg.norm(left_eyeglass - right_eyeglass)
                            ratio = inter_eye_distance / inter_lens_distance
                            w_final = int(glass_img.shape[1] * ratio)
                            h_final = int(glass_img.shape[0] * ratio)
                            w_final2 = int(smoke_img.shape[1] * ratio)
                            h_final2 = int(smoke_img.shape[0] * ratio)

                            resized_glasses = cv2.resize(glass_img, (w_final, h_final), interpolation=cv2.INTER_AREA)
                            smoke_img =  cv2.resize(smoke_img, (w_final2, h_final2), interpolation=cv2.INTER_AREA)

                            smoke_resized = np.array((215,270)).astype(int)
                            smoke_resized =  (smoke_resized * ratio).astype(int)
                        
                            left_lens_resized = (left_eyeglass * ratio).astype(int)
                            right_lens_resized = (right_eyeglass * ratio).astype(int)

                            canvas = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=frame.dtype)
                            x = left_eye[0] - left_lens_resized[0]
                            y = left_eye[1] - left_lens_resized[1]
                            w = resized_glasses.shape[1]
                            h = resized_glasses.shape[0]

                            # x2 = lip[0] - smoke_resized[0]
                            # y2 = lip[1] - smoke_resized[1]
                            x2 = lip[0] 
                            y2 = lip[1] 
                            w2 = smoke_img.shape[1]
                            h2 = smoke_img.shape[0]
                            
                            try:
                                canvas[y:y + h, x:x + w, :] = resized_glasses
                            except:
                                pass

                            M = cv2.getRotationMatrix2D((int(left_eye[0]),int(left_eye[1])),rotation_angle, 1.0)

                            rotated_canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))

                            try:
                                rotated_canvas[y2:y2 + h2, x2:x2 + w2, :] = smoke_img
                            except:
                                pass

                            rotated_canvas_rbg = cv2.cvtColor(rotated_canvas,cv2.COLOR_BGR2RGB)
                            b, g, r, a = cv2.split(rotated_canvas)
                            bgr_rotated_canvas = cv2.merge([b, g, r])
                            mask_rotated_canvas = cv2.merge([a, a, a])

                            maskHeight, maskWidth = mask_rotated_canvas.shape[0:2]
                            maskSmall = cv2.resize(mask_rotated_canvas, (400, int(maskHeight * 400.0 / maskWidth)))
                            maskSmall = cv2.GaussianBlur(maskSmall, (3, 3), 0, 0)
                            mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
                            bgr = cv2.bitwise_and(bgr_rotated_canvas, bgr_rotated_canvas, mask=a)
                            final = self.alphaBlend(mask, bgr, frame)    
                            frame = np.asarray(final)
                            save_vid.write(frame)

                    try:
                        frame= cv2.putText(frame," Rec", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass

        else:
            while (vid.isOpened()):
                ret, frame = vid.read()
                if ret:
                    save_vid.write(frame)

                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass   

        vid.release()
        save_vid.release()
        cv2.destroyAllWindows()


