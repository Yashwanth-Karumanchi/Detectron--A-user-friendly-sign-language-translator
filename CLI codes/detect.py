import json
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import pyttsx3
import argparse
import pygame

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LeakyReLU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import wordninja
from googletrans import Translator
import gtts
from gtts import gTTS

class Detect_Signs:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence_length = 30
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 120)
        
    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    def get_folder_names(self, directory):
        folder_names = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        return folder_names
    
    def model_build(self, model_name):
        model = load_model(model_name)
        
        return model

    def prob_viz(self, res, data, input_frame, colors):
        output_frame = input_frame.copy()
        detected_index = np.argmax(res)
        cv2.rectangle(output_frame, (0,60*40), ((100), 90*40), colors[0], -1)
        cv2.putText(output_frame, f'{data[detected_index]}: {res[detected_index]:.2f}', (0, 85+5*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return output_frame
    
    def detect(self, folders, model, colors):
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.6
        prediction_start_time = None
        last_added_time = 0.0
        repeat_threshold = 3.0
        popped_element = ''
        display=[]

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                cam = int(input("Default camera is not recognized. Please specify the camera device number to use that camera: "))
                if not cap.isOpened():
                    print("Error: No valid camera found. Please make sure you have a working camera connected.")

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_landmarks(image, results)

                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = self.extract_keypoints(results)
                    if any(keypoints):
                        sequence.append(keypoints)
                        sequence = sequence[-30:]

                    if len(sequence) == self.sequence_length: 
                        if prediction_start_time is None:
                            prediction_start_time = time.time()
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predictions.append(np.argmax(res))

                        current_time = time.time()
                        if np.unique(predictions[-10:])[0] == np.argmax(res): 
                            if res[np.argmax(res)] > threshold and folders[np.argmax(res)] != popped_element:
                                if len(sentence) > 0: 
                                    latest_prediction = folders[np.argmax(res)]
                                    last_element = sentence[-1]
                                    if latest_prediction != last_element or (len(latest_prediction)==1 
                                        and sentence[-2:] != [latest_prediction, latest_prediction] and current_time - last_added_time >= repeat_threshold):
                                        sentence.append(latest_prediction)
                                        latest_element = sentence[-1]
                                        last_added_time = current_time

                                else:
                                    sentence.append(folders[np.argmax(res)])
                                    latest_element = sentence[-1]
                                    last_added_time = current_time

                                self.engine.say(latest_element)
                                self.engine.runAndWait()

                                display = sentence[-6:]
                                popped_element = ''
                                
                        sequence.clear()
                        predictions.clear()
                        
                        image = self.prob_viz(res, folders, image, colors)
                         
                cv2.rectangle(image, (0, 0), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 40), (0, 0, 0), -1)
                cv2.putText(image, ' '.join(display), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                font_scale = 0.4
                font_thickness = 1

                text = 'Press: Q -> Speech Generation. E -> EXIT. BACKSPACE -> clear. R -> UNDO clear'
                text_position = (30, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                background_position1 = (text_position[0] - 10, text_position[1] - text_height - 5)
                background_position2 = (text_position[0] + text_width + 10, text_position[1] + 5)
                cv2.rectangle(image, background_position1, background_position2, (0, 0, 0), -1)
                cv2.putText(image, text, text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)
                
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break
                elif key == ord('e') or cv2.getWindowProperty('OpenCV Feed', cv2.WND_PROP_VISIBLE) < 1:
                    exit()
                elif key == 8: 
                    if len(sentence) > 0:
                        popped_element = sentence.pop()
                        display = sentence[-6:]
                elif key == ord('r'):
                    if popped_element != '':
                        sentence.append(popped_element)
                        popped_element = ''
                        display = sentence[-6:]

            cap.release()
            cv2.destroyAllWindows()
            
            return sentence
    
    def text_generation(self, sentence):
        text = ''.join(map(str, sentence))
        words = wordninja.split(text)
        text = ' '.join(map(str, words))
        return text

    def text_to_speech(self, text, language, counter):
        translator = Translator()
        if len(text) > 0:
            translation = translator.translate(text, src='en', dest=language)
            text = translation.text
            tts = gTTS(text=text, lang=language)
            file_name = 'output'+str(counter)+'.mp3'
            tts.save(os.path.join('./audio_files', file_name))
            
            pygame.mixer.init()
            pygame.mixer.music.load(os.path.join('./audio_files', file_name))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.quit()
        
    def get_arguments(self):
        parser = argparse.ArgumentParser(description="Detect the signs.")
        parser.add_argument("--data", type=str, default='../data', help="Relative location of the data folder")
        parser.add_argument("--model", type=str, default='../models/model.h5', help="Trained model file relative location")
        parser.add_argument("--lang", type=str, default='english', help="Language to be delivered")

        args = parser.parse_args()
        return args
    
    
def main():
    detector = Detect_Signs()
    args = detector.get_arguments()
    DATA_PATH = os.path.join(args.data)
    model_path = os.path.join(args.model)
    save_text_path = os.path.join('./text_files')
    save_audio_path = os.path.join('./audio_files')
    with open('../supportedLanguages.json', 'r') as file:
        languages = json.load(file)
        
    counter = 0
    if os.path.exists(save_audio_path) == False:
        os.makedirs(save_audio_path)
    
    if os.path.exists(save_text_path) == False:
        os.makedirs(save_text_path)
    
    if os.path.exists(DATA_PATH) == False:
        print(f"NO {DATA_PATH} FOLDER FOUND")
        exit()
    
    if os.path.exists(model_path) == False:
        print(f"NO {model_path} FOLDER FOUND")
        exit()

    if args.lang.lower() not in languages:
        print(f"{args.lang.lower()} LANGUAGE IS NOT AVAILABLE. AVAILABLE CODES ARE: \n\n {gtts.lang.tts_langs()}")
        exit()
    
    folders = np.array(detector.get_folder_names(DATA_PATH))
    colors = [(245,117,16)]
    model = detector.model_build(folders, args.model)
    while(True):
        counter += 1
        sentence = detector.detect(folders, model, colors)
        text = detector.text_generation(sentence)
        file_name = 'run'+str(counter)+'.txt'
        with open(os.path.join(save_text_path, file_name), 'w') as file:
            file.write(text)
        detector.text_to_speech(text, args.lang, counter)
        print("Detected times: ", counter)

if __name__ == '__main__':
    main()