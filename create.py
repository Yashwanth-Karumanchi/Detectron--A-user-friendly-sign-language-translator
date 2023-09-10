import cv2
import numpy as np
import os
import mediapipe as mp
import shutil

class Dataset_Creation:    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.no_sequences = 30
        self.sequence_length = 30

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

    def create_folders(self, DATA_PATH, data):        
        for folder in data:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, folder, str(sequence)))
                except:
                    pass
                
    def create_folders_specified(self, DATA_PATH, extend):
        for sequence in range(self.no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, extend, str(sequence)))
            except:
                pass


    def get_folder_names(self, directory):
        folder_names = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        return folder_names

    def collect_data(self, folders, DATA_PATH):
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            for folder in folders:
                length = self.count_directories(os.path.join(DATA_PATH, folder))
                for sequence in range(length):
                    if(not any(os.scandir(os.path.join(DATA_PATH, folder, str(sequence))))):
                        for frame_num in range(self.sequence_length):
                            ret, frame = cap.read()

                            image, results = self.mediapipe_detection(frame, holistic)

                            self.draw_landmarks(image, results)

                            if frame_num == 0: 
                                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(folder, sequence), (15,12), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                cv2.imshow('OpenCV Feed', image)
                                cv2.waitKey(2000)
                            else: 
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(folder, sequence), (15,12), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                cv2.imshow('OpenCV Feed', image)


                            keypoints = self.extract_keypoints(results)
                            npy_path = os.path.join(DATA_PATH, folder, str(sequence), str(frame_num))
                            np.save(npy_path, keypoints)


                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

        cap.release()
        cv2.destroyAllWindows()
    
    def count_directories(self, folder_path):
        count = 0
        for item in os.listdir(folder_path):
                count += 1
        return count  
        
def main():
    path = input("Enter relative path to the Data Folder:")
    DATA_PATH = os.path.join(path)
    try:
        os.makedirs(DATA_PATH)
        print(f"Folder {DATA_PATH} created")
    except:
        pass
    create = Dataset_Creation()
    
    data = create.get_folder_names(DATA_PATH)
    data = [element.lower() for element in data]
    while(True):
        action = input("[~] Choose an Option\n [1] Add Data\n [2] Replace Data\n [3] Extend existing Data\n [4] Exit\n")

        if action == '1':
            addition = input("Enter new symbol or gesture: ")
            if addition.lower() not in data:
                data.append(addition.lower())
                create.create_folders(DATA_PATH, data)
                folders = np.array(create.get_folder_names(DATA_PATH))
                print("Collecting Data")
                create.collect_data(folders, DATA_PATH)
                print("DATA SUCCESSFULLY ADDED")
                cv2.destroyAllWindows()
            else:
                print("DATA ALREADY EXISTS OR NO DATA IS ADDED. TRY REPLACING!")
        elif action == '2':
            replace = input("Enter symbol or gesture to replace: ")
            if replace.lower() in data:
                shutil.rmtree(os.path.join(DATA_PATH, replace))
                create.create_folders(DATA_PATH, data)
                folders = np.array(create.get_folder_names(DATA_PATH))
                print("Collecting Data")
                create.collect_data(folders, DATA_PATH)
                print("SUCCESSFULLY REPLACED")
                cv2.destroyAllWindows()
            else:
                print("NO RECORD FOUND TO REPLACE. TRY ADDING!")
        elif action == '3':
            extend = input("Enter symbol or gesture to replace: ")
            if extend.lower() in data:
                length = create.count_directories(os.path.join(DATA_PATH, extend.lower()))
                create.no_sequences = length + 30
                create.create_folders_specified(DATA_PATH, extend.lower())
                folders = np.array(create.get_folder_names(DATA_PATH))
                print("Collecting Data")
                create.collect_data(folders, DATA_PATH)
                print("SUCCESSFULLY EXTENDED")
                create.no_sequences = 30
                cv2.destroyAllWindows()
            else:
                print("NO RECORD FOUND TO EXTEND. TRY ADDING!")
        elif action == '4':
            break

if __name__ == '__main__':
    main()