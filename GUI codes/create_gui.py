import cv2
import numpy as np
import os
import mediapipe as mp
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

class Dataset_Creation:    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.no_sequences = 30
        self.sequence_length = 30

        self.master = tk.Tk()
        self.master.title("Dataset Creation")
        self.master.geometry("400x300")

        self.label_data = tk.Label(self.master, text="Data Folder:")
        self.label_data.pack(pady=(10, 0))

        self.entry_data = tk.Entry(self.master, width=30)
        self.entry_data.pack(pady=(0, 10))

        self.btn_browse_data = tk.Button(self.master, text="Choose data folder", command=self.browse_data)
        self.btn_browse_data.pack(pady=(10, 0))

        self.btn_add_data = tk.Button(self.master, text="Add Data", command=self.add_data)
        self.btn_add_data.pack(pady=(20, 0))

        self.btn_replace_data = tk.Button(self.master, text="Replace Data", command=self.replace_data)
        self.btn_replace_data.pack(pady=(10, 0))

        self.btn_extend_data = tk.Button(self.master, text="Extend Data", command=self.extend_data)
        self.btn_extend_data.pack(pady=(10, 0))

        self.btn_exit = tk.Button(self.master, text="Exit", command=self.master.quit)
        self.btn_exit.pack(pady=(20, 0))
        
        self.set_default_values()
    
    def set_default_values(self):
        # Set default paths here
        self.entry_data.insert(0, '../data')

    def browse_data(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_data.delete(0, tk.END)
            self.entry_data.insert(0, folder)

    def add_data(self):
        data_path = self.entry_data.get()
        os.makedirs(data_path, exist_ok=True)
        data = self.get_folder_names(data_path)
        data = [element.lower() for element in data]

        new_symbol = simpledialog.askstring("Input", "Enter new symbol or gesture:")
        if new_symbol is not None and new_symbol != "" and new_symbol.lower() not in data:
            new_symbol = new_symbol.lower()
            self.create_folders(data_path, new_symbol)

            print(f"Collecting Data for {new_symbol}")
            self.collect_data([new_symbol], data_path)
            messagebox.showinfo("Addition Complete", "Data has been added successfully!")
        else:
            messagebox.showinfo("Addition Incomplete", "Invalid input or no data is added!")

    def replace_data(self):
        data_path = self.entry_data.get()
        os.makedirs(data_path, exist_ok=True)
        data = self.get_folder_names(data_path)
        data = [element.lower() for element in data]

        replace = simpledialog.askstring("Input", "Enter symbol or gesture to replace:")
        if replace is not None and replace != "" and replace.lower() in data:
            replace = replace.lower()
            if replace in os.listdir(data_path):
                shutil.rmtree(os.path.join(data_path, replace))
                self.create_folders(data_path, replace)
                print(f"Collecting Data for {replace}")
                self.collect_data([replace], data_path)
                messagebox.showinfo("Replace Complete", "Data has been replaced successfully!")
            else:
                messagebox.showinfo("Replace Inomplete", "No record found to replace. Try Adding!")
        else:
            messagebox.showinfo("Replace Inomplete", "Invalid field!")

    def extend_data(self):
        data_path = self.entry_data.get()
        os.makedirs(data_path, exist_ok=True)
        data = self.get_folder_names(data_path)
        data = [element.lower() for element in data]

        extend = simpledialog.askstring("Input", "Enter symbol or gesture to extend:")
        if extend is not None and extend != "" and extend.lower() in data:
            extend = extend.lower()
            extend_folder = os.path.join(data_path, extend)
            if os.path.exists(extend_folder):
                length = len(os.listdir(extend_folder))
                self.no_sequences = length + 30

                print(f"Extending Data for {extend}")
                self.create_folders_specified(data_path, extend)
                self.collect_data([extend], data_path)
                messagebox.showinfo("Extension Complete", f"Successfully extended to {self.no_sequences} sequences")
                self.no_sequences = 30
            else:
                messagebox.showinfo("Extension Inomplete", "No record found to extend. Try Adding!")
        else:
            messagebox.showinfo("Extension Inomplete", "Invalid field!")

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
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                cam = int(input("Default camera is not recognized. Please specify the camera device number to use that camera: "))
                if not cap.isOpened():
                    print("Error: No valid camera found. Please make sure you have a working camera connected.")
                        
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

    def run(self):
        self.master.mainloop()

if __name__ == '__main__':
    create = Dataset_Creation()
    create.run()
