#!/usr/bin/env python

import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import shutil
import time

import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import filedialog, messagebox

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

class Train_Model:
    def __init__(self, data_path=None, epochs=None, patience=None, model_name=None):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence_length = 30
        self.data_path = data_path 
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        
    def get_folder_names(self, directory):
        folder_names = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        return folder_names
    
    def temporary_directory(self, source_dir, temp_dir):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        shutil.copytree(source_dir, temp_dir)
    
    
    def data_split(self, temp_dir, folders, label_map):
        train_label_list, val_label_list = [], []
        for folder in os.listdir(temp_dir):
            source_folder = os.path.join(temp_dir, folder)
            train_folder = os.path.join(source_folder, "train")
            val_folder = os.path.join(source_folder, "val")


            # Create the train and val folders if they don't exist
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)

            subfolder_paths = []
            labels = []

             # Iterate through each subfolder in the source_folder
            for subfolder in os.listdir(source_folder):
                subfolder_path = os.path.join(source_folder, subfolder)

                if os.path.isdir(subfolder_path) and (subfolder != 'train' and subfolder != 'val'):
                    subfolder_paths.append(subfolder_path)
                    labels.append(label_map[folder])
            if any(subfolder_paths):
                # Split the subfolders into train and validation sets
                train_subfolder_paths, val_subfolder_paths, train_labels, val_labels = train_test_split(
                    subfolder_paths, labels, test_size=0.3, random_state=None
                )

                train_label_list.append(train_labels)
                val_label_list.append(val_labels)

                # Move the train subfolders to the train_folder
                for subfolder_path in train_subfolder_paths:
                    dest_path = os.path.join(train_folder)
                    shutil.move(subfolder_path, dest_path)

                # # Move the validation subfolders to the val_folder
                for subfolder_path in val_subfolder_paths:
                    dest_path = os.path.join(val_folder)
                    shutil.move(subfolder_path, dest_path)

        train_sequences, train_labels = [], []
        for folder in folders:
            train_path = os.path.join(temp_dir, folder, 'train')
            for sequence in os.listdir(train_path):
                window = [np.load(os.path.join(temp_dir, folder, 'train', str(sequence), "{}.npy".format(frame_num))) for frame_num in range(self.sequence_length)]
                train_sequences.append(window)
                train_labels.append(label_map[folder])

        val_sequences, val_labels = [], []
        for folder in folders:
            val_path = os.path.join(temp_dir, folder, 'val')
            for sequence in os.listdir(val_path):
                window = [np.load(os.path.join(temp_dir, folder, 'val', str(sequence), "{}.npy".format(frame_num))) for frame_num in range(self.sequence_length)]
                val_sequences.append(window)
                val_labels.append(label_map[folder])

        X_train = np.array(train_sequences)
        y_train = to_categorical(train_labels).astype(int)
        X_val = np.array(val_sequences)
        y_val = to_categorical(val_labels).astype(int)
        shutil.rmtree(temp_dir)

        return X_train, y_train, X_val, y_val
    
    def model_train(self, X_train, y_train, X_val, y_val, folders, epochs, patience):
        model = load_model('../models/model.h5')
        model = Sequential(model.layers[:-1])
        
        new_output_layer = Dense(folders.shape[0], activation='softmax', name='new_dense_layer')
        model.add(new_output_layer)
        
        model.summary()

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
        
        return model, history

    def model_plots(self, history, target_names, model, X_val, y_val, save_runs):
        train_loss=history.history['loss']
        val_loss = history.history['val_loss']
        runs_count = self.count_directories(save_runs)
        file_name = os.path.join(save_runs, 'run'+str(runs_count+1))
        os.makedirs(file_name)
        # Plotting training and validation loss
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(file_name, 'training_validation_loss.png'))
        plt.show()


        train_accuracy=history.history['categorical_accuracy']
        val_accuracy = history.history['val_categorical_accuracy']

        # Plotting training and validation loss
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.title('Accuracies vs epochs')
        plt.legend()
        plt.savefig(os.path.join(file_name, 'training_validation_accuracies.png'))
        plt.show()

        pred = model.predict(X_val)
        ytrue = np.argmax(y_val, axis=1).tolist()
        ypred = np.argmax(pred, axis=1).tolist()
        print("*************************************************************************")
        print("ACCURACY: ",accuracy_score(ytrue, ypred))
        print("*************************************************************************")
        report = classification_report(ytrue, ypred, target_names=target_names)
        print("CLASSIFICATION REPORT")
        print("*************************************************************************")
        print(report)
        print("*************************************************************************")
        
        f1 = f1_score(ytrue, ypred, average='weighted')
        recall = recall_score(ytrue, ypred, average='weighted')
        
        print("F1-score:", f1)
        print("Recall:", recall)
        print("*************************************************************************")
        
        with open(os.path.join(file_name, 'metrics_and_report.txt'), 'w') as metrics_report_file:
            metrics_report_file.write(f'Accuracy Score: {accuracy_score(ytrue, ypred)}\n')
            metrics_report_file.write(f'F1 Score: {f1}\n')
            metrics_report_file.write(f'Recall: {recall}\n\n')
            metrics_report_file.write('Classification Report:\n')
            metrics_report_file.write(report)

    def count_directories(self, folder_path):
        count = 0
        for entry in os.scandir(folder_path):
            count += 1
        return count 

    def train(self):
        DATA_PATH = self.data_path
        save_runs = os.path.join('../runs')
        if os.path.exists(save_runs) == False:
            os.makedirs(save_runs)
            
        if os.path.exists(DATA_PATH) == False:
            print(f"NO {DATA_PATH} FOLDER FOUND")
            exit()
        print("Data Processing ...")

        try:
            folders = np.array(self.get_folder_names(DATA_PATH))
            label_map = {label:num for num, label in enumerate(folders)}
            temp_dir = './tempdata'
            target_names = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
            self.temporary_directory(DATA_PATH, temp_dir)
            X_train, y_train, X_val, y_val = self.data_split(temp_dir, folders, label_map)
            print("Data Processing done. Created temporary directory. Training in progress ...")
            start_time = time.time()
            model, history = self.model_train(X_train, y_train, X_val, y_val, folders, self.epochs, self.patience)
            elapsed_time = (time.time() - start_time)/3600
            self.model_plots(history, target_names, model, X_val, y_val, save_runs)
            print(f"Training time: {elapsed_time} hrs")
            print("*************************************************************************")
            run_count = self.count_directories(save_runs)
            run_path = os.path.join(save_runs, 'run'+str(run_count))
            model.save(os.path.join(run_path, self.model_name))
            print(f"Run information and model saved at {run_path} folder")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def run(self):
        self.train()

class TrainModelGUI:
    def __init__(self):
        self.master = tk.Tk()
        self.master.title("Train Model")
        self.master.geometry("400x400")

        self.label_data = tk.Label(self.master, text="Data Folder:")
        self.label_data.pack(pady=(10, 0))

        self.entry_data = tk.Entry(self.master, width=30)
        self.entry_data.pack(pady=(0, 10))

        self.btn_browse_data = tk.Button(self.master, text="Choose data folder", command=self.browse_data)
        self.btn_browse_data.pack(pady=(10, 0))

        self.label_epochs = tk.Label(self.master, text="Number of Epochs:")
        self.label_epochs.pack(pady=(10, 0))

        self.entry_epochs = tk.Entry(self.master, width=10)
        self.entry_epochs.pack(pady=(0, 10))

        self.label_patience = tk.Label(self.master, text="Patience:")
        self.label_patience.pack(pady=(10, 0))

        self.entry_patience = tk.Entry(self.master, width=10)
        self.entry_patience.pack(pady=(0, 10))

        self.label_model = tk.Label(self.master, text="Model Name:")
        self.label_model.pack(pady=(10, 0))

        self.entry_model = tk.Entry(self.master, width=20)
        self.entry_model.pack(pady=(0, 10))

        self.btn_train_model = tk.Button(self.master, text="Train Model", command=self.train_model)
        self.btn_train_model.pack(pady=(20, 0))

        self.btn_exit = tk.Button(self.master, text="Exit", command=self.master.quit)
        self.btn_exit.pack(pady=(20, 0))
        
        self.set_default_values()
        
    def set_default_values(self):
        # Set default paths here
        self.entry_data.insert(0, '../data')
        self.entry_model.insert(0, 'exp.h5')
        self.entry_patience.insert(0, '10')
        self.entry_epochs.insert(0, '600')

    def browse_data(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_data.delete(0, tk.END)
            self.entry_data.insert(0, folder)

    def train_model(self):
        data_path = self.entry_data.get()
        epochs = int(self.entry_epochs.get())
        patience = int(self.entry_patience.get())
        model_name = self.entry_model.get()
        self.master.destroy()
        
        if data_path and epochs and patience and model_name:
            _, ext = os.path.splitext(model_name)
            valid_extensions = ['.h5', '.keras', '.model']

            if ext not in valid_extensions:
                model_name += '.h5'
            
            train = Train_Model(data_path=data_path, epochs=epochs, patience=patience, model_name=model_name)
            train.run()
            messagebox.showinfo("Training Complete", "Model has been trained successfully!")
        else:
            messagebox.showwarning("Missing Information", "Please fill in all fields.")

    def run(self):
        self.master.mainloop()

def main():
    gui = TrainModelGUI()
    gui.run()

if __name__ == '__main__':
    main()
