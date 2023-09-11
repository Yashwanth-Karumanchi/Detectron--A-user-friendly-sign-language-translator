import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import shutil
import time

import warnings
warnings.filterwarnings('ignore')
import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LeakyReLU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, f1_score, recall_score, classification_report


class Train_Model:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence_length = 30 
        
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
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 126)),
            BatchNormalization(),  # Add batch normalization
            LeakyReLU(alpha=0.1),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dense(64),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(folders.shape[0], activation='softmax')
        ])
        
        model.summary()

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_val, y_val), callbacks=[early_stopping])
        
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

    
    def get_arguments(self):
        parser = argparse.ArgumentParser(description="Train a model.")
        parser.add_argument("--data", type=str, default='data', help="Relative location of the data folder")
        parser.add_argument("--epochs", type=int, default=600, help="Number of training epochs")
        parser.add_argument("--patience", type=int, default=100, help="Patience for early stopping")
        parser.add_argument("--model", type=str, default='exp.h5', help="Name for the generated model file")

        args = parser.parse_args()
        return args


def main():
    train = Train_Model()
    args = train.get_arguments()
    DATA_PATH = os.path.join(args.data)
    save_runs = os.path.join('./runs')
    if os.path.exists(save_runs) == False:
        os.makedirs(save_runs)
        
    if os.path.exists(DATA_PATH) == False:
        print(f"NO {DATA_PATH} FOLDER FOUND")
        exit()
    print("Data Processing ...")

    try:
        folders = np.array(train.get_folder_names(DATA_PATH))
        label_map = {label:num for num, label in enumerate(folders)}
        temp_dir = './tempdata'
        target_names = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
        train.temporary_directory(DATA_PATH, temp_dir)
        X_train, y_train, X_val, y_val = train.data_split(temp_dir, folders, label_map)
        print("Data Processing done. Created temporary directory. Training in progress ...")
        start_time = time.time()
        model, history = train.model_train(X_train, y_train, X_val, y_val, folders, args.epochs, args.patience)
        elapsed_time = (time.time() - start_time)/3600
        train.model_plots(history, target_names, model, X_val, y_val, save_runs)
        print(f"Training time: {elapsed_time} hrs")
        print("*************************************************************************")
        run_count = train.count_directories(save_runs)
        run_path = os.path.join(save_runs, 'run'+str(run_count))
        model.save(os.path.join(run_path, args.model))
        print(f"Run information and model saved at {run_path} folder")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
if __name__ == '__main__':
    main()