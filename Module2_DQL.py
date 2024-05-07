import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from sklearn.linear_model import LinearRegression
import featureClassification as fc

def findPRFC(predicted, actual, display=True) :
    f1 = f1_score(predicted, actual, average="macro")
    pre = precision_score(predicted, actual, average="macro")
    acc = accuracy_score(predicted, actual)
    rec = recall_score(predicted, actual, average="macro")
    conf_matrix = confusion_matrix(predicted, actual);
    diff_arr = np.array(predicted) - np.array(actual)
    idx = np.where(diff_arr == 0)[0]
    
    if(display) :
        print("Prediction f1 score : %s "% f1)
        print("Prediction Precision score : %s "% pre)
        print("Prediction accuracy score : %s "% acc)
        print("Prediction Recall score : %s "% rec)
        print('Confusion matrix')
        print(conf_matrix)
    return idx

class DDPGClassifier:
    def __init__(self, input_shape, output_shape):
        DR = 0.1
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DR))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DR))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DR))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(DR))
        self.model.add(Dense(output_shape, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    
    def train(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        return history
    
    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        return y_pred, accuracy, precision, recall, f1, confusion

class PPO_TD3Classifier:
    def __init__(self, input_shape, output_shape, learning_rate=0.1, discount_factor=0.99, num_bins=10):
        self.model = LinearRegression()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_bins = num_bins
        self.q_table = np.zeros((num_bins, output_shape))
    
    def discretize_state(self, state):
        return np.digitize(state, bins=np.linspace(0, 1, self.num_bins))
    
    def choose_action(self, state):
        try :
            q_values = self.q_table[state]
        except :
            q_values = self.q_table[0]
            
        action = np.argmax(q_values)
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        try :
            current_q = self.q_table[state, action]
        except :
            try :
                current_q = self.q_table[0, action]
            except :
                current_q = 0
        
        try :
            max_next_q = np.max(self.q_table[next_state])
        except :
            max_next_q = np.max(self.q_table[1])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        try :
            self.q_table[state, action] = new_q
        except :
            try :
                self.q_table[0, action] = new_q
            except :
                x = 0
    
    def train(self, X_train, y_train, num_episodes=10):
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.discretize_state(X_train[np.random.choice(X_train.shape[0])])  # Sample a random state
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state = self.discretize_state(X_train[np.random.choice(X_train.shape[0])])  # Sample the next state
                reward = int(y_train[np.random.choice(len(y_train))] == action)
                
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                
                if np.random.rand() > 0.95:  # 5% chance of ending an episode
                    done = True
            
            episode_rewards.append(total_reward)
        
        return episode_rewards
    
    def predict(self, X_test):
        predictions = []
        for state in X_test:
            state = self.discretize_state(state)
            action = self.choose_action(state)
            predictions.append(action)
        return predictions
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        return y_pred, accuracy, precision, recall, f1, confusion

t1 = time.time()

import easygui
import pandas as pd
import numpy as np

# Use easygui to ask for the file to use
file_path = easygui.fileopenbox(msg="Select CSV File", title="Select CSV File", default="*.csv")

# Load the CSV dataset
dataset = pd.read_csv(file_path).fillna(0)

# Get the list of column names
column_names = list(dataset.columns)

# Use easygui to ask which column is the class
class_column = easygui.choicebox(msg="Select the class column", title="Select Class Column", choices=column_names)

# Get the index of the class column
class_idx = column_names.index(class_column)

# Prepare X and y
y = np.array(dataset[class_column].values).astype(np.float)
X = np.array(dataset.drop(columns=[class_column]).values)

print("X shape:", X.shape)
print("y shape:", y.shape)

t1 = time.time()
# Convert categorical labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
ts = float(input('Enter Test Size(0 to 1):'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

# Initialize and train the DDPG classifier
print('Running DPPG...')
DDPG_classifier = DDPGClassifier(input_shape=(X_train.shape[1],), output_shape=len(label_encoder.classes_))
history = DDPG_classifier.train(X_train, y_train)
# Evaluate the classifier on the testing set
y_pred1, accuracy, precision, recall, f1, confusion = DDPG_classifier.evaluate(X_test, y_test)
# Output the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', confusion)

input_shape = X_train.shape[1]
output_shape = len(np.unique(y_train))
print('Running PPO with TD3...')
ql_classifier = PPO_TD3Classifier(input_shape, output_shape)
episode_rewards = ql_classifier.train(X_train, y_train)
# Evaluate the classifier on the testing set
y_pred2, accuracy, precision, recall, f1, confusion = ql_classifier.evaluate(X_test, y_test)
# Output the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', confusion)

print('Running SAC...')
idx1 = findPRFC(y_pred1, y_test)
idx2 = findPRFC(y_pred2, y_test)
idx3 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 1, True)
idx4 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 3, True)
final_array = np.union1d(idx1, idx2)
final_array = np.union1d(final_array, idx3)
final_array = np.union1d(final_array, idx4)
final_array = np.unique(final_array)
y_final = [0] * len(y_test)
for count in range(0, len(final_array)) :
    y_final[final_array[count]] = y_test[final_array[count]]

print('Final Prediction Results')
t2 = time.time()
findPRFC(y_test, y_final)
delay = t2 - t1
print('Delay needed %0.04f ms' % (delay))