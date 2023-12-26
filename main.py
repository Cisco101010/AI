import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Paso 3: Preprocesamiento de datos
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents_file = "intents.json"

with open(intents_file) as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training_data = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append([bag, output_row])

# Ajuste de longitudes
max_len = len(max(training_data, key=lambda x: len(x[0]))[0])
training_data = [[pad_sequence(seq, max_len) for seq in example] for example in training_data]

np.random.shuffle(training_data)
training_data = np.array(training_data)

train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

# Paso 4: Construir el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Utiliza learning_rate en lugar de lr y elimina el argumento decay
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Paso 5: Entrenar el modelo
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Paso 6: Guardar el modelo
model.save("chatbot_model.h5")
