# Import libs
import os
import pickle
from gingerit.gingerit import GingerIt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Stop unnecessary TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the class
class NewModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_len = None

    def train(self, in_sequences=None, out_sequences=None, epochs=100):
        # Tokenize the training data
        self.tokenizer = Tokenizer()
        if in_sequences and out_sequences:
            self.tokenizer.fit_on_texts(in_sequences + out_sequences)
            total_words = len(self.tokenizer.word_index) + 1
        else:
            self.tokenizer.fit_on_texts([training_data])
            total_words = len(self.tokenizer.word_index) + 1

        # Create input sequences
        input_sequences = []
        if in_sequences and out_sequences:
            training_data = in_sequences + out_sequences
        for sequence in training_data:
            tokenized_sequence = self.tokenizer.texts_to_sequences([sequence])[0]
            for i in range(1, len(tokenized_sequence)):
                n_gram_sequence = tokenized_sequence[:i+1]
                input_sequences.append(n_gram_sequence)

        # Pad sequences for consistent input shape
        self.max_sequence_len = max([len(seq) for seq in input_sequences])
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')

        # Create predictors and labels
        predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
        labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        # Define and compile the model
        self.model = Sequential()
        self.model.add(Embedding(total_words, 100, input_length=self.max_sequence_len-1))
        self.model.add(LSTM(150))
        self.model.add(Dense(total_words, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Train the model
        self.model.fit(predictors, labels, epochs=epochs, verbose=1)

    def generate(self, prompt, max_words=20, fix=False, fix_gram=False):
        seed_text = prompt.lower()
        generated_text = [seed_text]

        for _ in range(max_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            predicted = self.model.predict(token_list, verbose=0)
            predicted_word_index = tf.argmax(predicted, axis=-1).numpy()[0]

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break
            seed_text += " " + output_word
            if fix:
                if (output_word != generated_text[-1]):
                    generated_text.append(output_word)
                else:
                    break
            else:
              generated_text.append(output_word)

        def dupe(words):
            result = [words[0]]  # Start with the first word

            for word in words[1:]:
                if word != result[-1]:  # Check if the current word is different from the previous word
                    result.append(word)

            return " ".join(result)
        
        def gram(sentence):
            parser = GingerIt()
            result = parser.parse(sentence)
            corrected_text = result['result']+"."
            return corrected_text

        done = dupe(generated_text)
        done = dupe(done.split())
        if fix_gram:
            done = gram(done)
        return done
    
    def save(self):
        print("________________________________________")
        print("Saving model.h5")
        # Save the model weights
        save_path = os.path.join("saved_models", "model.h5")
        self.model.save_weights(save_path)
        print(f"    Weights saved to {save_path}")

        # Save the tokenizer
        tokenizer_path = os.path.join("saved_models", "tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"    Tokenizer saved to {tokenizer_path}")

        # Save the max_sequence_len
        max_sequence_len_path = os.path.join("saved_models", "max_sequence_len.pkl")
        with open(max_sequence_len_path, 'wb') as f:
            pickle.dump(self.max_sequence_len, f)
        print(f"    max_sequence_len saved to {max_sequence_len_path}")
        print(f"Model saved to {save_path}!")
        print("________________________________________")

    def load(self):
        print("________________________________________")
        print("Loading model.h5")
        # Check if the model file exists
        if not os.path.exists(os.path.join("saved_models", "model.h5")):
            print("No saved model found. Train the model or save a model first.")
            return

        # Create a new instance of NewModel
        new_model = NewModel()
        print(" Created NewModel instance")
                
        # Load the tokenizer
        tokenizer_path = os.path.join("saved_models", "tokenizer.pkl")
        with open(tokenizer_path, 'rb') as f:
            new_model.tokenizer = pickle.load(f)
        print(" Loaded tokenizer")

        # Load the max_sequence_len
        max_sequence_len_path = os.path.join("saved_models", "max_sequence_len.pkl")
        with open(max_sequence_len_path, 'rb') as f:
            new_model.max_sequence_len = pickle.load(f)
        print(f"    Loaded max_sequence_len: {new_model.max_sequence_len}")

        # Instantiate the model architecture
        total_words = len(new_model.tokenizer.word_index) + 1
        print(f"    Loaded {total_words} words")
        new_model.model = Sequential()
        new_model.model.add(Embedding(total_words, 100, input_length=new_model.max_sequence_len - 1))
        new_model.model.add(LSTM(150))
        new_model.model.add(Dense(total_words, activation='softmax'))
        print(" Instantiated model architecture")

        # Load the model weights
        save_path = os.path.join("saved_models", "model.h5")
        new_model.model.load_weights(save_path)
        print(" Loaded model weights")

        # Update the attributes of the current instance with the loaded model's attributes
        self.model = new_model.model
        self.tokenizer = new_model.tokenizer
        self.max_sequence_len = new_model.max_sequence_len

        print(f"Model loaded from {save_path}!")
        print("________________________________________")

    def reset(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_len = None