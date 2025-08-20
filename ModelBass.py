from music21 import converter, instrument, note, chord, stream, midi, environment, note as m21_note, chord as m21_chord
import numpy as np
import os
import tensorflow as tf
from keras import layers
import fluidsynth
import subprocess

environment.UserSettings()['warnings'] = 0

def extract_bass(file_path):
   
    try:
        midi = converter.parse(file_path)
        parts = instrument.partitionByInstrument(midi)

        bass_notes = []

        if parts:  
            for part in parts.parts:
                inst = part.getInstrument()
                inst_name = inst.instrumentName if inst.instrumentName else "Unknown"

                
                if "Bass" in inst_name:
                    print(f"Found bass instrument in {os.path.basename(file_path)}: {inst_name}")
                    for element in part.recurse():
                        if isinstance(element, note.Note):
                            bass_notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            bass_notes.append('.'.join(str(n) for n in element.normalOrder))

        return bass_notes

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_dataset_for_bass(dataset_path, file_limit=None):
    
    bass_data = {}
    file_count = 0

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.mid') or file_name.endswith('.midi'):
            file_path = os.path.join(dataset_path, file_name)
            print(f"Processing: {file_name}")
            bass_notes = extract_bass(file_path)
            bass_data[file_name] = bass_notes
            file_count += 1

            if file_limit and file_count >= file_limit:
                break

    print(f"Processed {file_count} MIDI files.")
    return bass_data

def generator(latent_dim, output_dim):
    model = tf.keras.Sequential(
        [
            layers.Dense(1024, activation="relu", input_dim=latent_dim),
            layers.Reshape((timesteps, features)),
            layers.BatchNormalization(),
            layers.LSTM(128, activation="tanh",return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(64, activation="tanh"),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation="softmax")
        ]
    )
    print(f"Reshaping to: {(timesteps, features)}")
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_dim=input_dim),
        layers.Reshape((timesteps, features)),
        layers.Dropout(0.3),
        layers.LSTM(256, activation="tanh",return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False  
    model = tf.keras.Sequential([generator, discriminator])
    return model

def train_gan(generator, discriminator, gan, data, latent_dim, epochs=1000, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

def generate_music(generator, latent_dim, bass_notes, midi_file="bass_guitar_output.mid", audio_file="bass_guitar_output.wav", sequence_length=100, soundfont="path\to\your\SQ_008_Rock_Bass.sf2"):
  
    noise = np.random.normal(0, 1, (sequence_length, latent_dim))
    generated_notes = generator.predict(noise)
    note_indices = np.argmax(generated_notes, axis=1)
    note_vocab = sorted(set(bass_notes))  
    valid_note_vocab = [n for n in note_vocab if n.isalpha() or (n[0].isalpha() and n[1:].isdigit())]
    generated_sequence = [valid_note_vocab[idx] for idx in note_indices if idx < len(valid_note_vocab)]   
    output_stream = stream.Part()
    output_stream.append(instrument.Bass())  

    for note_chord in generated_sequence:
        try:
            if '.' in note_chord:  
                chord_notes = [int(n) for n in note_chord.split('.') if n.isdigit()]
                if chord_notes:  
                    new_chord = m21_chord.Chord(chord_notes)
                    output_stream.append(new_chord)
            else:  
                if note_chord[-1].isdigit():  
                    new_note = m21_note.Note(note_chord)
                    output_stream.append(new_note)
        except Exception as e:
            print(f"Skipping invalid note/chord '{note_chord}': {e}")

    midi_file_path = midi.translate.music21ObjectToMidiFile(output_stream)
    midi_file_path.open(midi_file, 'wb')
    midi_file_path.write()
    midi_file_path.close()

    print(f"Generated MIDI file saved as {midi_file}")

    try:
        subprocess.run(["fluidsynth", "-ni", soundfont, midi_file, "-F", audio_file], check=True)
        print(f"Generated bass guitar audio saved as {audio_file}")
    except FileNotFoundError:
        print("Error: Fluidsynth CLI not found")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio conversion: {e}")

dataset_path = "path_to_your_dataset"  
file_limit = 85  
bass_tracks = process_dataset_for_bass(dataset_path, file_limit=file_limit)

bass_notes = set()
for notes in bass_tracks.values():
    bass_notes.update(notes)  

if not bass_notes:
    raise ValueError("No bass notes found in the dataset. Ensure the dataset contains valid bass tracks.")

valid_bass_notes = [note for note in bass_notes if note.isalpha() or note[-1].isdigit()]
if not valid_bass_notes:
    raise ValueError("Bass notes extracted are invalid. Check the dataset for proper MIDI formatting.")

bass_notes = sorted(valid_bass_notes)

latent_dim = 1024
timesteps =16
features = latent_dim//timesteps
output_dim = len(bass_notes)  

generator_model = generator(latent_dim, output_dim)

generator_model.compile(optimizer="adam", loss="categorical_crossentropy")

generate_music(generator_model, latent_dim, bass_notes, midi_file="bass_guitar_output.mid", audio_file="bass_guitar_output.wav", sequence_length=500)

