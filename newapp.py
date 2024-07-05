import streamlit as st
from IDP_MODEL import pipeline  # Import your speaker verification pipeline function
import sounddevice as sd
import soundfile as sf
import os

def save_audio(filename, duration, sample_rate, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float64")
    sd.wait()
    sf.write(os.path.join(folder_path, filename), recording, sample_rate)

def get_latest_file(test_folder_path, user_name):
    latest_file_number = 0
    latest_file = None

    for filename in os.listdir(test_folder_path):
        if filename.startswith(f"{user_name}_test_audio_"):
            number_str=filename.split("_")[-1].split(".")[0]
            number=int(number_str)
            if number>latest_file_number:
                latest_file_number=number
                latest_file=filename
    return latest_file

def main():
    st.title("Speaker Verification App")
    st.markdown("---")

    sample_rate = 16000  # Set sample rate
    duration = 10  # Set recording duration in seconds

    user_type = st.radio("Select User Type:", ("New User", "Existing User"))

    folder_path="C:\\Users\\HP\\Downloads\\IDP Project\\data\\train"

    if user_type == "New User":
        train_folder_path = "C:\\Users\\HP\\Downloads\\IDP Project\\data\\train"  # Folder path to save train audio
        user_name = st.text_input("Enter Your Name", "")  # Ask for user's name

        if st.button("Record Train Audio"):
            st.write("Recording Train Audio...")
            train_filename = f"{user_name}_train_audio.wav" if user_name else "train_audio.wav"
            save_audio(train_filename, duration, sample_rate, train_folder_path)
            st.write(f"Train Audio saved as '{train_filename}' in 'train' folder")

        test_folder_path = "C:\\Users\\HP\\Downloads\\IDP Project\\data\\test"   # Folder path to save test audio
        test_filename = f"{user_name}_test_audio.wav" if user_name else "test_audio.wav"

        # If test file already exists, increment the number until a unique filename is found
        if os.path.exists(os.path.join(test_folder_path, test_filename)):
            file_number = 1
            while True:
                new_test_filename = f"{user_name}_test_audio_{file_number}.wav" if user_name else f"test_audio_{file_number}.wav"
                if not os.path.exists(os.path.join(test_folder_path, new_test_filename)):
                    test_filename = new_test_filename
                    break
                file_number += 1
                

        if st.button("Record Test Audio"):
            st.write("Recording Test Audio...")
            save_audio(test_filename, duration, sample_rate, test_folder_path)
            st.write(f"Test Audio saved as '{test_filename}' in 'test' folder")
        

        new_file=get_latest_file(test_folder_path,user_name)
        print(new_file)
        if st.button("Train and Test Model"):
            st.write("Training and testing the model...")
            st.write(pipeline(os.path.join(test_folder_path, new_file), folder_path))


    elif user_type == "Existing User":
        option = st.selectbox("Select Option:", ("Record Test Audio", "Select Audio from Test Folder"))

        if option == "Record Test Audio":
            user_name = st.text_input("Enter Your Name", "")
            test_folder_path = "C:\\Users\\HP\\Downloads\\IDP Project\\data\\test"   # Folder path to save test audio
            test_filename = f"{user_name}_test_audio.wav" if user_name else "test_audio.wav"

            # If test file already exists, increment the number until a unique filename is found
            if os.path.exists(os.path.join(test_folder_path, test_filename)):
                file_number = 1
                while True:
                    new_test_filename = f"{user_name}_test_audio_{file_number}.wav" if user_name else f"test_audio_{file_number}.wav"
                    if not os.path.exists(os.path.join(test_folder_path, new_test_filename)):
                        test_filename = new_test_filename
                        break
                    file_number += 1
                    

            if st.button("Record Test Audio"):
                st.write("Recording Test Audio...")
                save_audio(test_filename, duration, sample_rate, test_folder_path)
                st.write(f"Test Audio saved as '{test_filename}' in 'test' folder")
            

            new_file=get_latest_file(test_folder_path,user_name)
            print(new_file)
            if st.button("Train and Test Model"):
                st.write("Training and testing the model...")
                st.write(pipeline(os.path.join(test_folder_path, new_file), folder_path))

        elif option == "Select Audio from Test Folder":
            # Logic to select audio from the test folder remains the same as provided in the previous code snippet
            test_files = os.listdir("C:\\Users\\HP\\Downloads\\IDP Project\\data\\test")  # Update with your test folder path
            selected_file = st.selectbox("Select Test Audio:", test_files)
            selected_file_path = os.path.join("C:\\Users\\HP\\Downloads\\IDP Project\\data\\test", selected_file)  # Update with your test folder path
            st.write("Selected Test Audio:", selected_file)
            if st.button("Test Model"):
                st.write("Testing the model...")
                st.write(pipeline(selected_file_path,folder_path))
            

if __name__ == "__main__":
    main()
