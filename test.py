import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key_google)

# print(responce.text)

def Generate_transcript(file_path):
    model = genai.GenerativeModel("gemini-2.0-flash")

    audio = genai.upload_file(file_path)
    prompt = "Generate a transcript of the speech."
    responce = model.generate_content(
        contents=[prompt, audio],
    )
    return responce.text









# from pydub import AudioSegment

# AudioFile = AudioSegment.from_file("./test.mp3", format="mp3")
# duration = len(AudioFile) / 1000  # Convert milliseconds to seconds
# print(f"Audio duration: {duration} seconds")




# transcript = responce.text
# question = "which is the fastest running animal in the world?"

# qa_prompt = f"""Here is a transcript of an audio:
# \"\"\"{transcript}\"\"\"
# Now, answer this question: {question}
# """

# qa_responce = model.generate_content(
#     contents=[qa_prompt],
# )
# print(qa_responce.text)

def generate_summary(text):

    model = genai.GenerativeModel("gemini-2.0-flash")
    
    qa_prompt = f"""Here is a transcript of an audio:
    \"\"\"{text}\"\"\"
    Please generate a detailed and clear summary of the content
    """

    qa_responce = model.generate_content(
        contents=[qa_prompt],
    )
    return qa_responce.text
