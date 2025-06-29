import google.generativeai as genai
import google.generativeai.types as types
import os
from dotenv import load_dotenv
load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key_google)
model = genai.GenerativeModel("gemini-2.0-flash")


def Create_Youtube_Transcript(url):
    # video = genai.upload_file("./test-2.mp4")
    # prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video."

    response = model.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Transcribe the audio from this video and don't give timestamp"},
                    {
                        "file_data": {
                            "file_uri": f"{url}",
                        }
                    }
                ]
            }
        ]
    )

    return response.text