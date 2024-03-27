# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from .preprocess import preprocess
from .predict import predict

from pydub import AudioSegment
from tempfile import NamedTemporaryFile


def mp3_to_wav(mp3_file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)

    # Export the audio to WAV format
    audio.export(wav_file, format="wav")


def home(request):
    if request.method == "POST" and "audio_file" in request.FILES:
        audio = request.FILES["audio_file"]
        # audio = request.FILES.get("audio_file")
        print(audio)
        if audio.content_type == "audio/mpeg":
            # Convert the MP3 file to WAV format
            temp_file = NamedTemporaryFile()
            mp3_to_wav(audio, temp_file)
            temp_file.seek(0)
            audio = temp_file
        # print(preprocess(audio))
        print(audio)
        preprocessed_audio = preprocess(audio)
        genre_result, confidence = predict(preprocessed_audio)
        # Convert confidence to a Python float
        confidence = confidence.item()

        return JsonResponse({"genre_result": genre_result, "confidence": confidence})
    context = {"page": "Home"}
    return render(request, "index.html", context)


def about(request):
    context = {"page": "About"}
    return render(request, "about.html", context)


def model(request):
    context = {"page": "Model"}
    return render(request, "model.html", context)
