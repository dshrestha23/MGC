# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from .preprocess import preprocess
from .predict import predict


def home(request):
    if request.method == "POST" and "audio_file" in request.FILES:
        audio = request.FILES["audio_file"]
        # audio = request.FILES.get("audio_file")
        print(audio)
        # print(preprocess(audio))
        preprocessed_audio = preprocess(audio)
        genre_result = predict(preprocessed_audio)

        return JsonResponse({"genre_result": genre_result})
    context = {"page": "Home"}
    return render(request, "index.html", context)


def about(request):
    context = {"page": "About"}
    return render(request, "about.html", context)


def model(request):
    context = {"page": "Model"}
    return render(request, "model.html", context)
