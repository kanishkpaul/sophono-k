from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import tempfile
import os
from .musicai import analyze_song, generate_final_analysis, extract_final_score, sigmoid_scale
from .lyricsai import analyze_lyrics


from django.shortcuts import render

def home_view(request):
    return render(request, 'home.html')


@csrf_exempt
def process_music(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        
        # Save temporary file
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        try:
            # Process the audio
            results = analyze_song(temp_path)
            analysis_text = generate_final_analysis(results)
            final_score = extract_final_score(analysis_text)
            potential_streams = sigmoid_scale(final_score)
            
            return JsonResponse({
                'status': 'success',
                'result': analysis_text,
                'score': final_score,
                'streams': potential_streams
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            os.remove(temp_path)
    
    return JsonResponse({'status': 'error', 'message': 'No file uploaded'})

@csrf_exempt
def process_lyrics(request):
    if request.method == 'POST' and request.POST.get('lyrics'):
        lyrics = request.POST['lyrics']
        try:
            analysis = analyze_lyrics(lyrics)
            return JsonResponse({
                'status': 'success',
                'result': analysis.get('analysis', '')
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'No lyrics provided'})