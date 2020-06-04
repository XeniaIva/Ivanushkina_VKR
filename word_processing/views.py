from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from get_key_words_lda import get_key_words_lda
from get_key_words_bert import get_key_words_bert
from package.models import File
from django.conf import settings
import os


@csrf_exempt
def word_page(request):
	if request.method == 'GET':
		return render(request, 'word_processing.html')	
	elif request.method == 'POST':
		text = request.POST.get('text')
		file = request.FILES.get('file')
		words = 1
		topics = 1
		if request.POST.get('words').isnumeric():
			words = request.POST.get('words')

		if request.POST.get('topics').isnumeric():
			topics = request.POST.get('topics')

		if (file):
			text = ''
			new_file = File(file=file)
			new_file.save()
			path = settings.BASE_DIR + '/media/' + str(new_file.file)
			with open(path) as file_handler:
				for line in file_handler:
				 	text += line
			new_file.delete()
			os.remove(path)
		print(text)
		keywords = []
		methods = ['LDA', 'LSA', 'LDA+BERT']
		for i in methods:
			if request.POST.get(i) =='LDA':
				text = request.POST.get('text')
				keywords.append(['LDA', get_key_words_lda(text, int(topics), int(words))])
			elif request.POST.get(i) =='LDA+BERT':
				keywords.append(['LDA+BERT', get_key_words_bert(text, int(topics), int(words))])
				# text = request.POST.get('text')
		return render(request, 'two_proc.html', {'keywords': keywords})



		return render(request, 'word_processing.html', {'data': data[:10], 'search_data': search_data})
	else:
		HttpResponseNotAllowed(['GET', 'POST'])


# Create your views here.
