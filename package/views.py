from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from clasters import clasters
from package.models import File
import os

@csrf_exempt
def package(request):
	if request.method == 'GET':
		return render(request, 'package.html')	
	elif request.method == 'POST':
		file_list = []
		models_list = []
		files = []
		i = request.FILES['files']
		for i in request.FILES.getlist('files'):
			new_file = File(file=i)
			new_file.save()
			models_list.append(new_file)
			file_list.append('media/'+str(new_file.file))
			files.append(str(new_file.file))
		clusters = int(request.POST.get('clusters'))
		if clusters > len(request.FILES.getlist('files')):
			clusters = len(request.FILES.getlist('files'))
		print(clusters)
		res = clasters(file_list, clusters)
		for i in models_list:
			os.remove('media/'+str(i.file))
		return render(request, 'package_answer.html', {'src': '/' + res, 'files': files})

from django.shortcuts import render

# Create your views here.
