from django import forms
from package.models import File

class FIleForm(forms.ModelForm):

	class Meta:
		model = File
		fields = ('file',)
