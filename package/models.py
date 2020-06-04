from django.db import models
class File(models.Model):
	file = models.FileField(null=True, upload_to='')
	
	class Meta:
		verbose_name = "Прикрепление"
		verbose_name_plural = "Прикрепления"
		ordering = ["file"]
