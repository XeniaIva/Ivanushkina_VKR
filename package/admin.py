from django.contrib import admin
from package.models import File

class FileAdmin(admin.ModelAdmin):
	  list_display = ('file',)

# Register your models here.

admin.site.register(File, FileAdmin)
# Register your models here.
