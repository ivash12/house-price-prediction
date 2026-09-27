from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def viewF(request):
     return HttpResponse("hey")