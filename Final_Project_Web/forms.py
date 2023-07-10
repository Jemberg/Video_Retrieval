from django import forms

class SearchForm(forms.Form):
    text = forms.CharField(max_length=200)
