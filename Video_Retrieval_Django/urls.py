from django.urls import path
from Video_Retrieval_Django import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import combined_clip, feedback_loop, send_result, search_lion, reset_scores, find_similar, search_histogram, show_surrounding

urlpatterns = [
    path('', views.home, name='home'),
    path('search_clip', search_lion, name='search_clip'),
    path("combined_clip", combined_clip, name="combined_clip"),
    path("send_result/", views.send_result, name="send_result"),
    path("feedback_loop", feedback_loop, name="feedback_loop"),
    path("reset_scores", reset_scores, name="reset_scores"),
    path("find_similar", find_similar, name="find_similar"),
    path("search_histogram", search_histogram, name="search_histogram"),
    path("show_surrounding", show_surrounding, name="show_surrounding")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

