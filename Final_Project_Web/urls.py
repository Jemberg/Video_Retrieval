from django.urls import path
from Final_Project_Web import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import combined_clip, feedback_loop, send_result, search_lion, reset_scores, find_similar, find_similar_histogram

urlpatterns = [
    path('', views.home, name='home'),
    path('search_clip', search_lion, name='search_clip'),
    path("combined_clip", combined_clip, name="combined_clip"),
    path("send_result/", send_result, name="send_result"),
    path("feedback_loop", feedback_loop, name="feedback_loop"),
    path("reset_scores", reset_scores, name="reset_scores"),
    path("find_similar", find_similar, name="find_similar"),
    path("find_similar_histogram", find_similar_histogram, name="find_similar_histogram")
    # path('send_result/<str:image_name>/', views.send_result, name='send_result'),
    # path('find_similar/<str:image_id>/', views.find_similar, name='find_similar'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

