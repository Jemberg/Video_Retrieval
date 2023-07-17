from django.urls import path
from Final_Project_Web import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import search_clip, combined_clip, feedback_loop, send_result

urlpatterns = [
    path('', views.home, name='home'),
    path('search_clip', search_clip, name='search_clip'),
    path("combined_clip", combined_clip, name="combined_clip"),
    path("send_result", send_result, name="send_result"),
    path("feedback_loop", feedback_loop, name="feedback_loop"),
    # path('send_result/<str:image_name>/', views.send_result, name='send_result'),
    # path('find_similar/<str:image_id>/', views.find_similar, name='find_similar'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

