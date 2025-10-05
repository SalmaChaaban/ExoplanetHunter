from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import ModelTraining, Prediction

@admin.register(ModelTraining)
class ModelTrainingAdmin(admin.ModelAdmin):
    list_display = ['id', 'timestamp', 'status', 'accuracy', 'roc_auc', 'training_time']
    list_filter = ['status', 'timestamp']
    readonly_fields = ['timestamp']

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['koi_id', 'predicted_class', 'confidence', 'timestamp']
    list_filter = ['predicted_class', 'timestamp']
    search_fields = ['koi_id']