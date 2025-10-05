from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

class ModelTraining(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    dataset_file = models.CharField(max_length=255)
    n_samples = models.IntegerField()
    n_features = models.IntegerField()
    
    # Hyperparameters
    rf_n_estimators = models.IntegerField(default=200)
    gb_n_estimators = models.IntegerField(default=200)
    gb_learning_rate = models.FloatField(default=0.1)
    test_size = models.FloatField(default=0.3)
    
    # Metrics
    accuracy = models.FloatField(null=True)
    f1_score = models.FloatField(null=True)
    roc_auc = models.FloatField(null=True)
    balanced_accuracy = models.FloatField(null=True)
    training_time = models.FloatField(null=True)
    
    status = models.CharField(max_length=20, default='pending')  # pending, training, completed, failed
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Training {self.id} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class Prediction(models.Model):
    model_training = models.ForeignKey(ModelTraining, on_delete=models.CASCADE, related_name='predictions')
    koi_id = models.CharField(max_length=100)
    predicted_class = models.CharField(max_length=20)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-confidence']
    
    def __str__(self):
        return f"{self.koi_id} - {self.predicted_class} ({self.confidence:.2%})"