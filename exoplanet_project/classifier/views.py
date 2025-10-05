from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from .models import ModelTraining, Prediction
from .ml_utils import train_model, predict_candidates
import os
from django.conf import settings

def dashboard(request):
    """Main dashboard showing model statistics"""
    latest_training = ModelTraining.objects.filter(status='completed').first()
    all_trainings = ModelTraining.objects.all()[:10]
    
    context = {
        'latest_training': latest_training,
        'all_trainings': all_trainings,
        'total_trainings': ModelTraining.objects.count(),
    }
    
    return render(request, 'classifier/dashboard.html', context)


def train_new_model(request):
    """Handle model training with hyperparameters"""
    if request.method == 'POST':
        # Get uploaded file
        if 'dataset' not in request.FILES:
            messages.error(request, 'No dataset file uploaded')
            return redirect('dashboard')
        
        dataset_file = request.FILES['dataset']
        
        # Get hyperparameters from form
        rf_estimators = int(request.POST.get('rf_estimators', 200))
        gb_estimators = int(request.POST.get('gb_estimators', 200))
        gb_lr = float(request.POST.get('gb_learning_rate', 0.1))
        test_size = float(request.POST.get('test_size', 0.3))
        
        # Save file
        media_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, dataset_file.name)
        
        with open(file_path, 'wb+') as destination:
            for chunk in dataset_file.chunks():
                destination.write(chunk)
        
        # Create training record
        training = ModelTraining.objects.create(
            dataset_file=dataset_file.name,
            n_samples=0,  # Will be updated
            n_features=0,  # Will be updated
            rf_n_estimators=rf_estimators,
            gb_n_estimators=gb_estimators,
            gb_learning_rate=gb_lr,
            test_size=test_size,
            status='training'
        )
        
        # Train model
        try:
            result = train_model(file_path, training)
            training.status = 'completed'
            training.n_samples = result['n_samples']
            training.n_features = result['n_features']
            training.accuracy = result['accuracy']
            training.f1_score = result['f1_score']
            training.roc_auc = result['roc_auc']
            training.balanced_accuracy = result['balanced_accuracy']
            training.training_time = result['training_time']
            training.save()
            
            messages.success(request, f'Model trained successfully! Accuracy: {result["accuracy"]:.2%}')
        except Exception as e:
            training.status = 'failed'
            training.save()
            messages.error(request, f'Training failed: {str(e)}')
        
        return redirect('dashboard')
    
    return render(request, 'classifier/train.html')


def predict_view(request):
    """Make predictions on candidate data"""
    if request.method == 'POST':
        if 'candidates_file' not in request.FILES:
            messages.error(request, 'No candidates file uploaded')
            return redirect('predict')
        
        candidates_file = request.FILES['candidates_file']
        
        # Save file
        media_dir = os.path.join(settings.MEDIA_ROOT, 'candidates')
        os.makedirs(media_dir, exist_ok=True)
        file_path = os.path.join(media_dir, candidates_file.name)
        
        with open(file_path, 'wb+') as destination:
            for chunk in candidates_file.chunks():
                destination.write(chunk)
        
        # Get latest model
        latest_training = ModelTraining.objects.filter(status='completed').first()
        if not latest_training:
            messages.error(request, 'No trained model available. Train a model first.')
            return redirect('train')
        
        try:
            predictions = predict_candidates(file_path, latest_training)
            
            # Save predictions to database
            for pred in predictions:
                Prediction.objects.create(
                    model_training=latest_training,
                    koi_id=pred['koi_id'],
                    predicted_class=pred['class'],
                    confidence=pred['confidence']
                )
            
            messages.success(request, f'Predictions complete! {len(predictions)} candidates analyzed.')
            return redirect('predictions_list')
            
        except Exception as e:
            messages.error(request, f'Prediction failed: {str(e)}')
            return redirect('predict')
    
    return render(request, 'classifier/predict.html')


def predictions_list(request):
    """Show all predictions"""
    latest_training = ModelTraining.objects.filter(status='completed').first()
    
    if latest_training:
        predictions = Prediction.objects.filter(model_training=latest_training)[:100]
    else:
        predictions = []
    
    context = {
        'predictions': predictions,
        'latest_training': latest_training,
    }
    
    return render(request, 'classifier/predictions.html', context)


def model_comparison(request):
    """Compare different model trainings"""
    trainings = ModelTraining.objects.filter(status='completed')
    
    context = {
        'trainings': trainings,
    }
    
    return render(request, 'classifier/comparison.html', context)