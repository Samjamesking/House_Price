import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Prediction
from .ml.service import MLPredictorService

def login_view(request):
    if request.method == 'POST':
        return redirect('dashboard')
    return render(request, 'predictor/login.html')

def register_view(request):
    if request.method == 'POST':
        return redirect('login')
    return render(request, 'predictor/register.html')

def dashboard_view(request):
    predictions = Prediction.objects.all().order_by('-created_at')
    total_predictions = predictions.count()
    
    if total_predictions > 0:
        avg_price = sum(float(p.predicted_price) for p in predictions) / total_predictions
    else:
        avg_price = 0
        
    service = MLPredictorService.get_instance()
    meta = service.metadata or {}
    metrics = meta.get('metrics', {})

    GOOGLE_MAPS_KEY = 'AIzaSyCH3MOqllfQCKw2pXW54idck2kl7wYCp2s'

    context = {
        'total_predictions': total_predictions,
        'avg_price': avg_price,
        'predictions': predictions[:10],
        'rf_metrics': metrics.get('random_forest', {}),
        'xgb_metrics': metrics.get('xgboost', {}),
        'total_dataset_samples': meta.get('total_samples', 27911),
        'google_maps_api_key': GOOGLE_MAPS_KEY
    }
    return render(request, 'predictor/dashboard.html', context)

def result_view(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id)
    service = MLPredictorService.get_instance()
    
    # Run service to obtain model comparison, actual comparables, and real trend curve
    ml_result = service.predict(
        city_input=prediction.city,
        sqft=prediction.sqft,
        bedrooms=prediction.bedrooms,
        bathrooms=prediction.bathrooms,
        floors=prediction.floors,
        year_built=prediction.year_built,
        vintage=prediction.vintage,
        parking=prediction.parking,
        gated_security=prediction.gated_security,
        gymnasium=prediction.gymnasium,
        swimming_pool=prediction.swimming_pool,
        elevator=prediction.elevator,
        furnishing=prediction.furnishing,
        model_choice=prediction.model_used,
        location_name=prediction.location,
        latitude=prediction.latitude,
        longitude=prediction.longitude
    )
    
    margin = (100 - prediction.confidence_score) / 100 * float(prediction.predicted_price)
    min_price = max(100000, float(prediction.predicted_price) - margin)
    max_price = float(prediction.predicted_price) + margin

    context = {
        'prediction': prediction,
        'min_price': min_price,
        'max_price': max_price,
        'alt_model_name': ml_result['alt_model_name'],
        'alt_predicted_price': ml_result['alt_predicted_price'],
        'model_diff_percent': ml_result['model_diff_percent'],
        'model_metrics': ml_result['model_metrics'],
        'similar_houses': ml_result['similar_houses'],
        'similar_houses_json': json.dumps(ml_result['similar_houses']),
        'trend_data': json.dumps(ml_result['trend_data']),
        'target_lat': ml_result['target_lat'],
        'target_lng': ml_result['target_lng'],
        'amenity_premium_percent': ml_result.get('amenity_premium_percent', 0.0),
        'google_maps_api_key': 'AIzaSyCH3MOqllfQCKw2pXW54idck2kl7wYCp2s'
    }
    return render(request, 'predictor/result.html', context)

@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            city = data.get('city', 'Mumbai')
            location = data.get('location', '').strip() or f'{city} City Center'
            sqft = float(data.get('sqft', 1000))
            bedrooms = int(data.get('bedrooms', 2))
            bathrooms = int(data.get('bathrooms', bedrooms))
            floors = int(data.get('floors', 1))
            vintage = data.get('vintage')
            vintage = int(vintage) if vintage is not None and str(vintage).strip() != '' else 2
            year_built = data.get('year_built')
            year_built = int(year_built) if year_built else (2026 - vintage)
            
            parking = bool(data.get('parking', False))
            gated_security = bool(data.get('gated_security', False))
            gymnasium = bool(data.get('gymnasium', False))
            swimming_pool = bool(data.get('swimming_pool', False))
            elevator = bool(data.get('elevator', False))
            
            furnishing = data.get('furnishing', 'Unfurnished')
            model_choice = data.get('model_choice', 'xgboost')
            
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            try:
                lat_val = float(latitude) if latitude is not None and str(latitude).strip() != '' else None
            except Exception:
                lat_val = None
            try:
                lng_val = float(longitude) if longitude is not None and str(longitude).strip() != '' else None
            except Exception:
                lng_val = None
            
            service = MLPredictorService.get_instance()
            ml_res = service.predict(
                city_input=city,
                sqft=sqft,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                floors=floors,
                year_built=year_built,
                vintage=vintage,
                parking=parking,
                gated_security=gated_security,
                gymnasium=gymnasium,
                swimming_pool=swimming_pool,
                elevator=elevator,
                furnishing=furnishing,
                model_choice=model_choice,
                location_name=location,
                latitude=lat_val,
                longitude=lng_val
            )
            
            # Save to database
            prediction = Prediction.objects.create(
                city=ml_res['city'],
                location=location,
                model_used=ml_res['model_used'],
                sqft=int(sqft),
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                floors=floors,
                year_built=year_built,
                vintage=vintage,
                parking=parking,
                gated_security=gated_security,
                gymnasium=gymnasium,
                swimming_pool=swimming_pool,
                elevator=elevator,
                furnishing=furnishing,
                latitude=ml_res['target_lat'],
                longitude=ml_res['target_lng'],
                predicted_price=ml_res['predicted_price'],
                alt_model_price=ml_res['alt_predicted_price'],
                confidence_score=ml_res['confidence_score']
            )
            
            return JsonResponse({
                'success': True,
                'prediction_id': prediction.id,
                'predicted_price': ml_res['predicted_price'],
                'alt_model_name': ml_res['alt_model_name'],
                'alt_predicted_price': ml_res['alt_predicted_price'],
                'confidence_score': ml_res['confidence_score'],
                'model_used': ml_res['model_used'],
                'city': ml_res['city']
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def delete_prediction(request, prediction_id):
    if request.method in ['POST', 'DELETE']:
        try:
            prediction = get_object_or_404(Prediction, id=prediction_id)
            prediction.delete()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid method'})
