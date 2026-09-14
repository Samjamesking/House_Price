from django.db import models
from django.utils import timezone

class Prediction(models.Model):
    city = models.CharField(max_length=100, default='Mumbai')
    location = models.CharField(max_length=255, default='Unknown')
    model_used = models.CharField(max_length=50, default='XGBoost')
    sqft = models.IntegerField()
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    floors = models.IntegerField(default=1)
    year_built = models.IntegerField(null=True, blank=True)
    vintage = models.IntegerField(default=2)  # Property Vintage in years
    parking = models.BooleanField(default=False)
    gated_security = models.BooleanField(default=False)
    gymnasium = models.BooleanField(default=False)
    swimming_pool = models.BooleanField(default=False)
    elevator = models.BooleanField(default=False)
    furnishing = models.CharField(max_length=50, default='Unfurnished')
    
    # Location coordinates for accurate mapping
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    
    predicted_price = models.DecimalField(max_digits=14, decimal_places=2)
    alt_model_price = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Prediction ({self.model_used}): ₹{self.predicted_price} ({self.city} - {self.location}, {self.sqft} sqft, {self.bedrooms} BHK)"
