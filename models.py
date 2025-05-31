from app import db
from datetime import datetime
import json

class SolarPlant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    capacity_mw = db.Column(db.Float, nullable=False)
    installation_date = db.Column(db.Date, nullable=False)
    panel_type = db.Column(db.String(50), nullable=False)
    efficiency_rating = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class WeatherData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('solar_plant.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    temperature = db.Column(db.Float, nullable=False)  # Celsius
    humidity = db.Column(db.Float, nullable=False)  # Percentage
    cloud_cover = db.Column(db.Float, nullable=False)  # Percentage
    wind_speed = db.Column(db.Float, nullable=False)  # km/h
    solar_irradiance = db.Column(db.Float, nullable=False)  # kWh/m²/day
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    plant = db.relationship('SolarPlant', backref=db.backref('weather_data', lazy=True))

class EnergyProduction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('solar_plant.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    energy_produced = db.Column(db.Float, nullable=False)  # kWh
    equipment_efficiency = db.Column(db.Float, nullable=False)  # Percentage
    revenue_inr = db.Column(db.Float, nullable=False)  # Indian Rupees
    tariff_rate = db.Column(db.Float, nullable=False)  # INR per kWh
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    plant = db.relationship('SolarPlant', backref=db.backref('energy_production', lazy=True))

class MaintenanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('solar_plant.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    maintenance_type = db.Column(db.String(100), nullable=False)
    cost_inr = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    efficiency_impact = db.Column(db.Float, default=0.0)  # Percentage improvement
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    plant = db.relationship('SolarPlant', backref=db.backref('maintenance_records', lazy=True))

class MLPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, db.ForeignKey('solar_plant.id'), nullable=False)
    prediction_date = db.Column(db.Date, nullable=False)
    predicted_energy = db.Column(db.Float, nullable=False)  # kWh
    predicted_revenue = db.Column(db.Float, nullable=False)  # INR
    predicted_efficiency = db.Column(db.Float, nullable=False)  # Percentage
    confidence_score = db.Column(db.Float, nullable=False)  # 0-1
    model_used = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    plant = db.relationship('SolarPlant', backref=db.backref('predictions', lazy=True))

class ModelPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    accuracy_score = db.Column(db.Float, nullable=False)
    rmse = db.Column(db.Float, nullable=False)
    mae = db.Column(db.Float, nullable=False)
    r2_score = db.Column(db.Float, nullable=False)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_size = db.Column(db.Integer, nullable=False)
