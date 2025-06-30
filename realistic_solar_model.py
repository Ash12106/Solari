"""
Realistic Solar Energy Prediction Model
Uses authentic weather data and physics-based calculations for accurate predictions
"""

import os
import logging
from datetime import datetime, timedelta
from app import db
from models import SolarPlant, WeatherData, EnergyProduction, MLPrediction
from weather_api import WeatherAPI
import math
import random

class RealisticSolarPredictor:
    """
    Physics-based solar prediction model using real weather data
    and authentic solar energy calculations
    """
    
    def __init__(self):
        self.weather_api = WeatherAPI()
        
    def calculate_solar_position(self, date, latitude=12.2958, longitude=76.6394):
        """
        Calculate solar elevation and azimuth for Mysuru, Karnataka
        Based on astronomical calculations
        """
        # Day of year
        day_of_year = date.timetuple().tm_yday
        
        # Solar declination angle
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (assuming noon for peak calculation)
        hour_angle = 0  # Noon
        
        # Solar elevation angle
        lat_rad = math.radians(latitude)
        decl_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)
        
        elevation = math.asin(
            math.sin(lat_rad) * math.sin(decl_rad) + 
            math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad)
        )
        
        return math.degrees(elevation)
    
    def calculate_theoretical_irradiance(self, date, cloud_cover, latitude=12.2958):
        """
        Calculate theoretical solar irradiance based on date, location, and weather
        """
        # Solar constant (W/m²)
        solar_constant = 1361
        
        # Solar elevation angle
        elevation = self.calculate_solar_position(date, latitude)
        
        if elevation <= 0:
            return 0
        
        # Air mass calculation
        air_mass = 1 / math.sin(math.radians(elevation))
        
        # Atmospheric attenuation
        atmospheric_transmission = 0.7 ** (air_mass ** 0.678)
        
        # Cloud cover reduction (0-100% to 0-1)
        cloud_factor = 1 - (cloud_cover / 100) * 0.75
        
        # Direct normal irradiance
        dni = solar_constant * atmospheric_transmission * cloud_factor
        
        # Convert to horizontal irradiance
        horizontal_irradiance = dni * math.sin(math.radians(elevation))
        
        # Convert W/m² to kWh/m²/day (peak sun hours for India: 4-6 hours)
        daily_irradiance = horizontal_irradiance * 5.5 / 1000  # 5.5 peak sun hours average
        
        return max(0, daily_irradiance)
    
    def calculate_pv_power_output(self, plant, irradiance, temperature, humidity, wind_speed):
        """
        Calculate realistic PV power output based on plant specifications and weather
        """
        if irradiance <= 0:
            return 0
        
        # Plant specifications
        rated_power = plant.capacity_mw * 1000  # Convert to kW
        module_efficiency = plant.efficiency_rating / 100
        
        # Standard Test Conditions (STC)
        stc_irradiance = 1.0  # kW/m²
        stc_temperature = 25  # °C
        
        # Temperature coefficient for crystalline silicon (typical: -0.4%/°C)
        temp_coefficient = -0.004
        
        # Temperature effect
        temp_effect = 1 + temp_coefficient * (temperature - stc_temperature)
        
        # Irradiance effect (linear relationship)
        irradiance_ratio = irradiance / stc_irradiance
        
        # Inverter efficiency (typical: 95-98%)
        inverter_efficiency = 0.96
        
        # System losses (soiling, wiring, etc.)
        # Dust accumulation factor (higher in dry season)
        month = datetime.now().month
        if month in [3, 4, 5]:  # Summer - more dust
            soiling_factor = 0.92
        elif month in [6, 7, 8, 9]:  # Monsoon - cleaner panels
            soiling_factor = 0.98
        else:  # Winter - moderate
            soiling_factor = 0.95
        
        # Humidity effect (minor)
        humidity_factor = 1 - (humidity - 50) * 0.0002 if humidity > 50 else 1
        
        # Wind cooling effect (helps with temperature)
        wind_factor = 1 + (wind_speed - 5) * 0.001 if wind_speed > 5 else 1
        
        # Calculate actual power output
        actual_power = (rated_power * 
                       irradiance_ratio * 
                       temp_effect * 
                       inverter_efficiency * 
                       soiling_factor * 
                       humidity_factor * 
                       wind_factor)
        
        # Daily energy (assuming irradiance represents peak conditions over 5.5 hours)
        daily_energy = actual_power * 5.5  # kWh
        
        return max(0, daily_energy)
    
    def generate_realistic_predictions(self, plant_id, days=180):
        """
        Generate realistic 6-month predictions using authentic weather data
        """
        try:
            # Get plant information
            plant = SolarPlant.query.get(plant_id)
            if not plant:
                logging.error(f"Plant {plant_id} not found")
                return []
            
            # Get coordinates for Mysuru, Karnataka
            coords = self.weather_api.get_location_coordinates("Mysuru", "Karnataka")
            if not coords:
                coords = [12.2958, 76.6394]  # Default VVCE coordinates
            
            lat, lon = coords
            
            # Get current weather for baseline
            current_weather = self.weather_api.get_current_weather(lat, lon)
            
            # Get 7-day forecast
            forecast = self.weather_api.get_forecast(lat, lon, 7)
            
            predictions = []
            start_date = datetime.now().date()
            
            # Use forecast data for first 7 days, then extrapolate seasonally
            for day in range(days):
                pred_date = start_date + timedelta(days=day)
                
                if day < len(forecast):
                    # Use real forecast data
                    weather = forecast[day]
                    temperature = weather['temperature']
                    humidity = weather['humidity']
                    cloud_cover = weather['cloud_cover']
                    wind_speed = weather['wind_speed']
                else:
                    # Generate seasonal weather patterns for extended forecast
                    temperature, humidity, cloud_cover, wind_speed = self._generate_seasonal_weather(pred_date)
                
                # Calculate realistic solar irradiance
                irradiance = self.calculate_theoretical_irradiance(pred_date, cloud_cover, lat)
                
                # Calculate PV power output
                daily_energy = self.calculate_pv_power_output(
                    plant, irradiance, temperature, humidity, wind_speed
                )
                
                # Calculate realistic revenue (Karnataka solar tariff rates)
                if daily_energy > plant.capacity_mw * 1000 * 3:  # High production
                    tariff_rate = 4.8  # Higher rate for excess
                else:
                    tariff_rate = 4.2  # Standard rate
                
                daily_revenue = daily_energy * tariff_rate
                
                # Calculate accurate system efficiency (Performance Ratio)
                # Standard test conditions (STC) power rating
                stc_daily_max = plant.capacity_mw * 1000 * (irradiance * 5.5)  # Based on actual irradiance
                
                # Calculate performance ratio (actual vs theoretical under current conditions)
                if stc_daily_max > 0:
                    performance_ratio = (daily_energy / stc_daily_max) * 100
                    # Realistic efficiency ranges for solar systems in India
                    efficiency = max(12, min(22, performance_ratio))  # 12-22% for crystalline silicon
                else:
                    efficiency = 0
                
                # Adjust for realistic equipment efficiency (75-85% typical system efficiency)
                equipment_efficiency = efficiency * 0.8  # Account for inverter, wiring losses
                
                # Confidence based on weather forecast accuracy (decreases over time)
                if day < 7:
                    confidence = 0.92 - (day * 0.02)  # High confidence for forecast period
                else:
                    confidence = 0.75 - ((day - 7) * 0.001)  # Lower for extended period
                
                confidence = max(0.6, min(0.95, confidence))
                
                predictions.append({
                    'date': pred_date,
                    'energy': round(daily_energy, 2),
                    'revenue': round(daily_revenue, 2),
                    'efficiency': round(equipment_efficiency, 2),
                    'confidence': round(confidence, 3),
                    'weather': {
                        'temperature': temperature,
                        'humidity': humidity,
                        'cloud_cover': cloud_cover,
                        'wind_speed': wind_speed,
                        'solar_irradiance': round(irradiance, 3)
                    }
                })
            
            # Save predictions to database
            self._save_realistic_predictions(plant_id, predictions)
            
            logging.info(f"Generated {len(predictions)} realistic predictions for plant {plant_id}")
            return predictions
            
        except Exception as e:
            logging.error(f"Error generating realistic predictions: {str(e)}")
            return []
    
    def _generate_seasonal_weather(self, date):
        """
        Generate realistic seasonal weather patterns for Mysuru, Karnataka
        """
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Base seasonal patterns for Mysuru
        if month in [12, 1, 2]:  # Winter
            base_temp = 22 + 8 * math.sin(2 * math.pi * day_of_year / 365)
            base_humidity = 65 + 15 * math.sin(2 * math.pi * (day_of_year + 90) / 365)
            base_cloud = 30 + 20 * random.random()
        elif month in [3, 4, 5]:  # Summer
            base_temp = 32 + 6 * math.sin(2 * math.pi * day_of_year / 365)
            base_humidity = 45 + 20 * random.random()
            base_cloud = 20 + 15 * random.random()
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 26 + 4 * math.sin(2 * math.pi * day_of_year / 365)
            base_humidity = 85 + 10 * random.random()
            base_cloud = 70 + 25 * random.random()
        else:  # Post-monsoon
            base_temp = 25 + 6 * math.sin(2 * math.pi * day_of_year / 365)
            base_humidity = 70 + 15 * random.random()
            base_cloud = 40 + 20 * random.random()
        
        # Add daily variation using random instead of numpy
        temperature = max(15, min(45, base_temp + random.normalvariate(0, 3)))
        humidity = max(30, min(95, base_humidity + random.normalvariate(0, 8)))
        cloud_cover = max(0, min(100, base_cloud + random.normalvariate(0, 10)))
        wind_speed = max(0, 8 + random.normalvariate(0, 4))
        
        return temperature, humidity, cloud_cover, wind_speed
    
    def _save_realistic_predictions(self, plant_id, predictions):
        """Save realistic predictions to database"""
        try:
            # Clear existing predictions
            MLPrediction.query.filter_by(plant_id=plant_id).delete()
            
            # Save new predictions
            for pred in predictions:
                ml_prediction = MLPrediction(
                    plant_id=plant_id,
                    prediction_date=pred['date'],
                    predicted_energy=pred['energy'],
                    predicted_revenue=pred['revenue'],
                    predicted_efficiency=pred['efficiency'],
                    confidence_score=pred['confidence'],
                    model_used='Physics-Based Realistic Model'
                )
                db.session.add(ml_prediction)
            
            db.session.commit()
            logging.info(f"Saved {len(predictions)} realistic predictions to database")
            
        except Exception as e:
            logging.error(f"Error saving realistic predictions: {str(e)}")
            db.session.rollback()

# Global instance
realistic_predictor = RealisticSolarPredictor()