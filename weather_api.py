import requests
import logging
import os
from datetime import datetime, timedelta
import json

class WeatherAPI:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "demo_key")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat, lon):
        """Get current weather data"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_current_weather(data)
            else:
                logging.error(f"Weather API error: {response.status_code} - {response.text}")
                return self._get_fallback_weather()
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Weather API request failed: {e}")
            return self._get_fallback_weather()
        except Exception as e:
            logging.error(f"Weather API error: {e}")
            return self._get_fallback_weather()
    
    def get_forecast(self, lat, lon, days=7):
        """Get weather forecast"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast(data, days)
            else:
                logging.error(f"Forecast API error: {response.status_code} - {response.text}")
                return self._get_fallback_forecast(days)
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Forecast API request failed: {e}")
            return self._get_fallback_forecast(days)
        except Exception as e:
            logging.error(f"Forecast API error: {e}")
            return self._get_fallback_forecast(days)
    
    def _parse_current_weather(self, data):
        """Parse current weather response"""
        try:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'cloud_cover': data['clouds']['all'],
                'wind_speed': data['wind'].get('speed', 0) * 3.6,  # Convert m/s to km/h
                'solar_irradiance': self._estimate_solar_irradiance(
                    data['clouds']['all'], 
                    data['main']['temp']
                ),
                'description': data['weather'][0]['description'],
                'timestamp': datetime.utcnow()
            }
        except KeyError as e:
            logging.error(f"Error parsing weather data: {e}")
            return self._get_fallback_weather()
    
    def _parse_forecast(self, data, days):
        """Parse forecast response"""
        try:
            forecast = []
            current_date = datetime.now().date()
            
            for item in data['list'][:days * 8]:  # 8 entries per day (3-hour intervals)
                date = datetime.fromtimestamp(item['dt']).date()
                
                forecast_item = {
                    'date': date,
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'cloud_cover': item['clouds']['all'],
                    'wind_speed': item['wind'].get('speed', 0) * 3.6,
                    'solar_irradiance': self._estimate_solar_irradiance(
                        item['clouds']['all'], 
                        item['main']['temp']
                    ),
                    'description': item['weather'][0]['description']
                }
                forecast.append(forecast_item)
            
            return forecast
            
        except KeyError as e:
            logging.error(f"Error parsing forecast data: {e}")
            return self._get_fallback_forecast(days)
    
    def _estimate_solar_irradiance(self, cloud_cover, temperature):
        """Estimate solar irradiance based on cloud cover and temperature"""
        # Simplified estimation for Indian conditions
        base_irradiance = 5.5  # kWh/m²/day (typical for India)
        
        # Reduce irradiance based on cloud cover
        cloud_factor = 1 - (cloud_cover / 100) * 0.8
        
        # Temperature adjustment (optimal around 25°C)
        temp_factor = 1.0
        if temperature > 35:
            temp_factor = 0.95  # High temperatures reduce efficiency
        elif temperature < 15:
            temp_factor = 0.9   # Low temperatures also reduce efficiency
        
        return base_irradiance * cloud_factor * temp_factor
    
    def _get_fallback_weather(self):
        """Provide fallback weather data when API fails"""
        logging.warning("Using fallback weather data")
        return {
            'temperature': 28.0,
            'humidity': 65.0,
            'cloud_cover': 30.0,
            'wind_speed': 15.0,
            'solar_irradiance': 4.8,
            'description': 'API Unavailable - Typical Indian Weather',
            'timestamp': datetime.utcnow(),
            'fallback': True
        }
    
    def _get_fallback_forecast(self, days):
        """Provide fallback forecast data when API fails"""
        logging.warning("Using fallback forecast data")
        forecast = []
        current_date = datetime.now().date()
        
        for i in range(days):
            date = current_date + timedelta(days=i)
            forecast.append({
                'date': date,
                'temperature': 28.0 + (i % 5),  # Vary temperature slightly
                'humidity': 65.0 - (i % 10),
                'cloud_cover': 30.0 + (i % 20),
                'wind_speed': 15.0,
                'solar_irradiance': 4.8 - (i % 3) * 0.2,
                'description': 'API Unavailable - Typical Forecast',
                'fallback': True
            })
        
        return forecast
    
    def get_location_coordinates(self, city, state=""):
        """Get coordinates for a city (for Indian locations)"""
        indian_cities = {
            'mysuru': (12.2958, 76.6394),
            'mysore': (12.2958, 76.6394),
            'bangalore': (12.9716, 77.5946),
            'jaipur': (26.9124, 75.7873),
            'jodhpur': (26.2389, 73.0243),
            'bikaner': (28.0229, 73.3119),
            'ahmedabad': (23.0225, 72.5714),
            'gandhinagar': (23.2156, 72.6369),
            'rajkot': (22.3039, 70.8022),
            'hubli': (15.3647, 75.1240),
            'chennai': (13.0827, 80.2707),
            'coimbatore': (11.0168, 76.9558),
            'madurai': (9.9252, 78.1198),
            'hyderabad': (17.3850, 78.4867),
            'warangal': (17.9689, 79.5941),
            'pune': (18.5204, 73.8567),
            'nashik': (19.9975, 73.7898),
            'nagpur': (21.1458, 79.0882)
        }
        
        city_lower = city.lower()
        if city_lower in indian_cities:
            return indian_cities[city_lower]
        else:
            # Default to Mysuru for VVCE
            logging.warning(f"City {city} not found, defaulting to Mysuru")
            return indian_cities['mysuru']

# Global weather API instance
weather_api = WeatherAPI()
