import random
import numpy as np
from datetime import datetime, timedelta, date
from models import SolarPlant, WeatherData, EnergyProduction, MaintenanceRecord, MLPrediction, ModelPerformance
from app import db
import logging

def initialize_data():
    """Initialize the database with sample data if it's empty"""
    try:
        # Check if VVCE data already exists
        vvce_plants = SolarPlant.query.filter(SolarPlant.name.like('%VVCE%')).count()
        if vvce_plants >= 5:
            logging.info("VVCE plants already exist, skipping initialization")
            return
        
        logging.info("Initializing sample data...")
        
        # Create VVCE solar plants in different college blocks
        plants = [
            {
                'name': 'VVCE Administrative Block Solar Plant',
                'location': 'Mysuru, Karnataka',
                'capacity_mw': 1.5,
                'panel_type': 'Monocrystalline',
                'efficiency_rating': 20.8
            },
            {
                'name': 'VVCE Boys Hostel Block Solar Plant',
                'location': 'Mysuru, Karnataka',
                'capacity_mw': 0.8,
                'panel_type': 'Polycrystalline',
                'efficiency_rating': 18.5
            },
            {
                'name': 'VVCE Engineering Block Solar Plant',
                'location': 'Mysuru, Karnataka',
                'capacity_mw': 1.2,
                'panel_type': 'Monocrystalline',
                'efficiency_rating': 19.9
            },
            {
                'name': 'VVCE Library Block Solar Plant',
                'location': 'Mysuru, Karnataka',
                'capacity_mw': 0.6,
                'panel_type': 'Monocrystalline',
                'efficiency_rating': 21.2
            },
            {
                'name': 'VVCE Cafeteria Block Solar Plant',
                'location': 'Mysuru, Karnataka',
                'capacity_mw': 0.4,
                'panel_type': 'Polycrystalline',
                'efficiency_rating': 17.8
            }
        ]
        
        plant_objects = []
        for plant_data in plants:
            plant = SolarPlant(
                name=plant_data['name'],
                location=plant_data['location'],
                capacity_mw=plant_data['capacity_mw'],
                installation_date=datetime(2023, 6, 1).date(),
                panel_type=plant_data['panel_type'],
                efficiency_rating=plant_data['efficiency_rating']
            )
            db.session.add(plant)
            plant_objects.append(plant)
        
        db.session.commit()
        
        # Generate 6 months of historical data
        start_date = date.today() - timedelta(days=180)
        end_date = date.today()
        
        for plant in plant_objects:
            generate_historical_data(plant, start_date, end_date)
        
        logging.info("Sample data initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing data: {e}")
        db.session.rollback()

def generate_historical_data(plant, start_date, end_date):
    """Generate realistic historical data for a solar plant"""
    try:
        current_date = start_date
        
        # Base values for the location
        location_factors = get_location_factors(plant.location)
        
        while current_date <= end_date:
            # Generate weather data
            weather = generate_daily_weather(current_date, location_factors)
            weather_record = WeatherData(
                plant_id=plant.id,
                date=current_date,
                temperature=weather['temperature'],
                humidity=weather['humidity'],
                cloud_cover=weather['cloud_cover'],
                wind_speed=weather['wind_speed'],
                solar_irradiance=weather['solar_irradiance']
            )
            db.session.add(weather_record)
            
            # Generate energy production data
            energy_data = calculate_energy_production(plant, weather, current_date)
            production_record = EnergyProduction(
                plant_id=plant.id,
                date=current_date,
                energy_produced=energy_data['energy_produced'],
                equipment_efficiency=energy_data['equipment_efficiency'],
                revenue_inr=energy_data['revenue_inr'],
                tariff_rate=energy_data['tariff_rate']
            )
            db.session.add(production_record)
            
            # Generate maintenance records (random schedule)
            if random.random() < 0.1:  # 10% chance of maintenance on any day
                maintenance = generate_maintenance_record(plant, current_date)
                if maintenance:
                    db.session.add(maintenance)
            
            current_date += timedelta(days=1)
        
        db.session.commit()
        logging.info(f"Generated historical data for {plant.name}")
        
    except Exception as e:
        logging.error(f"Error generating historical data for {plant.name}: {e}")
        db.session.rollback()

def get_location_factors(location):
    """Get location-specific factors for weather and solar data"""
    if 'rajasthan' in location.lower() or 'jaipur' in location.lower():
        return {
            'base_temperature': 28,
            'temp_variation': 15,
            'base_irradiance': 5.8,
            'humidity_factor': 0.8,
            'monsoon_impact': 0.7
        }
    elif 'gujarat' in location.lower() or 'ahmedabad' in location.lower():
        return {
            'base_temperature': 30,
            'temp_variation': 12,
            'base_irradiance': 5.6,
            'humidity_factor': 0.9,
            'monsoon_impact': 0.8
        }
    elif 'karnataka' in location.lower() or 'bangalore' in location.lower():
        return {
            'base_temperature': 25,
            'temp_variation': 8,
            'base_irradiance': 5.2,
            'humidity_factor': 1.1,
            'monsoon_impact': 0.6
        }
    else:
        # Default values
        return {
            'base_temperature': 28,
            'temp_variation': 12,
            'base_irradiance': 5.5,
            'humidity_factor': 1.0,
            'monsoon_impact': 0.75
        }

def generate_daily_weather(date, location_factors):
    """Generate realistic daily weather data"""
    month = date.month
    
    # Seasonal adjustments
    if 6 <= month <= 9:  # Monsoon season
        temp_factor = 0.85
        humidity_base = 85
        cloud_base = 70
        irradiance_factor = location_factors['monsoon_impact']
    elif 11 <= month <= 2:  # Winter
        temp_factor = 0.7
        humidity_base = 60
        cloud_base = 20
        irradiance_factor = 0.9
    elif 3 <= month <= 5:  # Summer
        temp_factor = 1.15
        humidity_base = 45
        cloud_base = 15
        irradiance_factor = 1.0
    else:  # Transition months
        temp_factor = 1.0
        humidity_base = 65
        cloud_base = 40
        irradiance_factor = 0.95
    
    # Generate weather parameters
    temperature = location_factors['base_temperature'] * temp_factor + random.gauss(0, 3)
    humidity = max(20, min(95, humidity_base + random.gauss(0, 15)))
    cloud_cover = max(0, min(100, cloud_base + random.gauss(0, 20)))
    wind_speed = max(0, random.gauss(12, 5))
    
    # Calculate solar irradiance
    base_irradiance = location_factors['base_irradiance']
    cloud_reduction = (cloud_cover / 100) * 0.8
    temp_efficiency = 1 - max(0, (temperature - 25) * 0.004)  # Efficiency decreases with heat
    
    solar_irradiance = base_irradiance * irradiance_factor * (1 - cloud_reduction) * temp_efficiency
    solar_irradiance = max(0.5, solar_irradiance)  # Minimum irradiance
    
    return {
        'temperature': round(temperature, 1),
        'humidity': round(humidity, 1),
        'cloud_cover': round(cloud_cover, 1),
        'wind_speed': round(wind_speed, 1),
        'solar_irradiance': round(solar_irradiance, 2)
    }

def calculate_energy_production(plant, weather, date):
    """Calculate daily energy production based on weather and plant characteristics"""
    # Base calculation: Capacity (MW) * Hours * Solar Irradiance * Efficiency
    peak_sun_hours = weather['solar_irradiance']  # Simplified: irradiance = peak sun hours
    
    # Equipment efficiency (starts at rated efficiency, degrades over time)
    days_since_installation = (date - datetime(2023, 6, 1).date()).days
    degradation_factor = 1 - (days_since_installation * 0.0002)  # 0.02% per day
    equipment_efficiency = plant.efficiency_rating * degradation_factor
    
    # Temperature coefficient (solar panels lose efficiency in high heat)
    temp_coefficient = 1 - max(0, (weather['temperature'] - 25) * 0.004)
    
    # Cloud and weather impact
    weather_efficiency = 1 - (weather['cloud_cover'] / 100) * 0.3
    
    # Calculate energy production (kWh)
    theoretical_max = plant.capacity_mw * 1000 * peak_sun_hours
    actual_efficiency = (equipment_efficiency / 100) * temp_coefficient * weather_efficiency
    energy_produced = theoretical_max * actual_efficiency
    
    # Add some random variation
    energy_produced *= random.uniform(0.9, 1.1)
    energy_produced = max(0, energy_produced)
    
    # Calculate revenue (Indian solar tariffs)
    tariff_rate = random.uniform(3.5, 5.5)  # INR per kWh
    revenue_inr = energy_produced * tariff_rate
    
    return {
        'energy_produced': round(energy_produced, 2),
        'equipment_efficiency': round(equipment_efficiency, 2),
        'revenue_inr': round(revenue_inr, 2),
        'tariff_rate': round(tariff_rate, 2)
    }

def generate_maintenance_record(plant, date):
    """Generate random maintenance records"""
    maintenance_types = [
        {'type': 'Panel Cleaning', 'cost_range': (5000, 15000), 'efficiency_impact': 2.0},
        {'type': 'Inverter Maintenance', 'cost_range': (20000, 50000), 'efficiency_impact': 1.5},
        {'type': 'Electrical Check', 'cost_range': (10000, 25000), 'efficiency_impact': 0.5},
        {'type': 'Performance Monitoring', 'cost_range': (8000, 18000), 'efficiency_impact': 0.0},
        {'type': 'Structure Inspection', 'cost_range': (15000, 35000), 'efficiency_impact': 0.2},
        {'type': 'Cable Replacement', 'cost_range': (25000, 75000), 'efficiency_impact': 1.0}
    ]
    
    maintenance_type = random.choice(maintenance_types)
    cost = random.uniform(*maintenance_type['cost_range'])
    
    return MaintenanceRecord(
        plant_id=plant.id,
        date=date,
        maintenance_type=maintenance_type['type'],
        cost_inr=round(cost, 2),
        description=f"Routine {maintenance_type['type'].lower()} performed",
        efficiency_impact=maintenance_type['efficiency_impact']
    )
