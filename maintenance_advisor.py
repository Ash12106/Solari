"""
VVCE Smart Maintenance & Cleaning Advisor
Provides intelligent maintenance recommendations for solar panels at VVCE
"""

import logging
from datetime import datetime, timedelta, date
from models import SolarPlant, EnergyProduction, WeatherData, MaintenanceRecord
from app import db
import numpy as np

class VVCECleaningAdvisor:
    def __init__(self):
        self.dust_threshold = 7  # Scale 1-10
        self.efficiency_drop_threshold = 5  # Percentage
        self.weather_factors = ['humidity', 'wind_speed', 'cloud_cover']
        self.mysuru_climate_factors = {
            'monsoon_months': [6, 7, 8, 9],  # June to September
            'dry_months': [11, 12, 1, 2],    # November to February
            'dust_storm_season': [3, 4, 5]   # March to May
        }
    
    def should_clean_panels(self, plant_id):
        """
        Determine if panels need cleaning based on:
        - Power output drop
        - Dust accumulation estimate
        - Weather conditions
        - Days since last cleaning
        """
        try:
            # Get recent performance data
            recent_data = self._get_recent_performance(plant_id, days=7)
            if not recent_data:
                return {"should_clean": False, "reason": "Insufficient data"}
            
            # Calculate efficiency drop
            efficiency_drop = self._calculate_efficiency_drop(plant_id)
            
            # Estimate dust accumulation
            dust_level = self._estimate_dust_accumulation(plant_id)
            
            # Get days since last cleaning
            days_since_cleaning = self._days_since_last_cleaning(plant_id)
            
            # Weather impact assessment
            weather_impact = self._assess_weather_impact(plant_id)
            
            # Decision logic
            should_clean = False
            priority = "low"
            reasons = []
            
            if efficiency_drop > 15:
                should_clean = True
                priority = "urgent"
                reasons.append(f"Efficiency dropped by {efficiency_drop:.1f}%")
            elif efficiency_drop > 8:
                should_clean = True
                priority = "high"
                reasons.append(f"Efficiency dropped by {efficiency_drop:.1f}%")
            elif efficiency_drop > 5:
                should_clean = True
                priority = "medium"
                reasons.append(f"Efficiency dropped by {efficiency_drop:.1f}%")
            
            if dust_level > 8:
                should_clean = True
                if priority == "low":
                    priority = "high"
                reasons.append(f"High dust accumulation estimated (level {dust_level}/10)")
            elif dust_level > 6:
                should_clean = True
                if priority == "low":
                    priority = "medium"
                reasons.append(f"Moderate dust accumulation (level {dust_level}/10)")
            
            if days_since_cleaning > 30:
                should_clean = True
                reasons.append(f"Last cleaning was {days_since_cleaning} days ago")
            
            # Weather-based recommendations
            if weather_impact['dust_storm_risk']:
                should_clean = True
                reasons.append("Post dust-storm cleaning recommended")
            
            if weather_impact['monsoon_approaching']:
                reasons.append("Clean before monsoon for optimal performance")
            
            return {
                "should_clean": should_clean,
                "priority": priority,
                "reasons": reasons,
                "efficiency_drop": efficiency_drop,
                "dust_level": dust_level,
                "days_since_cleaning": days_since_cleaning,
                "weather_impact": weather_impact
            }
            
        except Exception as e:
            logging.error(f"Error in cleaning advisor: {e}")
            return {"should_clean": False, "reason": "Error in analysis"}
    
    def generate_cleaning_schedule(self, plant_id):
        """
        Generate optimal cleaning schedule for VVCE
        considering academic calendar and weather
        """
        try:
            schedule = []
            current_date = date.today()
            
            # Generate schedule for next 6 months
            for i in range(180):  # 6 months
                check_date = current_date + timedelta(days=i)
                month = check_date.month
                
                # Determine cleaning frequency based on season
                if month in self.mysuru_climate_factors['dust_storm_season']:
                    # During dust storm season (Mar-May), clean every 10 days
                    if i % 10 == 0:
                        schedule.append({
                            'date': check_date,
                            'type': 'Regular Cleaning',
                            'priority': 'high',
                            'reason': 'Dust storm season - increased frequency'
                        })
                
                elif month in self.mysuru_climate_factors['monsoon_months']:
                    # During monsoon (Jun-Sep), clean every 20 days
                    if i % 20 == 0:
                        schedule.append({
                            'date': check_date,
                            'type': 'Monsoon Cleaning',
                            'priority': 'medium',
                            'reason': 'Monsoon season maintenance'
                        })
                
                else:
                    # During dry season (Oct-Feb), clean every 15 days
                    if i % 15 == 0:
                        schedule.append({
                            'date': check_date,
                            'type': 'Regular Cleaning',
                            'priority': 'medium',
                            'reason': 'Regular maintenance schedule'
                        })
                
                # Special academic calendar considerations
                if self._is_semester_break(check_date):
                    # Intensive maintenance during semester breaks
                    if i % 30 == 0:
                        schedule.append({
                            'date': check_date,
                            'type': 'Deep Cleaning & Inspection',
                            'priority': 'high',
                            'reason': 'Semester break - comprehensive maintenance'
                        })
            
            return schedule
            
        except Exception as e:
            logging.error(f"Error generating cleaning schedule: {e}")
            return []
    
    def _get_recent_performance(self, plant_id, days=7):
        """Get recent performance data"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        return EnergyProduction.query.filter(
            EnergyProduction.plant_id == plant_id,
            EnergyProduction.date >= start_date
        ).order_by(EnergyProduction.date.desc()).all()
    
    def _calculate_efficiency_drop(self, plant_id):
        """Calculate efficiency drop compared to optimal performance"""
        # Get last 7 days vs previous 7 days
        recent_data = self._get_recent_performance(plant_id, 7)
        previous_data = self._get_previous_performance(plant_id, 7, 14)
        
        if not recent_data or not previous_data:
            return 0
        
        recent_avg = np.mean([d.equipment_efficiency for d in recent_data])
        previous_avg = np.mean([d.equipment_efficiency for d in previous_data])
        
        return max(0, previous_avg - recent_avg)
    
    def _get_previous_performance(self, plant_id, days, offset):
        """Get performance data from a previous period"""
        end_date = date.today() - timedelta(days=offset)
        start_date = end_date - timedelta(days=days)
        
        return EnergyProduction.query.filter(
            EnergyProduction.plant_id == plant_id,
            EnergyProduction.date >= start_date,
            EnergyProduction.date <= end_date
        ).all()
    
    def _estimate_dust_accumulation(self, plant_id):
        """Estimate dust accumulation level (1-10 scale)"""
        # Get weather data for last 7 days
        recent_weather = WeatherData.query.filter(
            WeatherData.plant_id == plant_id,
            WeatherData.date >= date.today() - timedelta(days=7)
        ).all()
        
        if not recent_weather:
            return 5  # Default moderate level
        
        # Calculate dust factors
        avg_wind = np.mean([w.wind_speed for w in recent_weather])
        avg_humidity = np.mean([w.humidity for w in recent_weather])
        
        # Lower wind and humidity = higher dust accumulation
        dust_level = 5  # Base level
        
        if avg_wind < 10:  # Low wind
            dust_level += 2
        elif avg_wind > 20:  # High wind (cleans panels)
            dust_level -= 1
        
        if avg_humidity < 40:  # Dry conditions
            dust_level += 1
        elif avg_humidity > 70:  # Humid conditions (settles dust)
            dust_level += 0.5
        
        # Season factor
        current_month = date.today().month
        if current_month in self.mysuru_climate_factors['dust_storm_season']:
            dust_level += 2
        
        return min(10, max(1, int(dust_level)))
    
    def _days_since_last_cleaning(self, plant_id):
        """Get days since last cleaning maintenance"""
        last_cleaning = MaintenanceRecord.query.filter(
            MaintenanceRecord.plant_id == plant_id,
            MaintenanceRecord.maintenance_type.like('%Cleaning%')
        ).order_by(MaintenanceRecord.date.desc()).first()
        
        if last_cleaning:
            return (date.today() - last_cleaning.date).days
        else:
            return 30  # Assume 30 days if no record
    
    def _assess_weather_impact(self, plant_id):
        """Assess weather impact on cleaning needs"""
        current_month = date.today().month
        
        return {
            'dust_storm_risk': current_month in self.mysuru_climate_factors['dust_storm_season'],
            'monsoon_approaching': current_month == 5,  # May - before monsoon
            'dry_season': current_month in self.mysuru_climate_factors['dry_months'],
            'monsoon_active': current_month in self.mysuru_climate_factors['monsoon_months']
        }
    
    def _is_semester_break(self, check_date):
        """Check if date falls during semester break"""
        month = check_date.month
        
        # Typical academic calendar breaks
        summer_break = month in [4, 5]  # April-May
        winter_break = month in [12, 1]  # December-January
        
        return summer_break or winter_break

# Alert types for VVCE system
VVCE_ALERT_TYPES = {
    'CLEANING_URGENT': {
        'message': '🚨 Urgent: Panel cleaning required - {}% efficiency drop detected',
        'priority': 'high',
        'action': 'Schedule cleaning within 24 hours',
        'icon': 'fas fa-exclamation-triangle'
    },
    'CLEANING_RECOMMENDED': {
        'message': '🧹 Cleaning recommended - Dust accumulation level {} detected',
        'priority': 'medium',
        'action': 'Schedule cleaning within 3-5 days',
        'icon': 'fas fa-brush'
    },
    'MAINTENANCE_DUE': {
        'message': '🔧 Scheduled maintenance due for {}',
        'priority': 'medium',
        'action': 'Contact VVCE facilities team',
        'icon': 'fas fa-wrench'
    },
    'WEATHER_ALERT': {
        'message': '🌦️ Weather conditions may affect performance - {}',
        'priority': 'low',
        'action': 'Monitor performance closely',
        'icon': 'fas fa-cloud-sun'
    },
    'SEMESTER_MAINTENANCE': {
        'message': '🎓 Semester break - Ideal time for comprehensive maintenance',
        'priority': 'medium',
        'action': 'Schedule deep cleaning and inspection',
        'icon': 'fas fa-graduation-cap'
    }
}

def generate_vvce_alerts(plant_id):
    """Generate maintenance alerts for VVCE solar plant"""
    advisor = VVCECleaningAdvisor()
    cleaning_analysis = advisor.should_clean_panels(plant_id)
    
    alerts = []
    
    if cleaning_analysis['should_clean']:
        if cleaning_analysis['priority'] == 'urgent':
            alert_type = VVCE_ALERT_TYPES['CLEANING_URGENT']
            message = alert_type['message'].format(cleaning_analysis['efficiency_drop'])
        else:
            alert_type = VVCE_ALERT_TYPES['CLEANING_RECOMMENDED']
            message = alert_type['message'].format(cleaning_analysis['dust_level'])
        
        alerts.append({
            'type': 'cleaning',
            'message': message,
            'priority': cleaning_analysis['priority'],
            'action': alert_type['action'],
            'icon': alert_type['icon'],
            'details': cleaning_analysis['reasons']
        })
    
    # Weather-based alerts
    weather_impact = cleaning_analysis.get('weather_impact', {})
    if weather_impact.get('dust_storm_risk'):
        alert_type = VVCE_ALERT_TYPES['WEATHER_ALERT']
        alerts.append({
            'type': 'weather',
            'message': alert_type['message'].format('Dust storm season active'),
            'priority': alert_type['priority'],
            'action': alert_type['action'],
            'icon': alert_type['icon']
        })
    
    # Semester break maintenance
    current_date = date.today()
    advisor_instance = VVCECleaningAdvisor()
    if advisor_instance._is_semester_break(current_date):
        alert_type = VVCE_ALERT_TYPES['SEMESTER_MAINTENANCE']
        alerts.append({
            'type': 'semester',
            'message': alert_type['message'],
            'priority': alert_type['priority'],
            'action': alert_type['action'],
            'icon': alert_type['icon']
        })
    
    return alerts

# Global advisor instance
vvce_advisor = VVCECleaningAdvisor()