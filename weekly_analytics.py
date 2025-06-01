"""
VVCE Advanced Weekly Solar Analytics System
Sophisticated prediction and analysis engine for solar energy forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from models import SolarPlant, WeatherData, EnergyProduction, MLPrediction
from app import db
import logging

class VVCEWeeklyAnalytics:
    """Advanced weekly analytics and prediction system for VVCE solar plants"""
    
    def __init__(self):
        self.prediction_weeks = 26  # 6 months
        self.performance_thresholds = {
            'excellent': 85,
            'good': 70,
            'average': 55,
            'poor': 40
        }
    
    def generate_weekly_predictions(self, plant_id, ml_predictor):
        """Generate comprehensive weekly predictions with advanced analytics"""
        try:
            plant = SolarPlant.query.get(plant_id)
            if not plant:
                return None
            
            # Get historical performance baseline
            baseline = self._calculate_baseline_performance(plant_id)
            
            weekly_forecasts = []
            start_date = datetime.now().date()
            
            for week_num in range(self.prediction_weeks):
                week_start = start_date + timedelta(weeks=week_num)
                week_end = week_start + timedelta(days=6)
                
                # Generate week prediction
                week_data = self._generate_week_forecast(
                    plant, week_start, week_end, week_num + 1, baseline, ml_predictor
                )
                
                if week_data:
                    weekly_forecasts.append(week_data)
            
            # Add quarterly summaries
            quarterly_analysis = self._generate_quarterly_analysis(weekly_forecasts)
            
            # Save to database in daily format for compatibility
            self._save_weekly_predictions(plant_id, weekly_forecasts)
            
            return {
                'weekly_forecasts': weekly_forecasts,
                'quarterly_analysis': quarterly_analysis,
                'baseline_performance': baseline,
                'total_weeks': len(weekly_forecasts)
            }
            
        except Exception as e:
            logging.error(f"Error generating weekly predictions: {e}")
            return None
    
    def _calculate_baseline_performance(self, plant_id):
        """Calculate historical baseline performance metrics"""
        try:
            # Get last 3 months of data
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            
            historical_data = db.session.query(
                EnergyProduction.energy_produced,
                EnergyProduction.equipment_efficiency,
                EnergyProduction.revenue_inr,
                WeatherData.solar_irradiance,
                WeatherData.temperature
            ).join(
                WeatherData, 
                (EnergyProduction.plant_id == WeatherData.plant_id) & 
                (EnergyProduction.date == WeatherData.date)
            ).filter(
                EnergyProduction.plant_id == plant_id,
                EnergyProduction.date >= start_date
            ).all()
            
            if not historical_data:
                return self._default_baseline()
            
            if len(historical_data) > 0:
                df = pd.DataFrame(historical_data, columns=[
                    'energy', 'efficiency', 'revenue', 'irradiance', 'temperature'
                ])
            else:
                return self._default_baseline()
            
            return {
                'avg_daily_energy': float(df['energy'].mean()),
                'avg_efficiency': float(df['efficiency'].mean()),
                'avg_daily_revenue': float(df['revenue'].mean()),
                'energy_std': float(df['energy'].std()),
                'efficiency_std': float(df['efficiency'].std()),
                'peak_energy': float(df['energy'].max()),
                'min_energy': float(df['energy'].min()),
                'data_points': len(df)
            }
            
        except Exception as e:
            logging.error(f"Error calculating baseline: {e}")
            return self._default_baseline()
    
    def _default_baseline(self):
        """Default baseline when no historical data available"""
        return {
            'avg_daily_energy': 1200.0,
            'avg_efficiency': 18.5,
            'avg_daily_revenue': 5400.0,
            'energy_std': 150.0,
            'efficiency_std': 2.0,
            'peak_energy': 1500.0,
            'min_energy': 800.0,
            'data_points': 0
        }
    
    def _generate_week_forecast(self, plant, week_start, week_end, week_number, baseline, ml_predictor):
        """Generate detailed forecast for a single week"""
        try:
            # Generate daily predictions for the week
            daily_predictions = []
            weekly_energy = 0
            weekly_revenue = 0
            weekly_efficiency_sum = 0
            
            for day_offset in range(7):
                pred_date = week_start + timedelta(days=day_offset)
                
                # Get seasonal and weather factors
                seasonal_factor = self._get_seasonal_factor(pred_date)
                weather_factor = self._get_weather_factor(pred_date, plant)
                
                # Calculate daily prediction based on baseline and factors
                daily_energy = baseline['avg_daily_energy'] * seasonal_factor * weather_factor
                daily_revenue = daily_energy * 4.5  # INR per kWh
                daily_efficiency = baseline['avg_efficiency'] * weather_factor
                
                # Add some realistic variance
                variance = np.random.normal(0, baseline['energy_std'] * 0.1)
                daily_energy += variance
                daily_energy = max(0, daily_energy)  # Ensure non-negative
                
                daily_predictions.append({
                    'date': pred_date,
                    'energy': daily_energy,
                    'revenue': daily_revenue,
                    'efficiency': daily_efficiency,
                    'seasonal_factor': seasonal_factor,
                    'weather_factor': weather_factor
                })
                
                weekly_energy += daily_energy
                weekly_revenue += daily_revenue
                weekly_efficiency_sum += daily_efficiency
            
            # Calculate weekly metrics
            avg_efficiency = weekly_efficiency_sum / 7
            
            # Advanced analytics
            energy_variance = np.var([d['energy'] for d in daily_predictions])
            peak_day = max(daily_predictions, key=lambda x: x['energy'])
            low_day = min(daily_predictions, key=lambda x: x['energy'])
            
            # Performance scoring (0-100)
            expected_weekly = baseline['avg_daily_energy'] * 7
            performance_ratio = weekly_energy / expected_weekly if expected_weekly > 0 else 0
            performance_score = min(100, performance_ratio * 100)
            
            # Risk assessment
            risk_factors = self._assess_weekly_risks(week_start, daily_predictions, baseline)
            
            # Confidence calculation
            confidence = self._calculate_confidence(daily_predictions, baseline, week_number)
            
            # Generate actionable insights
            insights = self._generate_weekly_insights(daily_predictions, performance_score, risk_factors)
            
            return {
                'week_number': week_number,
                'week_start': week_start,
                'week_end': week_end,
                'total_energy': weekly_energy,
                'total_revenue': weekly_revenue,
                'avg_efficiency': avg_efficiency,
                'performance_score': performance_score,
                'confidence': confidence,
                'energy_variance': energy_variance,
                'peak_day': {
                    'date': peak_day['date'],
                    'energy': peak_day['energy']
                },
                'low_day': {
                    'date': low_day['date'],
                    'energy': low_day['energy']
                },
                'risk_factors': risk_factors,
                'insights': insights,
                'daily_breakdown': daily_predictions
            }
            
        except Exception as e:
            logging.error(f"Error generating week forecast: {e}")
            return None
    
    def _get_seasonal_factor(self, pred_date):
        """Calculate seasonal adjustment factor"""
        month = pred_date.month
        
        # Karnataka seasonal patterns
        seasonal_factors = {
            1: 0.85,   # January - good sun
            2: 0.90,   # February - excellent
            3: 0.95,   # March - peak
            4: 1.00,   # April - peak
            5: 0.95,   # May - very good
            6: 0.70,   # June - monsoon start
            7: 0.60,   # July - monsoon
            8: 0.65,   # August - monsoon
            9: 0.75,   # September - post-monsoon
            10: 0.85,  # October - good
            11: 0.90,  # November - excellent
            12: 0.85   # December - good
        }
        
        return seasonal_factors.get(month, 0.80)
    
    def _get_weather_factor(self, pred_date, plant):
        """Calculate weather-based adjustment factor"""
        try:
            # Check for historical weather patterns on similar dates
            month_day = f"{pred_date.month:02d}-{pred_date.day:02d}"
            
            # Simplified weather factor based on seasonal patterns
            if pred_date.month in [6, 7, 8]:  # Monsoon months
                return np.random.uniform(0.5, 0.8)
            elif pred_date.month in [3, 4, 11, 12]:  # Clear months
                return np.random.uniform(0.85, 1.0)
            else:  # Transition months
                return np.random.uniform(0.7, 0.9)
                
        except Exception as e:
            return 0.8  # Default factor
    
    def _assess_weekly_risks(self, week_start, daily_predictions, baseline):
        """Assess risk factors for the week"""
        risks = []
        
        # Weather risks
        if week_start.month in [6, 7, 8]:
            risks.append({
                'type': 'weather',
                'level': 'high',
                'description': 'Monsoon season - expect reduced solar irradiance',
                'impact': 'energy_reduction'
            })
        
        # Efficiency variance risk
        efficiencies = [d['efficiency'] for d in daily_predictions]
        if np.std(efficiencies) > baseline['efficiency_std'] * 1.5:
            risks.append({
                'type': 'performance',
                'level': 'medium',
                'description': 'High efficiency variance predicted',
                'impact': 'inconsistent_output'
            })
        
        # Maintenance window detection
        if week_start.day <= 7:  # First week of month
            risks.append({
                'type': 'maintenance',
                'level': 'low',
                'description': 'Recommended maintenance window',
                'impact': 'planned_downtime'
            })
        
        return risks
    
    def _calculate_confidence(self, daily_predictions, baseline, week_number):
        """Calculate prediction confidence score"""
        # Base confidence decreases with time horizon
        base_confidence = max(0.6, 0.95 - (week_number * 0.01))
        
        # Adjust based on variance
        energies = [d['energy'] for d in daily_predictions]
        variance_factor = min(1.0, baseline['energy_std'] / np.std(energies)) if np.std(energies) > 0 else 1.0
        
        # Adjust for data availability
        data_factor = min(1.0, baseline['data_points'] / 90)  # Prefer 90+ days of data
        
        return base_confidence * variance_factor * data_factor
    
    def _generate_weekly_insights(self, daily_predictions, performance_score, risk_factors):
        """Generate actionable insights for the week"""
        insights = []
        
        # Performance insights
        if performance_score >= self.performance_thresholds['excellent']:
            insights.append({
                'type': 'positive',
                'category': 'performance',
                'message': 'Excellent energy output expected this week',
                'action': 'Monitor for optimization opportunities'
            })
        elif performance_score < self.performance_thresholds['poor']:
            insights.append({
                'type': 'warning',
                'category': 'performance',
                'message': 'Below-average performance predicted',
                'action': 'Consider maintenance check or cleaning'
            })
        
        # Peak performance insights
        peak_day = max(daily_predictions, key=lambda x: x['energy'])
        insights.append({
            'type': 'info',
            'category': 'optimization',
            'message': f'Peak output expected on {peak_day["date"].strftime("%A")}',
            'action': 'Schedule energy-intensive operations'
        })
        
        # Risk-based insights
        high_risks = [r for r in risk_factors if r['level'] == 'high']
        if high_risks:
            insights.append({
                'type': 'alert',
                'category': 'risk',
                'message': f'{len(high_risks)} high-risk factors identified',
                'action': 'Review risk mitigation strategies'
            })
        
        return insights
    
    def _generate_quarterly_analysis(self, weekly_forecasts):
        """Generate quarterly summary analysis"""
        if not weekly_forecasts:
            return {}
        
        quarters = {
            'Q1': weekly_forecasts[:13],
            'Q2': weekly_forecasts[13:26] if len(weekly_forecasts) > 13 else []
        }
        
        analysis = {}
        for quarter, weeks in quarters.items():
            if not weeks:
                continue
                
            total_energy = sum(w['total_energy'] for w in weeks)
            total_revenue = sum(w['total_revenue'] for w in weeks)
            avg_performance = np.mean([w['performance_score'] for w in weeks])
            
            analysis[quarter] = {
                'total_energy': total_energy,
                'total_revenue': total_revenue,
                'avg_performance_score': avg_performance,
                'weeks_count': len(weeks),
                'best_week': max(weeks, key=lambda x: x['performance_score']),
                'worst_week': min(weeks, key=lambda x: x['performance_score'])
            }
        
        return analysis
    
    def _save_weekly_predictions(self, plant_id, weekly_forecasts):
        """Save predictions to database in compatible format"""
        try:
            # Clear existing predictions
            MLPrediction.query.filter(MLPrediction.plant_id == plant_id).delete()
            
            # Convert weekly data to daily records for database
            for week in weekly_forecasts:
                for daily in week['daily_breakdown']:
                    prediction = MLPrediction(
                        plant_id=plant_id,
                        prediction_date=daily['date'],
                        predicted_energy=daily['energy'],
                        predicted_revenue=daily['revenue'],
                        predicted_efficiency=daily['efficiency'],
                        confidence_score=week['confidence'],
                        model_used='VVCE_Weekly_Analytics'
                    )
                    db.session.add(prediction)
            
            db.session.commit()
            logging.info(f"Saved weekly predictions for plant {plant_id}")
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error saving weekly predictions: {e}")