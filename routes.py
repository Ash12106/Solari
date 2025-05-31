from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import SolarPlant, WeatherData, EnergyProduction, MLPrediction, ModelPerformance, MaintenanceRecord
from ml_models import ml_predictor
from weather_api import weather_api
from datetime import datetime, timedelta, date
import logging
import json

@app.route('/')
def index():
    """Home page with overview"""
    try:
        # Get total plants and basic stats
        total_plants = SolarPlant.query.count()
        total_capacity = db.session.query(db.func.sum(SolarPlant.capacity_mw)).scalar() or 0
        
        # Get recent energy production
        recent_production = db.session.query(
            db.func.sum(EnergyProduction.energy_produced)
        ).filter(
            EnergyProduction.date >= date.today() - timedelta(days=30)
        ).scalar() or 0
        
        # Get recent revenue
        recent_revenue = db.session.query(
            db.func.sum(EnergyProduction.revenue_inr)
        ).filter(
            EnergyProduction.date >= date.today() - timedelta(days=30)
        ).scalar() or 0
        
        # Get plants for selection
        plants = SolarPlant.query.all()
        
        stats = {
            'total_plants': total_plants,
            'total_capacity': round(total_capacity, 1),
            'recent_production': round(recent_production, 0),
            'recent_revenue': round(recent_revenue, 0)
        }
        
        return render_template('index.html', stats=stats, plants=plants)
        
    except Exception as e:
        logging.error(f"Error in index route: {e}")
        flash('Error loading dashboard data', 'error')
        return render_template('index.html', stats={}, plants=[])

@app.route('/dashboard')
@app.route('/dashboard/<int:plant_id>')
def dashboard(plant_id=None):
    """Main dashboard with charts and current data"""
    try:
        # Get plant
        if plant_id:
            plant = SolarPlant.query.get_or_404(plant_id)
        else:
            plant = SolarPlant.query.first()
            if not plant:
                flash('No solar plants found. Please add a plant first.', 'warning')
                return redirect(url_for('index'))
            plant_id = plant.id
        
        # Get current weather
        coords = weather_api.get_location_coordinates(plant.location.split(',')[0])
        current_weather = weather_api.get_current_weather(coords[0], coords[1])
        
        # Get recent production data (last 30 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        production_data = db.session.query(
            EnergyProduction.date,
            EnergyProduction.energy_produced,
            EnergyProduction.revenue_inr,
            EnergyProduction.equipment_efficiency,
            WeatherData.temperature,
            WeatherData.solar_irradiance
        ).join(
            WeatherData, 
            (EnergyProduction.plant_id == WeatherData.plant_id) & 
            (EnergyProduction.date == WeatherData.date)
        ).filter(
            EnergyProduction.plant_id == plant_id,
            EnergyProduction.date >= start_date
        ).order_by(EnergyProduction.date).all()
        
        # Get predictions
        predictions = MLPrediction.query.filter(
            MLPrediction.plant_id == plant_id
        ).order_by(MLPrediction.prediction_date).limit(30).all()
        
        # Get model performance
        model_performance = ModelPerformance.query.order_by(
            ModelPerformance.training_date.desc()
        ).limit(5).all()
        
        # Calculate current efficiency
        if production_data:
            latest_production = production_data[-1]
            max_possible = plant.capacity_mw * 1000 * 8  # 8 hours peak sun
            current_efficiency = (latest_production.energy_produced / max_possible) * 100
        else:
            current_efficiency = 0
        
        # Get all plants for navigation
        all_plants = SolarPlant.query.all()
        
        return render_template('dashboard.html', 
                             plant=plant,
                             all_plants=all_plants,
                             current_weather=current_weather,
                             production_data=production_data,
                             predictions=predictions,
                             model_performance=model_performance,
                             current_efficiency=current_efficiency)
        
    except Exception as e:
        logging.error(f"Error in dashboard route: {e}")
        flash('Error loading dashboard', 'error')
        return redirect(url_for('index'))

@app.route('/predictions')
@app.route('/predictions/<int:plant_id>')
def predictions(plant_id=None):
    """Detailed predictions page"""
    try:
        # Get plant
        if plant_id:
            plant = SolarPlant.query.get_or_404(plant_id)
        else:
            plant = SolarPlant.query.first()
            if not plant:
                flash('No solar plants found', 'warning')
                return redirect(url_for('index'))
            plant_id = plant.id
        
        # Get all predictions
        all_predictions = MLPrediction.query.filter(
            MLPrediction.plant_id == plant_id
        ).order_by(MLPrediction.prediction_date).all()
        
        # Get weather forecast
        coords = weather_api.get_location_coordinates(plant.location.split(',')[0])
        weather_forecast = weather_api.get_forecast(coords[0], coords[1], 7)
        
        # Calculate summary statistics
        if all_predictions:
            total_predicted_energy = sum(p.predicted_energy for p in all_predictions)
            total_predicted_revenue = sum(p.predicted_revenue for p in all_predictions)
            avg_efficiency = sum(p.predicted_efficiency for p in all_predictions) / len(all_predictions)
            avg_confidence = sum(p.confidence_score for p in all_predictions) / len(all_predictions)
        else:
            total_predicted_energy = 0
            total_predicted_revenue = 0
            avg_efficiency = 0
            avg_confidence = 0
        
        summary = {
            'total_energy': round(total_predicted_energy, 0),
            'total_revenue': round(total_predicted_revenue, 0),
            'avg_efficiency': round(avg_efficiency, 1),
            'avg_confidence': round(avg_confidence * 100, 1)
        }
        
        # Get all plants for navigation
        all_plants = SolarPlant.query.all()
        
        return render_template('predictions.html',
                             plant=plant,
                             all_plants=all_plants,
                             predictions=all_predictions,
                             weather_forecast=weather_forecast,
                             summary=summary)
        
    except Exception as e:
        logging.error(f"Error in predictions route: {e}")
        flash('Error loading predictions', 'error')
        return redirect(url_for('index'))

@app.route('/api/train_model/<int:plant_id>', methods=['POST'])
def train_model(plant_id):
    """API endpoint to train ML model"""
    try:
        success = ml_predictor.train_models(plant_id)
        
        if success:
            # Generate new predictions
            predictions = ml_predictor.predict_6_months(plant_id)
            
            if predictions:
                return jsonify({
                    'success': True,
                    'message': 'Model trained successfully and predictions generated',
                    'predictions_count': len(predictions)
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Model trained but prediction generation failed'
                })
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed'
            })
            
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/generate_predictions/<int:plant_id>', methods=['POST'])
def generate_predictions(plant_id):
    """API endpoint to generate predictions"""
    try:
        predictions = ml_predictor.predict_6_months(plant_id)
        
        if predictions:
            return jsonify({
                'success': True,
                'message': 'Predictions generated successfully',
                'predictions_count': len(predictions)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to generate predictions'
            })
            
    except Exception as e:
        logging.error(f"Error generating predictions: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/weather/<int:plant_id>')
def get_weather(plant_id):
    """API endpoint to get current weather"""
    try:
        plant = SolarPlant.query.get_or_404(plant_id)
        coords = weather_api.get_location_coordinates(plant.location.split(',')[0])
        weather = weather_api.get_current_weather(coords[0], coords[1])
        
        return jsonify({
            'success': True,
            'weather': weather
        })
        
    except Exception as e:
        logging.error(f"Error getting weather: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/maintenance/<int:plant_id>')
def get_maintenance_schedule(plant_id):
    """API endpoint to get maintenance schedule"""
    try:
        # Get recent maintenance records
        maintenance_records = MaintenanceRecord.query.filter(
            MaintenanceRecord.plant_id == plant_id
        ).order_by(MaintenanceRecord.date.desc()).limit(10).all()
        
        # Calculate next maintenance dates (simplified)
        next_maintenance = []
        maintenance_types = ['Panel Cleaning', 'Inverter Check', 'Electrical Inspection']
        
        for i, mtype in enumerate(maintenance_types):
            next_date = date.today() + timedelta(days=30 * (i + 1))
            next_maintenance.append({
                'type': mtype,
                'date': next_date.isoformat(),
                'priority': 'Medium'
            })
        
        return jsonify({
            'success': True,
            'recent_maintenance': [
                {
                    'date': record.date.isoformat(),
                    'type': record.maintenance_type,
                    'cost': record.cost_inr,
                    'description': record.description
                } for record in maintenance_records
            ],
            'upcoming_maintenance': next_maintenance
        })
        
    except Exception as e:
        logging.error(f"Error getting maintenance schedule: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/chart_data/<int:plant_id>/<chart_type>')
def get_chart_data(plant_id, chart_type):
    """API endpoint to get chart data"""
    try:
        if chart_type == 'production':
            # Get last 30 days production data
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            data = db.session.query(
                EnergyProduction.date,
                EnergyProduction.energy_produced,
                EnergyProduction.revenue_inr
            ).filter(
                EnergyProduction.plant_id == plant_id,
                EnergyProduction.date >= start_date
            ).order_by(EnergyProduction.date).all()
            
            return jsonify({
                'success': True,
                'data': [
                    {
                        'date': record.date.isoformat(),
                        'energy': record.energy_produced,
                        'revenue': record.revenue_inr
                    } for record in data
                ]
            })
        
        elif chart_type == 'predictions':
            # Get predictions data
            predictions = MLPrediction.query.filter(
                MLPrediction.plant_id == plant_id
            ).order_by(MLPrediction.prediction_date).limit(60).all()
            
            return jsonify({
                'success': True,
                'data': [
                    {
                        'date': pred.prediction_date.isoformat(),
                        'energy': pred.predicted_energy,
                        'revenue': pred.predicted_revenue,
                        'efficiency': pred.predicted_efficiency,
                        'confidence': pred.confidence_score
                    } for pred in predictions
                ]
            })
        
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid chart type'
            })
            
    except Exception as e:
        logging.error(f"Error getting chart data: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
