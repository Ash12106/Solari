# VVCE Solar Plant ML Prediction System

## Overview

This is a comprehensive machine learning-powered web application for monitoring, predicting, and optimizing solar power plants at Vidhyavardhaka College of Engineering (VVCE) in Mysuru, Karnataka. The system provides 6-month forecasting, intelligent maintenance recommendations, and real-time performance analytics using Indian standards and currency (INR).

## System Architecture

### Backend Architecture
- **Framework**: Flask web application with SQLAlchemy ORM
- **Database**: SQLite for development (configured to support PostgreSQL via environment variables)
- **ML Framework**: Scikit-learn with XGBoost for advanced predictions
- **API Integration**: OpenWeather API for real-time weather data
- **Session Management**: Flask sessions with configurable secret key

### Frontend Architecture
- **Template Engine**: Jinja2 with Flask
- **CSS Framework**: Bootstrap 5.3.0
- **JavaScript Libraries**: Chart.js for data visualization, Font Awesome for icons
- **Responsive Design**: Mobile-first approach with fluid containers

### Data Architecture
- **Relational Model**: Normalized database structure with foreign key relationships
- **Time Series Data**: Daily weather and energy production records
- **ML Predictions**: Stored predictions with performance metrics tracking

## Key Components

### Core Models
1. **SolarPlant**: Plant configuration and specifications
2. **WeatherData**: Daily weather metrics (temperature, humidity, cloud cover, solar irradiance)
3. **EnergyProduction**: Daily energy output and financial data in INR
4. **MaintenanceRecord**: Maintenance logs with cost tracking
5. **MLPrediction**: ML model predictions and confidence intervals
6. **ModelPerformance**: ML model accuracy metrics

### ML Engine (`ml_models.py`)
- **SolarMLPredictor**: Multi-model prediction system
- **Models**: Random Forest and XGBoost regressors
- **Features**: Weather data, equipment efficiency, plant capacity, seasonality
- **Preprocessing**: StandardScaler for feature normalization
- **Model Persistence**: Joblib for model serialization

### Analytics Engine (`weekly_analytics.py`)
- **VVCEWeeklyAnalytics**: Advanced 26-week prediction system
- **Performance Classification**: Excellent/Good/Average/Poor thresholds
- **Quarterly Summaries**: Aggregated insights for academic planning

### Maintenance Advisor (`maintenance_advisor.py`)
- **VVCECleaningAdvisor**: Intelligent cleaning recommendations
- **Climate Factors**: Mysuru-specific weather patterns (monsoon, dry season, dust storms)
- **Decision Logic**: Multi-factor analysis for maintenance scheduling

### Weather Integration (`weather_api.py`)
- **WeatherAPI**: OpenWeather API integration with fallback data
- **Current Weather**: Real-time weather fetching
- **Forecasting**: 7-day weather predictions
- **Error Handling**: Graceful degradation to synthetic data

## Data Flow

1. **Data Collection**: Weather API fetches current conditions and forecasts
2. **Data Storage**: Daily weather and energy production stored in database
3. **ML Pipeline**: Historical data preprocessed and fed to trained models
4. **Prediction Generation**: 6-month forecasts created using ensemble methods
5. **Analytics Processing**: Weekly and quarterly summaries generated
6. **Dashboard Rendering**: Real-time data visualization with Chart.js
7. **Maintenance Alerts**: Intelligent recommendations based on performance degradation

## External Dependencies

### Python Packages
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting framework
- **NumPy/Pandas**: Data manipulation
- **Requests**: HTTP client for weather API

### Frontend Libraries
- **Bootstrap 5.3.0**: UI framework
- **Chart.js**: Data visualization
- **Font Awesome 6.4.0**: Icon library

### External APIs
- **OpenWeather API**: Real-time weather data (requires API key)

## Deployment Strategy

### Environment Configuration
- **Database**: Configurable via `DATABASE_URL` environment variable
- **Session Security**: `SESSION_SECRET` for production security
- **API Keys**: `OPENWEATHER_API_KEY` for weather integration
- **Development**: SQLite with auto-initialization
- **Production**: PostgreSQL support built-in

### Application Structure
- **Entry Point**: `app.py` with Flask application factory
- **Database Initialization**: Auto-creation of tables and sample data
- **Error Handling**: Custom 404/500 error pages
- **Static Assets**: Organized CSS/JS with proper caching headers

### Scalability Considerations
- **Connection Pooling**: SQLAlchemy engine options for production
- **Proxy Support**: ProxyFix middleware for reverse proxy deployment
- **Model Persistence**: Saved models directory for ML model storage

## Changelog
- June 30, 2025: Initial setup
- June 30, 2025: Successfully migrated from Replit Agent to standard Replit environment
- June 30, 2025: Enhanced security configuration, removed hardcoded fallback keys
- June 30, 2025: Fixed ML prediction system errors and stabilized core functionality
- June 30, 2025: Prepared advanced LSTM architecture for future TensorFlow integration

## User Preferences

Preferred communication style: Simple, everyday language.
Project focus: 6-month solar forecasting with LSTM-based deep learning models for VVCE solar plants.