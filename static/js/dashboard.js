/**
 * Solar Energy ML Predictor - Dashboard JavaScript
 * Handles dashboard interactions, API calls, and real-time updates
 */

class SolarDashboard {
    constructor() {
        this.charts = {};
        this.refreshInterval = null;
        this.weatherRefreshInterval = 300000; // 5 minutes
        this.init();
    }

    init() {
        console.log('Solar Dashboard initialized');
        this.setupEventListeners();
        this.startAutoRefresh();
        this.addAnimations();
    }

    setupEventListeners() {
        // Plant selector change
        document.addEventListener('change', (e) => {
            if (e.target.classList.contains('plant-selector')) {
                const plantId = e.target.value;
                if (plantId) {
                    this.switchPlant(plantId);
                }
            }
        });

        // Button clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('[data-action]')) {
                const action = e.target.closest('[data-action]').dataset.action;
                const plantId = e.target.closest('[data-action]').dataset.plantId;
                this.handleAction(action, plantId);
            }
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });

        // Page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.stopAutoRefresh();
            } else {
                this.startAutoRefresh();
            }
        });
    }

    handleAction(action, plantId) {
        switch (action) {
            case 'train-model':
                this.trainModel(plantId);
                break;
            case 'generate-predictions':
                this.generatePredictions(plantId);
                break;
            case 'refresh-weather':
                this.refreshWeather(plantId);
                break;
            case 'export-data':
                this.exportData(plantId);
                break;
            default:
                console.warn('Unknown action:', action);
        }
    }

    async trainModel(plantId) {
        if (!plantId) {
            this.showNotification('error', 'Plant ID is required');
            return;
        }

        try {
            this.showLoading('Training ML model with historical data...');
            
            const response = await fetch(`/api/train_model/${plantId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.showNotification('success', 
                    `Model trained successfully! Generated ${data.predictions_count || 0} predictions.`);
                
                // Refresh charts and data
                setTimeout(() => {
                    this.refreshDashboard();
                }, 2000);
            } else {
                this.showNotification('error', `Training failed: ${data.message}`);
            }
        } catch (error) {
            this.hideLoading();
            console.error('Training error:', error);
            this.showNotification('error', 'Training failed due to network error');
        }
    }

    async generatePredictions(plantId) {
        if (!plantId) {
            this.showNotification('error', 'Plant ID is required');
            return;
        }

        try {
            this.showLoading('Generating 6-month predictions...');
            
            const response = await fetch(`/api/generate_predictions/${plantId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.showNotification('success', 
                    `Generated ${data.predictions_count} predictions successfully!`);
                
                // Refresh prediction charts
                this.refreshPredictionCharts(plantId);
            } else {
                this.showNotification('error', `Prediction generation failed: ${data.message}`);
            }
        } catch (error) {
            this.hideLoading();
            console.error('Prediction error:', error);
            this.showNotification('error', 'Prediction generation failed due to network error');
        }
    }

    async refreshWeather(plantId) {
        if (!plantId) {
            this.showNotification('error', 'Plant ID is required');
            return;
        }

        try {
            this.showLoading('Refreshing weather data...');
            
            const response = await fetch(`/api/weather/${plantId}`);
            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.showNotification('success', 'Weather data refreshed successfully!');
                this.updateWeatherDisplay(data.weather);
            } else {
                this.showNotification('warning', `Weather refresh failed: ${data.message}`);
            }
        } catch (error) {
            this.hideLoading();
            console.error('Weather error:', error);
            this.showNotification('warning', 'Weather refresh failed due to network error');
        }
    }

    updateWeatherDisplay(weatherData) {
        const weatherElements = {
            temperature: document.querySelector('[data-weather="temperature"]'),
            humidity: document.querySelector('[data-weather="humidity"]'),
            windSpeed: document.querySelector('[data-weather="wind-speed"]'),
            solarIrradiance: document.querySelector('[data-weather="solar-irradiance"]'),
            cloudCover: document.querySelector('[data-weather="cloud-cover"]'),
            description: document.querySelector('[data-weather="description"]')
        };

        if (weatherData) {
            if (weatherElements.temperature) {
                weatherElements.temperature.textContent = `${weatherData.temperature.toFixed(1)}°C`;
            }
            if (weatherElements.humidity) {
                weatherElements.humidity.textContent = `${weatherData.humidity.toFixed(1)}%`;
            }
            if (weatherElements.windSpeed) {
                weatherElements.windSpeed.textContent = `${weatherData.wind_speed.toFixed(1)} km/h`;
            }
            if (weatherElements.solarIrradiance) {
                weatherElements.solarIrradiance.textContent = `${weatherData.solar_irradiance.toFixed(1)} kWh/m²`;
            }
            if (weatherElements.cloudCover) {
                weatherElements.cloudCover.textContent = `${weatherData.cloud_cover.toFixed(1)}%`;
            }
            if (weatherElements.description) {
                weatherElements.description.textContent = weatherData.description;
            }

            // Add fallback indicator if needed
            if (weatherData.fallback) {
                this.showNotification('warning', 'Using fallback weather data - API unavailable');
            }
        }
    }

    switchPlant(plantId) {
        // Redirect to dashboard with new plant ID
        window.location.href = `/dashboard/${plantId}`;
    }

    async refreshDashboard() {
        // Refresh the current page to get updated data
        window.location.reload();
    }

    async refreshPredictionCharts(plantId) {
        // This would be called to refresh prediction-specific charts
        if (typeof initializePredictionCharts === 'function') {
            initializePredictionCharts(plantId);
        }
    }

    resizeCharts() {
        // Resize all Chart.js instances
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }

    startAutoRefresh() {
        // Auto-refresh weather data every 5 minutes
        this.refreshInterval = setInterval(() => {
            const plantId = this.getCurrentPlantId();
            if (plantId && !document.hidden) {
                this.refreshWeather(plantId);
            }
        }, this.weatherRefreshInterval);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    getCurrentPlantId() {
        // Extract plant ID from URL or active plant selector
        const urlPath = window.location.pathname;
        const match = urlPath.match(/\/dashboard\/(\d+)/);
        if (match) {
            return match[1];
        }
        
        // Fallback: get from active plant selector
        const activeItem = document.querySelector('.list-group-item.active');
        if (activeItem) {
            const href = activeItem.getAttribute('href');
            const hrefMatch = href.match(/plant_id=(\d+)/);
            if (hrefMatch) {
                return hrefMatch[1];
            }
        }
        
        return null;
    }

    addAnimations() {
        // Add fade-in animation to cards
        const cards = document.querySelectorAll('.card, .stat-card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('fade-in');
        });

        // Add solar panel glow animation
        const solarIcons = document.querySelectorAll('.fa-solar-panel');
        solarIcons.forEach(icon => {
            icon.addEventListener('mouseenter', () => {
                icon.style.animation = 'solarGlow 0.5s ease-in-out';
            });
            icon.addEventListener('mouseleave', () => {
                icon.style.animation = 'solarGlow 3s ease-in-out infinite alternate';
            });
        });

        // Add energy flow animation to progress bars
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.style.background = 'linear-gradient(90deg, transparent, rgba(255, 193, 7, 0.5), transparent)';
            bar.classList.add('energy-flow');
        });
    }

    exportData(plantId) {
        // This is a placeholder for data export functionality
        this.showNotification('info', 'Export functionality coming soon');
    }

    showLoading(message = 'Loading...') {
        let modal = document.getElementById('loadingModal');
        
        if (!modal) {
            // Create loading modal if it doesn't exist
            modal = document.createElement('div');
            modal.id = 'loadingModal';
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Processing...</h5>
                        </div>
                        <div class="modal-body text-center">
                            <div class="spinner-border text-primary mb-3" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p id="loadingMessage">Please wait...</p>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        const messageElement = modal.querySelector('#loadingMessage');
        if (messageElement) {
            messageElement.textContent = message;
        }

        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    hideLoading() {
        const modal = document.getElementById('loadingModal');
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
    }

    showNotification(type, message, duration = 5000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show notification-toast`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);

        return notification;
    }

    // Utility method to format numbers
    formatNumber(num, decimals = 0) {
        if (typeof num !== 'number') return '0';
        return num.toLocaleString('en-IN', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    // Utility method to format currency
    formatCurrency(amount) {
        if (typeof amount !== 'number') return '₹0';
        return '₹' + this.formatNumber(amount, 0);
    }

    // Method to register a chart instance
    registerChart(name, chartInstance) {
        this.charts[name] = chartInstance;
    }

    // Method to get a chart instance
    getChart(name) {
        return this.charts[name];
    }

    // Method to destroy all charts
    destroyCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Global dashboard instance
let dashboardInstance = null;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    dashboardInstance = new SolarDashboard();
    
    // Make dashboard available globally for other scripts
    window.solarDashboard = dashboardInstance;
});

// Cleanup when page is unloaded
window.addEventListener('beforeunload', function() {
    if (dashboardInstance) {
        dashboardInstance.stopAutoRefresh();
        dashboardInstance.destroyCharts();
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SolarDashboard;
}
