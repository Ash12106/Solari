/**
 * Enhanced Solar Energy Dashboard with Realistic Physics-Based Predictions
 * Provides smooth UX with loading states, notifications, and authentic data display
 */

class EnhancedSolarDashboard {
    constructor() {
        this.charts = {};
        this.notifications = [];
        this.loadingState = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Enhanced button click handlers
        document.addEventListener('click', (e) => {
            if (e.target.matches('[onclick*="trainRealisticModel"]')) {
                e.preventDefault();
                const plantId = this.extractPlantId(e.target.getAttribute('onclick'));
                this.trainRealisticModel(plantId);
            }
            
            if (e.target.matches('[onclick*="generateRealisticPredictions"]')) {
                e.preventDefault();
                const plantId = this.extractPlantId(e.target.getAttribute('onclick'));
                this.generateRealisticPredictions(plantId);
            }
        });

        // Handle window resize for responsive charts
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });
    }

    extractPlantId(onclickStr) {
        const match = onclickStr.match(/\d+/);
        return match ? parseInt(match[0]) : 1;
    }

    async trainRealisticModel(plantId) {
        this.showLoading('Training Physics-Based Model with Real Weather Data...');
        
        try {
            const response = await fetch(`/api/train_model/${plantId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                this.showNotification('success', `
                    🎯 Realistic Model Trained Successfully!
                    <br>• ${data.predictions_count} authentic predictions generated
                    <br>• Using ${data.model_type || 'Physics-Based Model'}
                    <br>• Real weather data from OpenWeather API
                `);
                
                // Refresh dashboard data
                setTimeout(() => {
                    this.refreshDashboard();
                }, 1000);
            } else {
                this.showNotification('error', `Training Failed: ${data.message}`);
            }
        } catch (error) {
            this.showNotification('error', `Network Error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async generateRealisticPredictions(plantId) {
        this.showLoading('Generating 6-Month Authentic Forecasts...');
        
        try {
            const response = await fetch(`/api/generate_predictions/${plantId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                this.showNotification('success', `
                    🌟 Realistic Predictions Generated!
                    <br>• ${data.predictions_count} authentic forecasts created
                    <br>• Based on real weather patterns for Mysuru, Karnataka
                    <br>• Physics-based solar calculations applied
                `);
                
                // Refresh charts with new data
                setTimeout(() => {
                    this.refreshCharts();
                }, 1000);
            } else {
                this.showNotification('error', `Prediction Failed: ${data.message}`);
            }
        } catch (error) {
            this.showNotification('error', `Network Error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async refreshDashboard() {
        // Reload the page to show updated data
        window.location.reload();
    }

    async refreshCharts() {
        // Refresh chart data without full page reload
        const plantId = this.getCurrentPlantId();
        
        try {
            // Update all charts with new data
            if (this.charts.production) {
                await this.updateChart('production', plantId);
            }
            if (this.charts.revenue) {
                await this.updateChart('revenue', plantId);
            }
            if (this.charts.efficiency) {
                await this.updateChart('efficiency', plantId);
            }
        } catch (error) {
            console.error('Error refreshing charts:', error);
        }
    }

    async updateChart(chartType, plantId) {
        try {
            const response = await fetch(`/api/chart_data/${plantId}/${chartType}`);
            const data = await response.json();
            
            if (data.success && this.charts[chartType]) {
                // Update chart data
                this.charts[chartType].data = data.chartData;
                this.charts[chartType].update('active');
            }
        } catch (error) {
            console.error(`Error updating ${chartType} chart:`, error);
        }
    }

    getCurrentPlantId() {
        // Extract plant ID from current URL or default to 1
        const pathParts = window.location.pathname.split('/');
        const dashboardIndex = pathParts.indexOf('dashboard');
        if (dashboardIndex !== -1 && pathParts[dashboardIndex + 1]) {
            return parseInt(pathParts[dashboardIndex + 1]);
        }
        return 1;
    }

    initializeCharts() {
        // Initialize enhanced chart configurations
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            family: 'Inter',
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#4f46e5',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: {
                        family: 'Inter',
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        family: 'Inter',
                        size: 12
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        borderColor: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        borderColor: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                }
            }
        };

        // Apply enhanced options to existing charts
        Chart.defaults.font.family = 'Inter';
        Chart.defaults.color = '#374151';
        
        // Store chart instances for updates
        const chartElements = ['productionChart', 'revenueChart', 'efficiencyChart'];
        chartElements.forEach(elementId => {
            const element = document.getElementById(elementId);
            if (element) {
                // Chart will be initialized by existing chart.js system
                // Store reference when created
                setTimeout(() => {
                    const chart = Chart.getChart(element);
                    if (chart) {
                        const chartName = elementId.replace('Chart', '');
                        this.charts[chartName] = chart;
                        
                        // Apply enhanced styling
                        chart.options = { ...chart.options, ...chartOptions };
                        chart.update();
                    }
                }, 1000);
            }
        });
    }

    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }

    startAutoRefresh() {
        // Auto-refresh every 5 minutes
        setInterval(() => {
            if (!this.loadingState) {
                this.refreshCharts();
            }
        }, 300000); // 5 minutes
    }

    showLoading(message = 'Processing...') {
        this.loadingState = true;
        
        // Remove existing loading overlay
        this.hideLoading();
        
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <h5 class="mb-2">${message}</h5>
                <p class="text-muted mb-0">Please wait while we process your request...</p>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }

    hideLoading() {
        this.loadingState = false;
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    showNotification(type, message, duration = 8000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>${message}</div>
                <button type="button" class="btn-close btn-close-white ms-2" onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
        
        // Store notification reference
        this.notifications.push(notification);
        
        // Limit to 3 notifications max
        if (this.notifications.length > 3) {
            const oldNotification = this.notifications.shift();
            if (oldNotification.parentElement) {
                oldNotification.remove();
            }
        }
    }

    formatNumber(num, decimals = 2) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return parseFloat(num).toFixed(decimals);
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }
}

// Legacy function support for existing onclick handlers
function trainRealisticModel(plantId) {
    if (window.enhancedDashboard) {
        window.enhancedDashboard.trainRealisticModel(plantId);
    }
}

function generateRealisticPredictions(plantId) {
    if (window.enhancedDashboard) {
        window.enhancedDashboard.generateRealisticPredictions(plantId);
    }
}

// Initialize enhanced dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.enhancedDashboard = new EnhancedSolarDashboard();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedSolarDashboard;
}