/**
 * Solar Energy ML Predictor - Charts JavaScript
 * Handles Chart.js chart creation and management for solar energy data visualization
 */

class SolarCharts {
    constructor() {
        this.charts = {};
        this.defaultOptions = this.getDefaultOptions();
        this.solarColors = this.getSolarColorPalette();
    }

    getSolarColorPalette() {
        return {
            primary: '#007bff',
            success: '#28a745',
            warning: '#ffc107',
            info: '#17a2b8',
            solar: '#ffeb3b',
            energy: '#4caf50',
            sky: '#2196f3',
            gradient: {
                primary: ['#007bff', '#ffc107'],
                success: ['#28a745', '#ffeb3b'],
                warning: ['#ff9800', '#ffc107'],
                info: ['#007bff', '#28a745']
            }
        };
    }

    getDefaultOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#ffc107',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            
                            // Format based on data type
                            if (context.dataset.label.includes('Revenue') || context.dataset.label.includes('₹')) {
                                label += '₹' + context.parsed.y.toLocaleString('en-IN');
                            } else if (context.dataset.label.includes('Energy') || context.dataset.label.includes('kWh')) {
                                label += context.parsed.y.toLocaleString('en-IN') + ' kWh';
                            } else if (context.dataset.label.includes('Efficiency') || context.dataset.label.includes('%')) {
                                label += context.parsed.y.toFixed(1) + '%';
                            } else {
                                label += context.parsed.y.toLocaleString('en-IN');
                            }
                            
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        };
    }

    createGradient(ctx, colors, direction = 'vertical') {
        const gradient = direction === 'vertical' 
            ? ctx.createLinearGradient(0, 0, 0, 400)
            : ctx.createLinearGradient(0, 0, 400, 0);
        
        gradient.addColorStop(0, colors[0]);
        gradient.addColorStop(1, colors[1]);
        return gradient;
    }

    async initializeCharts(plantId) {
        try {
            // Destroy existing charts
            this.destroyAllCharts();

            // Load production data and create charts
            const productionData = await this.fetchChartData(plantId, 'production');
            if (productionData && productionData.length > 0) {
                this.createProductionChart(productionData);
                this.createRevenueChart(productionData);
                this.createEfficiencyChart(productionData);
            }

            // Load prediction data if on predictions page
            if (window.location.pathname.includes('predictions')) {
                const predictionData = await this.fetchChartData(plantId, 'predictions');
                if (predictionData && predictionData.length > 0) {
                    this.createPredictionCharts(predictionData);
                }
            }

        } catch (error) {
            console.error('Error initializing charts:', error);
            this.showChartError('Failed to load chart data');
        }
    }

    async fetchChartData(plantId, chartType) {
        try {
            const response = await fetch(`/api/chart_data/${plantId}/${chartType}`);
            const result = await response.json();
            
            if (result.success) {
                return result.data;
            } else {
                throw new Error(result.message || 'Failed to fetch chart data');
            }
        } catch (error) {
            console.error(`Error fetching ${chartType} data:`, error);
            return null;
        }
    }

    createProductionChart(data) {
        const ctx = document.getElementById('productionChart');
        if (!ctx) return;

        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.primary);

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Energy Production (kWh)',
                    data: data.map(d => d.energy),
                    borderColor: this.solarColors.primary,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.solarColors.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Energy (kWh)',
                            font: { weight: '600' }
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toLocaleString('en-IN') + ' kWh';
                            }
                        }
                    }
                }
            }
        });

        this.charts.production = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('production', chart);
        }
    }

    createRevenueChart(data) {
        const ctx = document.getElementById('revenueChart');
        if (!ctx) return;

        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.success);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => new Date(d.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Revenue (₹)',
                    data: data.map(d => d.revenue),
                    backgroundColor: gradient,
                    borderColor: this.solarColors.success,
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Revenue (₹)',
                            font: { weight: '600' }
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        }
                    }
                }
            }
        });

        this.charts.revenue = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('revenue', chart);
        }
    }

    createEfficiencyChart(data) {
        const ctx = document.getElementById('efficiencyChart');
        if (!ctx) return;

        // Create efficiency data by calculating from energy and max possible
        const efficiencyData = data.map(d => {
            // This is a simplified calculation - in real scenario, we'd need plant capacity
            const maxPossible = 100000; // Placeholder max kWh per day
            return Math.min(100, (d.energy / maxPossible) * 100);
        });

        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.warning);

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Equipment Efficiency (%)',
                    data: efficiencyData,
                    borderColor: this.solarColors.warning,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.solarColors.warning,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Efficiency (%)',
                            font: { weight: '600' }
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        this.charts.efficiency = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('efficiency', chart);
        }
    }

    createPredictionCharts(predictions) {
        this.createEnergyPredictionChart(predictions);
        this.createRevenuePredictionChart(predictions);
        this.createEfficiencyPredictionChart(predictions);
        this.createMonthlyForecastChart(predictions);
    }

    createEnergyPredictionChart(predictions) {
        const ctx = document.getElementById('energyPredictionChart');
        if (!ctx) return;

        const displayData = predictions.slice(0, 60); // Show 2 months
        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.primary);

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: displayData.map(p => new Date(p.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Predicted Energy (kWh)',
                    data: displayData.map(p => p.energy),
                    borderColor: this.solarColors.primary,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 4
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Energy (kWh)',
                            font: { weight: '600' }
                        }
                    }
                }
            }
        });

        this.charts.energyPrediction = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('energyPrediction', chart);
        }
    }

    createRevenuePredictionChart(predictions) {
        const ctx = document.getElementById('revenuePredictionChart');
        if (!ctx) return;

        const displayData = predictions.slice(0, 60);
        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.success);

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: displayData.map(p => new Date(p.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Predicted Revenue (₹)',
                    data: displayData.map(p => p.revenue),
                    borderColor: this.solarColors.success,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 4
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        title: {
                            display: true,
                            text: 'Revenue (₹)',
                            font: { weight: '600' }
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        }
                    }
                }
            }
        });

        this.charts.revenuePrediction = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('revenuePrediction', chart);
        }
    }

    createEfficiencyPredictionChart(predictions) {
        const ctx = document.getElementById('efficiencyPredictionChart');
        if (!ctx) return;

        const displayData = predictions.slice(0, 60);
        const gradient = this.createGradient(ctx.getContext('2d'), this.solarColors.gradient.warning);

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: displayData.map(p => new Date(p.date).toLocaleDateString('en-IN', { 
                    month: 'short', day: 'numeric' 
                })),
                datasets: [{
                    label: 'Predicted Efficiency (%)',
                    data: displayData.map(p => p.efficiency),
                    borderColor: this.solarColors.warning,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 4
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    ...this.defaultOptions.scales,
                    y: {
                        ...this.defaultOptions.scales.y,
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Efficiency (%)',
                            font: { weight: '600' }
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        this.charts.efficiencyPrediction = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('efficiencyPrediction', chart);
        }
    }

    createMonthlyForecastChart(predictions) {
        const ctx = document.getElementById('monthlyForecastChart');
        if (!ctx) return;

        // Group predictions by month
        const monthlyData = {};
        predictions.forEach(p => {
            const date = new Date(p.date);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            
            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = { energy: 0, revenue: 0, count: 0 };
            }
            
            monthlyData[monthKey].energy += p.energy || 0;
            monthlyData[monthKey].revenue += p.revenue || 0;
            monthlyData[monthKey].count++;
        });

        const months = Object.keys(monthlyData).sort().slice(0, 6); // Show 6 months
        const energyData = months.map(month => monthlyData[month].energy);
        const revenueData = months.map(month => monthlyData[month].revenue);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: months.map(m => {
                    const date = new Date(m + '-01');
                    return date.toLocaleDateString('en-IN', { year: 'numeric', month: 'short' });
                }),
                datasets: [{
                    label: 'Energy (kWh)',
                    data: energyData,
                    backgroundColor: 'rgba(0, 123, 255, 0.7)',
                    borderColor: this.solarColors.primary,
                    borderWidth: 1,
                    yAxisID: 'y'
                }, {
                    label: 'Revenue (₹)',
                    data: revenueData,
                    backgroundColor: 'rgba(40, 167, 69, 0.7)',
                    borderColor: this.solarColors.success,
                    borderWidth: 1,
                    yAxisID: 'y1'
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    x: this.defaultOptions.scales.x,
                    y: {
                        ...this.defaultOptions.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Energy (kWh)',
                            font: { weight: '600' }
                        }
                    },
                    y1: {
                        ...this.defaultOptions.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Revenue (₹)',
                            font: { weight: '600' }
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            ...this.defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        }
                    }
                }
            }
        });

        this.charts.monthlyForecast = chart;
        if (window.solarDashboard) {
            window.solarDashboard.registerChart('monthlyForecast', chart);
        }
    }

    destroyChart(chartName) {
        if (this.charts[chartName]) {
            this.charts[chartName].destroy();
            delete this.charts[chartName];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(chartName => {
            this.destroyChart(chartName);
        });
    }

    showChartError(message) {
        console.error('Chart Error:', message);
        
        // Find chart containers and show error message
        const chartContainers = document.querySelectorAll('canvas[id$="Chart"]');
        chartContainers.forEach(canvas => {
            const container = canvas.closest('.card-body');
            if (container) {
                container.innerHTML = `
                    <div class="text-center py-4">
                        <i class="fas fa-exclamation-triangle text-warning fa-2x mb-3"></i>
                        <p class="text-muted">${message}</p>
                        <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">
                            <i class="fas fa-refresh me-1"></i>Retry
                        </button>
                    </div>
                `;
            }
        });
    }

    getChart(name) {
        return this.charts[name];
    }

    getAllCharts() {
        return this.charts;
    }
}

// Global charts instance
let chartsInstance = null;

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    chartsInstance = new SolarCharts();
    
    // Make charts available globally
    window.solarCharts = chartsInstance;
});

// Global function to initialize charts (called from templates)
function initializeCharts(plantId) {
    if (chartsInstance) {
        chartsInstance.initializeCharts(plantId);
    }
}

// Global function to initialize prediction charts
function initializePredictionCharts(plantId) {
    if (chartsInstance) {
        chartsInstance.fetchChartData(plantId, 'predictions').then(data => {
            if (data && data.length > 0) {
                chartsInstance.createPredictionCharts(data);
            }
        });
    }
}

// Cleanup when page is unloaded
window.addEventListener('beforeunload', function() {
    if (chartsInstance) {
        chartsInstance.destroyAllCharts();
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SolarCharts;
}
