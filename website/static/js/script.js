/*
Aircraft Predictive Maintenance Website JavaScript
*/

$(document).ready(function() {
    // Initialize tooltips
    $('[data-bs-toggle="tooltip"]').tooltip();
    
    // Initialize popovers
    $('[data-bs-toggle="popover"]').popover();
    
    // Form validation enhancement
    $('form').on('submit', function(e) {
        let form = $(this);
        let isValid = true;
        
        // Highlight invalid fields
        form.find('.form-control').each(function() {
            if ($(this).is(':invalid')) {
                $(this).addClass('is-invalid');
                isValid = false;
            } else {
                $(this).removeClass('is-invalid');
            }
        });
        
        if (!isValid) {
            e.preventDefault();
            showToast('Please fill all required fields correctly.', 'warning');
        }
    });
    
    // Real-time input validation
    $('.form-control').on('blur', function() {
        validateField($(this));
    });
    
    // Range input display
    $('input[type="range"]').each(function() {
        let id = $(this).attr('id');
        let valueDisplay = $(`#${id}-value`);
        
        $(this).on('input', function() {
            valueDisplay.text($(this).val());
            validateRange($(this));
        });
    });
    
    // Smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(e) {
        if ($(this).attr('href') !== '#') {
            e.preventDefault();
            let target = $($(this).attr('href'));
            if (target.length) {
                $('html, body').animate({
                    scrollTop: target.offset().top - 80
                }, 500);
            }
        }
    });
    
    // Auto-refresh session warning
    let sessionWarningShown = false;
    
    function checkSession() {
        if (!sessionWarningShown) {
            // Show warning after 25 minutes of inactivity
            setTimeout(function() {
                if (document.hasFocus()) {
                    showToast('Your session will expire in 5 minutes. Please save your work.', 'warning');
                    sessionWarningShown = true;
                }
            }, 25 * 60 * 1000); // 25 minutes
        }
    }
    
    // Check session on user interaction
    $(document).on('click keypress scroll', function() {
        checkSession();
    });
    
    // Prediction form specific handlers
    $('.parameter-card input').on('change', function() {
        updateParameterStatus($(this));
    });
    
    // Export prediction data
    $('#export-btn').on('click', function() {
        exportPredictionData();
    });
    
    // Initialize
    initPage();
});

// Function to validate individual field
function validateField(field) {
    let value = field.val();
    let min = field.attr('min');
    let max = field.attr('max');
    let pattern = field.attr('pattern');
    
    if (field.is(':required') && !value.trim()) {
        showFieldError(field, 'This field is required');
        return false;
    }
    
    if (min && value < min) {
        showFieldError(field, `Value must be at least ${min}`);
        return false;
    }
    
    if (max && value > max) {
        showFieldError(field, `Value must be at most ${max}`);
        return false;
    }
    
    if (pattern) {
        let regex = new RegExp(pattern);
        if (!regex.test(value)) {
            showFieldError(field, 'Invalid format');
            return false;
        }
    }
    
    clearFieldError(field);
    return true;
}

// Function to validate range inputs
function validateRange(input) {
    let value = parseFloat(input.val());
    let min = parseFloat(input.attr('min'));
    let max = parseFloat(input.attr('max'));
    let normalMin = parseFloat(input.data('normal-min'));
    let normalMax = parseFloat(input.data('normal-max'));
    
    let indicator = $(`#${input.attr('id')}-indicator`);
    
    if (value < normalMin || value > normalMax) {
        indicator.removeClass('bg-success').addClass('bg-danger');
        showToast(`${input.attr('name')} is outside normal range`, 'warning');
    } else {
        indicator.removeClass('bg-danger').addClass('bg-success');
    }
}

// Function to update parameter status
function updateParameterStatus(input) {
    let card = input.closest('.parameter-card');
    let value = parseFloat(input.val());
    let normalMin = parseFloat(input.data('normal-min'));
    let normalMax = parseFloat(input.data('normal-max'));
    
    if (isNaN(value)) return;
    
    if (value < normalMin || value > normalMax) {
        card.addClass('parameter-warning');
        card.find('.parameter-header h5').addClass('text-danger');
    } else {
        card.removeClass('parameter-warning');
        card.find('.parameter-header h5').removeClass('text-danger');
    }
}

// Function to show field error
function showFieldError(field, message) {
    field.addClass('is-invalid');
    let errorDiv = field.next('.invalid-feedback');
    if (errorDiv.length === 0) {
        errorDiv = $('<div class="invalid-feedback"></div>').insertAfter(field);
    }
    errorDiv.text(message);
}

// Function to clear field error
function clearFieldError(field) {
    field.removeClass('is-invalid');
    field.next('.invalid-feedback').remove();
}

// Toast notification function
function showToast(message, type = 'info') {
    let toast = $(`
        <div class="toast align-items-center text-white bg-${type} border-0 position-fixed"
             style="top: 20px; right: 20px; z-index: 9999;" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `);
    
    $('body').append(toast);
    let bsToast = new bootstrap.Toast(toast[0]);
    bsToast.show();
    
    toast.on('hidden.bs.toast', function() {
        $(this).remove();
    });
}

// Export prediction data as JSON
function exportPredictionData() {
    let predictionId = $('meta[name="prediction-id"]').attr('content');
    
    if (!predictionId) {
        showToast('No prediction data available to export', 'warning');
        return;
    }
    
    $.ajax({
        url: `/api/prediction/${predictionId}/export`,
        method: 'GET',
        success: function(data) {
            let blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `prediction-${predictionId}-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('Prediction data exported successfully', 'success');
        },
        error: function() {
            showToast('Failed to export prediction data', 'danger');
        }
    });
}

// Initialize page-specific functionality
function initPage() {
    // Home page
    if ($('.model-card').length) {
        initModelCards();
    }
    
    // Survey page
    if ($('.parameter-card').length) {
        initParameterCards();
    }
    
    // Prediction page
    if ($('.confidence-badge').length) {
        initPredictionPage();
    }
}

// Initialize model cards
function initModelCards() {
    $('.model-card').each(function() {
        $(this).hover(
            function() {
                $(this).find('img').css('transform', 'scale(1.05)');
                $(this).find('.btn').addClass('btn-hover');
            },
            function() {
                $(this).find('img').css('transform', 'scale(1)');
                $(this).find('.btn').removeClass('btn-hover');
            }
        );
    });
}

// Initialize parameter cards
function initParameterCards() {
    $('.parameter-card input').each(function() {
        updateParameterStatus($(this));
    });
}

// Initialize prediction page
function initPredictionPage() {
    // Animate confidence score
    let confidence = parseFloat($('.confidence-badge .badge').text().match(/(\d+\.?\d*)/)[0]);
    let color = $('.confidence-badge .badge').hasClass('bg-success') ? 'success' : 
                $('.confidence-badge .badge').hasClass('bg-warning') ? 'warning' : 'danger';
    
    // Create confidence meter
    let meter = $(`
        <div class="confidence-meter mt-3">
            <div class="meter-background"></div>
            <div class="meter-fill bg-${color}"></div>
            <div class="meter-labels d-flex justify-content-between mt-2">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
        </div>
    `);
    
    $('.confidence-badge').after(meter);
    
    // Animate the meter fill
    setTimeout(function() {
        $('.meter-fill').css('width', confidence + '%');
    }, 500);
    
    // Initialize print functionality
    initPrintFunctionality();
}

// Initialize print functionality
function initPrintFunctionality() {
    let printBtn = $('button:contains("Print Report")');
    if (printBtn.length) {
        printBtn.click(function() {
            // Add print-specific classes
            $('body').addClass('printing');
            
            // Wait a moment for CSS to apply, then print
            setTimeout(function() {
                window.print();
                
                // Remove print classes after printing
                setTimeout(function() {
                    $('body').removeClass('printing');
                }, 1000);
            }, 100);
        });
    }
}

// API call for real-time validation
function validateParameter(parameterName, value) {
    return $.ajax({
        url: '/api/validate-parameter',
        method: 'POST',
        data: JSON.stringify({
            parameter: parameterName,
            value: value
        }),
        contentType: 'application/json'
    });
}

// Debounce function for input validation
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for scroll events
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Page visibility API
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden
        console.log('Page is hidden');
    } else {
        // Page is visible
        console.log('Page is visible');
    }
});

// Service Worker Registration (if needed)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/service-worker.js').then(
            function(registration) {
                console.log('ServiceWorker registration successful');
            },
            function(err) {
                console.log('ServiceWorker registration failed: ', err);
            }
        );
    });
}