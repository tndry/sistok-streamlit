import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gdown
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats




def darken_color(hex_color, amount=0.2):
    """Menggelapkan warna heksadesimal."""
    try:
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(max(0, int(c * (1 - amount))) for c in rgb)
        return f"#{''.join([f'{c:02x}' for c in darker_rgb])}"
    except Exception:
        return "#6c757d" # Fallback color

def calculate_evaluation_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
                    
    # Basic validation
    if len(y_true) < 2:
        return {metric: np.nan for metric in ['R2', 'RMSE', 'MAE', 'MAPE']}
                    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res < 1e-9 else 0.0)
                
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
                
    # MAPE (Mean Absolute Percentage Error)
    # Ensure y_true does not contain zero for MAPE calculation
    y_true_safe = np.where(y_true == 0, 1e-9, y_true) # Replace 0 with a small number to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100 if np.all(y_true != 0) else np.nan # Original check kept for semantic correctness
    if np.any(y_true == 0): # If original y_true had zeros, MAPE is problematic
        mape_adjusted = np.mean(np.abs((y_true - y_pred) / y_true_safe[y_true_safe!=0])) * 100
        mape = mape_adjusted # Or decide to return np.nan or a warning

    return {
        'R2': r2 if pd.notnull(r2) else np.nan,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def perform_residual_analysis(y_true, y_pred, x_values, model_name):
    """Perform comprehensive residual analysis"""
    residuals = np.array(y_true) - np.array(y_pred)
    
    if len(residuals) == 0 or np.std(residuals) == 0: # Handle empty or constant residuals
        standardized_residuals = residuals # Or np.nan if preferred
    else:
        standardized_residuals = residuals / np.std(residuals)
    
    # Normality test (Shapiro-Wilk)
    normality_stat, normality_p = (stats.shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan))
    
    # Durbin-Watson test for autocorrelation (simplified)
    dw_stat = (np.sum(np.diff(residuals)**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 and len(residuals) > 1 else np.nan)
    
    # Heteroscedasticity check (correlation between |residuals| and fitted values)
    # Ensure predictions are also passed or handled if x_values is not appropriate
    hetero_corr, hetero_p = (stats.pearsonr(np.abs(residuals), np.array(y_pred)) if len(residuals) >= 3 else (np.nan, np.nan))
    
    return {
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'normality_stat': normality_stat,
        'normality_p': normality_p,
        'dw_statistic': dw_stat,
        'hetero_correlation': hetero_corr,
        'hetero_p': hetero_p
    }

            
def calculate_schaefer_enhanced(data_input):
    """Enhanced Schaefer model calculation with comprehensive metrics"""
    data = data_input.copy()
    
    # Validation checks
    if 'effort (hari)' not in data.columns or 'CPUE' not in data.columns:
        return {'name': 'Schaefer', 'error': 'Kolom effort (hari) atau CPUE tidak ditemukan.', 'status': 'error'}
    if len(data) < 3:
        return {'name': 'Schaefer', 'error': 'Data kurang dari 3 tahun', 'status': 'error'}
    
    if data['CPUE'].isnull().any() or np.isinf(data['CPUE']).any() or np.any(data['CPUE'] < 0):
        return {'name': 'Schaefer', 'error': 'CPUE mengandung nilai invalid (null, infinity, atau negatif)', 'status': 'error'}
    if data['effort (hari)'].isnull().any() or np.isinf(data['effort (hari)']).any() or np.any(data['effort (hari)'] < 0):
        return {'name': 'Schaefer', 'error': 'Effort mengandung nilai invalid (null, infinity, atau negatif)', 'status': 'error'}


    try:
        X = data['effort (hari)'].values.reshape(-1, 1)
        Y = data['CPUE'].values
        
        model = LinearRegression()
        model.fit(X, Y)
        
        a_param = model.intercept_  # intercept
        b_param = model.coef_[0]    # slope
        Y_pred = model.predict(X)
        
        # Calculate all evaluation metrics
        metrics = calculate_evaluation_metrics(Y, Y_pred)
        
        # Residual analysis
        residual_analysis = perform_residual_analysis(Y, Y_pred, X.flatten(), 'Schaefer')
        
        # Biological parameters
        eopt_val, cmsy_val = np.nan, np.nan
        status = 'warning' # Default to warning if biological params are not ideal
        
        if b_param < -1e-9:  # Valid biological model only if slope is sufficiently negative
            a_s = a_param
            b_s = -b_param  # Make positive for calculations
            if b_s > 1e-9 and a_s > 0: # ensure a_s is positive for meaningful Eopt/CMSY
                eopt_val = a_s / (2 * b_s)
                cmsy_val = (a_s**2) / (4 * b_s)
                if eopt_val > 0 and cmsy_val > 0:
                    status = 'success'
                else: # Eopt or CMSY is not positive
                    status = 'warning' 
                    eopt_val, cmsy_val = np.nan, np.nan # Invalidate if not positive
            else: # b_s too small or a_s not positive
                status = 'warning'
                eopt_val, cmsy_val = np.nan, np.nan
        else: # b_param is not negative enough or positive
            status = 'warning' # Indicates model parameters are not biologically sound for this model type
            # For Schaefer, a positive b or b close to zero is problematic.
            # If a_param is also non-positive, this is further indication of issues.
            if a_param <=0 and b_param >= -1e-9 :
                status = 'error' # e.g. CPUE increases with effort, or is flat and non-positive intercept.
            eopt_val, cmsy_val = np.nan, np.nan


        return {
            'name': 'Schaefer',
            'status': status,
            'parameters': {'a': a_param, 'b': b_param},
            'biological': {'Eopt': eopt_val, 'CMSY': cmsy_val},
            'metrics': metrics,
            'residuals': residual_analysis,
            'predictions': Y_pred,
            'actual': Y,
            'effort': X.flatten()
        }
                  
    except Exception as e:
        return {'name': 'Schaefer', 'error': f"Kesalahan kalkulasi: {str(e)}", 'status': 'error'}

def calculate_fox_enhanced(data_input):
    """Enhanced Fox model calculation with comprehensive metrics"""
    data = data_input.copy()
    
    # Validation checks
    if 'effort (hari)' not in data.columns or 'CPUE' not in data.columns:
        return {'name': 'Fox', 'error': 'Kolom effort (hari) atau CPUE tidak ditemukan.', 'status': 'error'}
    if len(data) < 3:
        return {'name': 'Fox', 'error': 'Data kurang dari 3 tahun', 'status': 'error'}
    
    if np.any(data['CPUE'] <= 1e-9): # CPUE must be positive for log
        return {'name': 'Fox', 'error': 'CPUE harus positif untuk transformasi logaritma (nilai <= 0.000000001 ditemukan)', 'status': 'error'}
    
    if data['CPUE'].isnull().any() or np.isinf(data['CPUE']).any():
        return {'name': 'Fox', 'error': 'CPUE mengandung nilai invalid (null atau infinity)', 'status': 'error'}
    if data['effort (hari)'].isnull().any() or np.isinf(data['effort (hari)']).any() or np.any(data['effort (hari)'] < 0): # effort also checked
        return {'name': 'Fox', 'error': 'Effort mengandung nilai invalid (null, infinity, atau negatif)', 'status': 'error'}

    try:
        X = data['effort (hari)'].values.reshape(-1, 1)
        # Log transform Y (CPUE), ensure CPUE > 0 already checked
        Y_log = np.log(data['CPUE'].values) 
        
        model = LinearRegression()
        model.fit(X, Y_log)
        
        c_param = model.intercept_  # intercept (ln(a) in some notations)
        d_param = model.coef_[0]    # slope (-b in some notations)
        Y_log_pred = model.predict(X)
        
        # Calculate all evaluation metrics (on log scale)
        metrics_log = calculate_evaluation_metrics(Y_log, Y_log_pred)
        
        # Also calculate metrics on original scale
        Y_orig = data['CPUE'].values
        Y_pred_orig = np.exp(Y_log_pred)
        metrics_orig = calculate_evaluation_metrics(Y_orig, Y_pred_orig)
        
        # Residual analysis (on log scale, as regression was performed on log-transformed data)
        residual_analysis = perform_residual_analysis(Y_log, Y_log_pred, X.flatten(), 'Fox (Log Scale)')
        
        # Biological parameters
        eopt_val, cmsy_val = np.nan, np.nan
        status = 'warning' # Default to warning
        
        if d_param < -1e-9:  # Valid biological model if slope is sufficiently negative
            # c_param is ln(qK) or ln(U_inf) depending on formulation.
            # Here, it's the intercept of ln(CPUE) = c + d*Effort
            # U_inf = exp(c) when d -> 0, which is qB_inf
            if np.exp(c_param) > 0: # Check if derived U_inf or equivalent is positive
                eopt_val = -1 / d_param
                cmsy_val = eopt_val * np.exp(c_param - 1) # Or -(1/d_param) * exp(c_param) * exp(-1)
                if eopt_val > 0 and cmsy_val > 0:
                    status = 'success'
                else: # Eopt or CMSY is not positive
                    status = 'warning'
                    eopt_val, cmsy_val = np.nan, np.nan # Invalidate
            else: # exp(c_param) is not positive (highly unlikely with real numbers unless c_param is -inf)
                status = 'warning'
                eopt_val, cmsy_val = np.nan, np.nan
        else: # d_param is not negative enough or positive
            status = 'warning' #Slope implies CPUE doesn't decrease or increases with effort
            if c_param <= 0 and d_param >= -1e-9: # e.g. ln(CPUE) is negative and flat or increasing
                status = 'error'
            eopt_val, cmsy_val = np.nan, np.nan
                            
        return {
            'name': 'Fox',
            'status': status,
            'parameters': {'c': c_param, 'd': d_param},
            'biological': {'Eopt': eopt_val, 'CMSY': cmsy_val},
            'metrics': metrics_log,  # Log scale metrics
            'metrics_original': metrics_orig,  # Original scale metrics
            'residuals': residual_analysis, # Residuals from log-transformed regression
            'predictions': Y_log_pred,  # Log scale predictions
            'predictions_original': Y_pred_orig,  # Original scale predictions
            'actual': Y_log,  # Log scale actual
            'actual_original': Y_orig,  # Original scale actual
            'effort': X.flatten()
        }
                        
    except Exception as e:
        return {'name': 'Fox', 'error': f"Kesalahan kalkulasi: {str(e)}", 'status': 'error'}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def _comparison_table(schaefer_results, fox_results):
    """Create an elegant comparison table for both models"""
    
    def get_status_indicator(status):
        if status == 'success':
            return "✅"
        elif status == 'warning':
            return "⚠️"
        else:
            return "❌"
    
    def format_val(value, precision):
        if pd.notnull(value) and not np.isinf(value):
            return f"{value:.{precision}f}"
        return "-"
    
    s_status = schaefer_results.get('status', 'error')
    f_status = fox_results.get('status', 'error')

    s_params = schaefer_results.get('parameters') or {}
    s_bio = schaefer_results.get('biological') or {}
    s_metrics = schaefer_results.get('metrics') or {}

    f_params = fox_results.get('parameters') or {}
    f_bio = fox_results.get('biological') or {}
    f_metrics_display = fox_results.get('metrics') or {}

    comparison_df = pd.DataFrame({
        'Parameter': [
            'Status Model',
            'Intercept (Schaefer: a, Fox: c=ln(Uinf))',
            'Slope (Schaefer: b, Fox: d)',
            'E optimal (hari)',
            'MSY/CMSY (ton)',
            'R² (Schaefer, Fox)',
            'RMSE (Schaefer, Fox)',
            'MAE (Schaefer, Fox)',
            'MAPE (%) (Schaefer, Fox)'
        ],
        'Schaefer': [
            get_status_indicator(s_status),
            format_val(s_params.get('a'), 4),
            format_val(s_params.get('b'), 6),
            format_val(s_bio.get('Eopt'), 2),
            format_val(s_bio.get('CMSY'), 2),
            format_val(s_metrics.get('R2'), 4),
            format_val(s_metrics.get('RMSE'), 4),
            format_val(s_metrics.get('MAE'), 4),
            format_val(s_metrics.get('MAPE'), 2)
        ],
        'Fox': [
            get_status_indicator(f_status),
            format_val(f_params.get('c'), 4),
            format_val(f_params.get('d'), 6),
            format_val(f_bio.get('Eopt'), 2),
            format_val(f_bio.get('CMSY'), 2),
            format_val(f_metrics_display.get('R2'), 4),
            format_val(f_metrics_display.get('RMSE'), 4),
            format_val(f_metrics_display.get('MAE'), 4),
            format_val(f_metrics_display.get('MAPE'), 2)
        ]
    })
    
    return comparison_df



def create_residual_plots(model_results, model_name):
    """Create comprehensive residual analysis plots"""
    if model_results.get('status') == 'error' or 'residuals' not in model_results:
        st.warning(f"Analisis residual tidak tersedia untuk {model_name} karena error atau data residual tidak ada.")
        return None
    
    residuals_data = model_results['residuals']
    effort = model_results.get('effort') # Ensure effort is available
    
    # Determine which predictions and actual values to use for plotting residuals
    # For Fox, residuals are usually analyzed on the log-transformed scale (where regression happened)
    if model_name == 'Fox' and 'actual' in model_results and 'predictions' in model_results:
        # Use log-scale actuals and predictions for residual plots as regression was on log-scale
        actual_for_residuals = model_results['actual'] # Y_log
        predictions_for_residuals = model_results['predictions'] # Y_log_pred
        plot_title_suffix = " (Skala Log)"
        y_axis_label = "Residual (Skala Log)"
    else: # Schaefer or if Fox original scale residuals were explicitly generated (not the case here)
        actual_for_residuals = model_results['actual']
        predictions_for_residuals = model_results['predictions']
        plot_title_suffix = ""
        y_axis_label = "Residual"

    # Residuals themselves are from the scale of regression
    residuals = residuals_data.get('residuals')
    std_residuals = residuals_data.get('standardized_residual')
    if residuals is None or len(residuals) == 0 :
        st.warning(f"Tidak ada data residual untuk {model_name}.")
        return None
    if predictions_for_residuals is None or len(predictions_for_residuals) != len(residuals):
        st.warning(f"Data prediksi tidak sesuai untuk plot residual {model_name}.")
        return None
    if effort is None or len(effort) != len(residuals):
        st.warning(f"Data effort tidak sesuai untuk plot residual {model_name}.")
        # Fallback for effort plot if effort data is mismatched
        effort_for_plot = np.arange(len(residuals)) # Use index if actual effort is problematic
        effort_axis_label = "Indeks Data"
    else:
        effort_for_plot = effort
        effort_axis_label = "Effort (hari)"


    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Residuals vs Fitted Values{plot_title_suffix}',
            f'Q-Q Plot (Normality Check){plot_title_suffix}',
            f'Residuals vs Effort{plot_title_suffix}',
            f'Histogram of Residuals{plot_title_suffix}'
        )
    )
    
    # 1. Residuals vs Fitted Values
    fig.add_trace(
        go.Scatter(
            x=predictions_for_residuals, y=residuals,
            mode='markers', name='Residuals',
            marker=dict(color='blue', size=8, opacity=0.7)
        ), row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Q-Q Plot (using standardized residuals if available and appropriate, otherwise raw)
    # Standardized residuals are better for Q-Q if scale varies, but raw can also be used.
    # Using raw residuals for consistency with histogram, shapiro test.
    # For Q-Q plot against normal distribution, it's common to use standardized residuals.
    # If std_residuals are not well-defined (e.g. std_dev was 0), fall back.
    residuals_for_qq = std_residuals if std_residuals is not None and len(std_residuals) == len(residuals) else residuals
    if len(residuals_for_qq) >=3 :
        sorted_residuals_qq = np.sort(residuals_for_qq)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals_qq)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=sorted_residuals_qq,
                mode='markers', name='Q-Q Plot',
                marker=dict(color='green', size=8, opacity=0.7)
            ), row=1, col=2
        )
        min_val = min(theoretical_quantiles.min(), sorted_residuals_qq.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals_qq.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Reference Line',
                line=dict(color='red', dash='dash'), showlegend=False
            ), row=1, col=2
        )
    else:
        fig.add_annotation(text="Data tidak cukup untuk Q-Q plot", row=1, col=2, showarrow=False)

                    
    # 3. Residuals vs Effort
    fig.add_trace(
        go.Scatter(
            x=effort_for_plot, y=residuals,
            mode='markers', name='Residuals vs Effort',
            marker=dict(color='orange', size=8, opacity=0.7)
        ), row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    # 4. Histogram of Residuals
    fig.add_trace(
        go.Histogram(
            x=residuals, name='Residual Distribution',
            marker=dict(color='purple', opacity=0.7), nbinsx=10 if len(residuals) > 10 else 5
        ), row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Analisis Residual - Model {model_name}",
        showlegend=False, height=700 # Increased height for better readability
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=f"Nilai Prediksi{plot_title_suffix}", row=1, col=1)
    fig.update_yaxes(title_text=y_axis_label, row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles (Residuals)", row=1, col=2)
    fig.update_xaxes(title_text=effort_axis_label, row=2, col=1)
    fig.update_yaxes(title_text=y_axis_label, row=2, col=1)
    fig.update_xaxes(title_text=y_axis_label, row=2, col=2) # X-axis is residual value for histogram
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
                    
    return fig

def _visualization(model_results, yearly_data, model_name):
    """Create enhanced model visualization with prediction intervals"""
    if model_results.get('status') == 'error':
        st.error(f"Visualisasi tidak dapat dibuat untuk {model_name} karena error pada model.")
        return None, None
    
    # Get model parameters
    bio_params = model_results.get('biological', {})
    eopt = bio_params.get('Eopt')
    cmsy = bio_params.get('CMSY')
    
    # Ensure Eopt and CMSY are valid numerical values for plotting
    if not (pd.notnull(eopt) and pd.notnull(cmsy) and np.isfinite(eopt) and np.isfinite(cmsy) and eopt > 0 and cmsy > 0):
        st.warning(f"Parameter biologis (Eopt/CMSY) tidak valid untuk visualisasi Model {model_name}.")
        return None, None
    
    # Create effort range for plotting
    max_actual_effort = yearly_data['effort (hari)'].max()
    # Ensure effort_max_plot is well above eopt and max_actual_effort
    effort_max_plot = max(max_actual_effort * 1.5, eopt * 2 if pd.notnull(eopt) else max_actual_effort * 1.5, 100) # Ensure positive range
    if effort_max_plot <=0 : effort_max_plot = 100 # Default if calculated is non-positive
    effort_range = np.linspace(0, effort_max_plot, 200)
                    
    # Calculate predictions based on model type
    catch_pred, cpue_pred = None, None

    if model_name == 'Schaefer':
        params = model_results.get('parameters', {})
        a, b = params.get('a'), params.get('b')
        if pd.notnull(a) and pd.notnull(b) and b < 0: # b must be negative for Schaefer
            b_pos = -b # Use positive b for formula Y = aE - bE^2
            catch_pred = a * effort_range - b_pos * (effort_range**2)
            cpue_pred = a + b * effort_range # CPUE = a + bE (b is negative)
        else:
            st.warning(f"Parameter (a atau b) tidak valid untuk visualisasi Schaefer.")
            return None, None
            
    elif model_name == 'Fox':
        params = model_results.get('parameters', {})
        c, d = params.get('c'), params.get('d')
        if pd.notnull(c) and pd.notnull(d) and d < 0: # d must be negative for Fox
            # Catch = Effort * CPUE = Effort * exp(c + d*Effort)
            cpue_pred = np.exp(c + d * effort_range)
            catch_pred = effort_range * cpue_pred
        else:
            st.warning(f"Parameter (c atau d) tidak valid untuk visualisasi Fox.")
            return None, None
    else:
        st.error(f"Tipe model tidak dikenal: {model_name}")
        return None, None

    # Ensure non-negative catches and CPUE
    catch_pred = np.maximum(0, catch_pred)
    cpue_pred = np.maximum(0, cpue_pred)
    if len(catch_pred) > 0: catch_pred[0] = 0 # Force catch to start from zero at zero effort
    if len(cpue_pred) > 0 and effort_range[0] == 0 and model_name == 'Schaefer' and pd.notnull(params.get('a')):
        cpue_pred[0] = params.get('a') # CPUE at zero effort is 'a' for Schaefer
    elif len(cpue_pred) > 0 and effort_range[0] == 0 and model_name == 'Fox' and pd.notnull(params.get('c')):
                        cpue_pred[0] = np.exp(params.get('c')) # CPUE at zero effort is exp(c) for Fox

    # Create surplus production curve
    fig_surplus = go.Figure()
    
    fig_surplus.add_trace(go.Scatter(
        x=effort_range, y=catch_pred, mode='lines', name=f'Kurva Model {model_name}',
        line=dict(color='#1f77b4', width=3)
    ))
    fig_surplus.add_trace(go.Scatter(
        x=yearly_data['effort (hari)'], y=yearly_data['catch (ton)'], mode='markers', name='Data Aktual (Catch)',
        marker=dict(color='#d62728', size=10, symbol='circle')
    ))
    fig_surplus.add_trace(go.Scatter(
        x=[eopt], y=[cmsy], mode='markers+text', name='Titik MSY',
        marker=dict(color='#2ca02c', size=15, symbol='star'),
        text=[f" MSY ({cmsy:.1f})"], textposition="top right"
    ))
    fig_surplus.add_vline(x=eopt, line_dash="dash", line_color="#2ca02c", 
                        annotation_text=f"E_opt = {eopt:.1f}", annotation_position="bottom right")
    fig_surplus.add_hline(y=cmsy, line_dash="dash", line_color="#2ca02c",
                        annotation_text=f"MSY = {cmsy:.1f}", annotation_position="bottom left")
    
    fig_surplus.update_layout(
        title=f'Kurva Produksi Surplus - Model {model_name}',
        xaxis_title='Upaya Penangkapan (hari)', yaxis_title='Hasil Tangkapan (ton)',
        template='plotly_white', hovermode='x unified', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
    )
                    
    # Create CPUE vs Effort plot
    fig_cpue = go.Figure()

    # Logika kondisional berdasarkan nama model
    if model_name == 'Fox':
        c = model_results.get('parameters', {}).get('c')
        d = model_results.get('parameters', {}).get('d')
        cpue_at_eopt = np.exp(c + d * eopt) if c is not None and d is not None and eopt is not None else np.nan
        y_actual = np.log(yearly_data['CPUE'])
        y_pred = np.log(cpue_pred)
        cpue_at_eopt_transformed = np.log(cpue_at_eopt)
        y_axis_title = 'ln(CPUE) (Log-scale)'
        cpue_opt_text = f" ln(CPUE_opt) ({cpue_at_eopt_transformed:.2f})"
    else:
        a = model_results.get('parameters', {}).get('a')
        b = model_results.get('parameters', {}).get('b')
        cpue_at_eopt = a + b * eopt if a is not None and b is not None and eopt is not None else np.nan
        y_actual = yearly_data['CPUE']
        y_pred = cpue_pred
        cpue_at_eopt_transformed = cpue_at_eopt
        y_axis_title = 'CPUE (ton/hari)'
        cpue_opt_text = f" CPUE_opt ({cpue_at_eopt_transformed:.2f})"

    # Menambahkan plot ke gambar (sekarang menggunakan variabel dinamis)
    fig_cpue.add_trace(go.Scatter(
        x=effort_range, y=y_pred, mode='lines', name=f'Kurva Model {model_name}',
        line=dict(color='#ff7f0e', width=3)
    ))
    fig_cpue.add_trace(go.Scatter(
        x=yearly_data['effort (hari)'], y=y_actual, mode='markers', name='Data Aktual',
        marker=dict(color='#d62728', size=10, symbol='circle')
    ))
    fig_cpue.add_vline(x=eopt, line_dash="dash", line_color="#2ca02c",
                            annotation_text=f"E_opt = {eopt:.1f}", annotation_position="bottom right")

    fig_cpue.add_trace(go.Scatter(
        x=[eopt], y=[cpue_at_eopt_transformed], mode='markers+text', name='Nilai pada E_opt',
        marker=dict(color='#2ca02c', size=10, symbol='diamond'),
        text=[cpue_opt_text], textposition="top right"
    ))

    fig_cpue.update_layout(
        title=f'Hubungan CPUE dan Upaya - Model {model_name}',
        xaxis_title='Upaya Penangkapan (hari)',
        yaxis_title=y_axis_title,  # Gunakan judul sumbu-y yang dinamis
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
    )

    # --- Modifikasi berakhir di sini ---
    return fig_surplus, fig_cpue

# def display_diagnostic_summary(model_results, model_name):
#     """Display diagnostic summary with interpretation"""
#     if model_results.get('status') == 'error':
#         st.error(f"❌ Model {model_name}: {model_results.get('error', 'Unknown error')}")
#         return
    
#     residuals_data = model_results.get('residuals', {})
#     if not residuals_data: # If residuals key exists but is empty dict
#         st.warning(f"Data diagnostik tidak tersedia untuk model {model_name}.")
#         return

#     st.subheader(f"📊 Diagnostik Model {model_name}")
    
#     # Use columns for a more compact layout
#     col1, col2 = st.columns(2)
                    
#     with col1:
#         st.markdown("**Uji Normalitas Residual (Shapiro-Wilk):**")
#         normality_p = residuals_data.get('normality_p', np.nan)
#         if pd.notnull(normality_p) and np.isfinite(normality_p):
#             if normality_p > 0.05:
#                 st.success(f"✅ Residual kemungkinan terdistribusi normal (p = {normality_p:.4f})")
#             else:
#                 st.warning(f"⚠️ Residual kemungkinan tidak terdistribusi normal (p = {normality_p:.4f})")
#         else:
#             st.info("ℹ️ Uji normalitas tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

#             st.markdown("**Uji Autokorelasi (Durbin-Watson):**")
#             dw_stat = residuals_data.get('dw_statistic', np.nan)
#             if pd.notnull(dw_stat) and np.isfinite(dw_stat):
#                 interpretation = "Tidak ada autokorelasi signifikan"
#                 emoji = "✅"
#                 if not (1.5 <= dw_stat <= 2.5): # General rule of thumb
#                     interpretation = "Kemungkinan ada autokorelasi"
#                     emoji = "⚠️"
#                     if dw_stat < 1.5: interpretation += " positif."
#                     elif dw_stat > 2.5: interpretation += " negatif."
#                 st.markdown(f"{emoji} {interpretation} (DW = {dw_stat:.3f})")
#             else:
#                 st.info("ℹ️ Uji Durbin-Watson tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

#     with col2:
#         st.markdown("**Uji Heteroskedastisitas (Pearson |resid| vs fitted):**")
#         hetero_p = residuals_data.get('hetero_p', np.nan)
#         if pd.notnull(hetero_p) and np.isfinite(hetero_p):
#             hetero_corr = residuals_data.get('hetero_correlation', np.nan)
#             corr_text = f", korelasi={hetero_corr:.3f}" if pd.notnull(hetero_corr) else ""
#             if hetero_p > 0.05:
#                 st.success(f"✅ Varians residual kemungkinan homoskedastik (p = {hetero_p:.4f}{corr_text})")
#             else:
#                 st.warning(f"⚠️ Varians residual kemungkinan heteroskedastik (p = {hetero_p:.4f}{corr_text})")
#         else:
#             st.info("ℹ️ Uji heteroskedastisitas tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

#             st.markdown("**Status Validitas Model Biologis:**")
#             status = model_results.get('status', 'error')
#             if status == 'success':
#                 st.success("✅ Model menghasilkan parameter biologis yang valid (Eopt & MSY > 0).")
#             elif status == 'warning':
#                 st.warning("⚠️ Parameter biologis mungkin tidak sepenuhnya valid atau model menunjukkan beberapa ketidaksesuaian (misal Eopt/MSY <= 0 atau koefisien model tidak ideal).")
#             else: # error
#                 st.error(f"❌ Model gagal menghasilkan parameter biologis yang valid. ({model_results.get('error', '')})")

def run_enhanced_surplus_production_analysis(yearly_data_for_model):
    """Run the complete enhanced surplus production analysis"""
    
    st.markdown("## 🎣 Analisis Model Produksi Surplus yang Ditingkatkan")
    st.markdown("---")

    if not isinstance(yearly_data_for_model, pd.DataFrame):
        st.error("Input data tidak valid. Harap unggah file CSV yang sesuai.")
        return None, None
    
    if yearly_data_for_model.empty:
        st.error("Data input kosong. Harap unggah file CSV dengan data.")
        return None, None
    
    required_cols_spm = ['catch (ton)', 'effort (hari)', 'CPUE']
    missing_cols = [col for col in required_cols_spm if col not in yearly_data_for_model.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}. Pastikan file CSV Anda memiliki header yang benar.")
        return None, None

    # Ensure data types are numeric
    for col in required_cols_spm:
        try:
            yearly_data_for_model[col] = pd.to_numeric(yearly_data_for_model[col])
        except ValueError:
            st.error(f"Kolom '{col}' mengandung nilai non-numerik. Harap periksa data Anda.")
            return None, None
    
    # Pastikan tidak ada NaN setelah konversi (jika errors='coerce' digunakan sebelumnya)
    if yearly_data_for_model[required_cols_spm].isnull().values.any():
        st.error("Data mengandung nilai NaN setelah pemeriksaan tipe. Harap periksa data Anda.")
        # Anda bisa menampilkan kolom mana yang NaN:
        # nan_cols = yearly_data_for_model.columns[yearly_data_for_model.isnull().any()].tolist()
        # st.error(f"Kolom dengan NaN: {nan_cols}")
        return None, None
            
    # Calculate both models
    with st.spinner('🔄 Menghitung model surplus produksi...'):
        schaefer_results = calculate_schaefer_enhanced(yearly_data_for_model)
        fox_results = calculate_fox_enhanced(yearly_data_for_model)
    # ⛔ CEK APAKAH MODEL MENGEMBALIKAN None
    if schaefer_results is None or fox_results is None:
        st.error("Gagal menghitung salah satu model (Schaefer atau Fox). Pastikan data cukup untuk melakukan estimasi.")
        return None, None
    if not isinstance(schaefer_results, dict) or not isinstance(fox_results, dict):
        st.error("Model tidak mengembalikan hasil yang valid.")
        return None, None
    
    # Display model comparison table
    st.subheader("📋 Perbandingan Model")
    st.write("🧪 DEBUG Schaefer Results", schaefer_results)
    st.write("🧪 DEBUG Fox Results", fox_results)
    comparison_df = _comparison_table(schaefer_results, fox_results)
    
    df_to_style = comparison_df.set_index('Parameter')

    def style_status_cells(val):
        s_val = str(val)  # Ensure it's a string
        if '✅' in s_val: 
            return 'background-color: #e8f5e8; color: black; font-weight: bold;'
        elif '⚠️' in s_val: 
            return 'background-color: #fff3cd; color: black; font-weight: bold;'
        elif '❌' in s_val: 
            return 'background-color: #f8d7da; color: black; font-weight: bold;'
        return ''  # Default no style

    # Apply styling only to the 'Status Model' row for Schaefer and Fox columns
    # Pandas Styler can be tricky for cell-specific styling based on row AND column.
    # Using applymap on a subset is cleaner if applicable.
    # Here, we can format the DataFrame to HTML for more control or use a simpler Styler.
    # For simplicity with Styler, we can make it apply to all cells in Schaefer/Fox columns and let the conditions handle specificity.
    
    styled_df = df_to_style.style.apply(
        lambda x: x.map(style_status_cells) if x.name in ['Schaefer', 'Fox'] else x, axis=0
    )
    st.dataframe(styled_df, use_container_width=True)
    
    # # Model diagnostics tabs
    # st.markdown("---")
    # st.subheader("🔍 Diagnostik dan Analisis Residual Rinci")
    # tab_schaefer, tab_fox = st.tabs(["Diagnostik Schaefer", "Diagnostik Fox"])

    # with tab_schaefer:
    #     display_diagnostic_summary(schaefer_results, 'Schaefer')
    #     if schaefer_results.get('status') != 'error':
    #         fig_residuals_s = create_residual_plots(schaefer_results, 'Schaefer')
    #         if fig_residuals_s:
    #             st.plotly_chart(fig_residuals_s, use_container_width=True)
    #         else:
    #             st.info("Plot residual untuk Schaefer tidak dapat ditampilkan.")
    
    # with tab_fox:
        # display_diagnostic_summary(fox_results, 'Fox')
        # if fox_results.get('status') != 'error':
        #     fig_residuals_f = create_residual_plots(fox_results, 'Fox')
        #     if fig_residuals_f:
        #         st.plotly_chart(fig_residuals_f, use_container_width=True)
        #     else:
        #         st.info("Plot residual untuk Fox tidak dapat ditampilkan.")
        
    # Model selection for visualization
    st.markdown("---")
    st.subheader("📈 Visualisasi Model dan Rekomendasi")
    
    valid_models_options = {}
    if schaefer_results.get('status') in ['success', 'warning'] and schaefer_results.get('biological', {}).get('Eopt') is not None:
        s_r2 = schaefer_results.get('metrics', {}).get('R2', np.nan)
        s_eopt = schaefer_results.get('biological', {}).get('Eopt', np.nan)
        if pd.notnull(s_r2) and pd.notnull(s_eopt) and s_eopt > 0:  # Ensure R2 is valid and Eopt is positive
            valid_models_options[f"Schaefer (R²={s_r2:.3f}, Eopt={s_eopt:.1f})"] = ('Schaefer', schaefer_results)
    
    if fox_results.get('status') in ['success', 'warning'] and fox_results.get('biological', {}).get('Eopt') is not None:
        f_r2_orig = fox_results.get('metrics', {}).get('R2', np.nan)
        f_eopt = fox_results.get('biological', {}).get('Eopt', np.nan)
        if pd.notnull(f_r2_orig) and pd.notnull(f_eopt) and f_eopt > 0:  # Ensure R2 is valid and Eopt is positive
            valid_models_options[f"Fox (R² Ori={f_r2_orig:.3f}, Eopt={f_eopt:.1f})"] = ('Fox', fox_results)
    
    if not valid_models_options:
        st.error("❌ Tidak ada model yang valid (dengan Eopt > 0) untuk visualisasi dan rekomendasi.")
        return None, None
    
    # Sort models by R-squared descending (extract R2 from key string for sorting)
    # This is a bit complex; simpler to just list them or sort by name if preferred.
    # For now, let user pick.
    
    selected_model_key = st.selectbox(
        "Pilih model untuk visualisasi dan rekomendasi:",
        list(valid_models_options.keys()),
        help="Pilih model berdasarkan performa statistik dan validitas biologis (Eopt > 0 diperlukan)."
    )
    
    if selected_model_key:
        model_name, model_results_selected = valid_models_options[selected_model_key]
        
        # Create visualizations
        fig_surplus, fig_cpue = _visualization(model_results_selected, yearly_data_for_model, model_name)
        
        if fig_surplus and fig_cpue:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_surplus, use_container_width=True)
            with col2:
                st.plotly_chart(fig_cpue, use_container_width=True)
            
            # Management recommendations based on the selected model
            st.markdown("---")
            st.subheader(f"💡 Rekomendasi Pengelolaan (berdasarkan Model {model_name})")
            
            bio_params = model_results_selected.get('biological', {})
            eopt = bio_params.get('Eopt')
            cmsy = bio_params.get('CMSY')
            
            if pd.notnull(eopt) and pd.notnull(cmsy) and eopt > 0 and cmsy > 0:
                current_effort = yearly_data_for_model['effort (hari)'].iloc[-1]
                current_catch = yearly_data_for_model['catch (ton)'].iloc[-1]
                actual_current_cpue = yearly_data_for_model['CPUE'].iloc[-1]

                # CPUE model at current effort
                current_cpue_model = np.nan
                if model_name == 'Schaefer':
                    a_param = model_results_selected.get('parameters', {}).get('a')
                    b_param = model_results_selected.get('parameters', {}).get('b')
                    if pd.notnull(a_param) and pd.notnull(b_param):
                        current_cpue_model = a_param + b_param * current_effort
                elif model_name == 'Fox':
                    c_param = model_results_selected.get('parameters', {}).get('c')
                    d_param = model_results_selected.get('parameters', {}).get('d')
                    if pd.notnull(c_param) and pd.notnull(d_param):
                        current_cpue_model = np.exp(c_param + d_param * current_effort)
                
                st.write(f"**Parameter Kunci Model {model_name}:**")
                st.markdown(f"- Upaya Optimal (E_opt): **{eopt:.2f} hari**")
                st.markdown(f"- Hasil Tangkapan Maksimum Lestari (MSY/CMSY): **{cmsy:.2f} ton**")
                st.markdown(f"**Kondisi Saat Ini (Data Terakhir):**")
                st.markdown(f"- Upaya (E_current): **{current_effort:.2f} hari**")
                st.markdown(f"- Hasil Tangkapan (C_current): **{current_catch:.2f} ton**")
                st.markdown(f"- CPUE Aktual (CPUE_current): **{actual_current_cpue:.2f} ton/hari**")
                if pd.notnull(current_cpue_model):
                    st.markdown(f"- CPUE Model pada E_current: **{max(0, current_cpue_model):.2f} ton/hari**")

                st.write("**Analisis Status Eksploitasi:**")
                exploitation_ratio_effort = current_effort / eopt if eopt > 0 else np.nan
                exploitation_ratio_catch = current_catch / cmsy if cmsy > 0 else np.nan  # More for context than primary indicator

                status_color = "blue"
                status_icon = "❓"

                if pd.notnull(exploitation_ratio_effort):
                    st.write(f"- Rasio Upaya (E_current / E_opt): **{exploitation_ratio_effort:.2f}**")
                    if exploitation_ratio_effort < 0.8:
                        status_text = "**Under-exploited (Kurang Dimanfaatkan)**"
                        recommendation_text = ("Upaya penangkapan saat ini secara signifikan di bawah tingkat optimal (E_opt). "
                                            "Ada potensi untuk meningkatkan upaya menuju E_opt guna menaikkan hasil tangkapan. "
                                            "Lakukan peningkatan secara bertahap dengan monitoring ketat.")
                        status_color = "green"
                        status_icon = "📉"
                    elif exploitation_ratio_effort <= 1.2:  # (0.8 to 1.2)
                        status_text = "**Fully-exploited (Dimanfaatkan Penuh)**"
                        recommendation_text = ("Upaya penangkapan saat ini berada pada atau mendekati tingkat optimal. "
                                            "Pertahankan atau lakukan penyesuaian kecil pada upaya. Fokus pada monitoring stok, "
                                            "efisiensi, dan keberlanjutan.")
                        status_color = "orange"
                        status_icon = "🎣"
                    else:  # > 1.2
                        status_text = "**Over-exploited (Dimanfaatkan Berlebih)**"
                        recommendation_text = ("Upaya penangkapan saat ini melebihi tingkat optimal. "
                                            "Dianjurkan untuk mengurangi upaya penangkapan menuju E_opt untuk pemulihan stok "
                                            "dan keberlanjutan jangka panjang. Pertimbangkan langkah pengelolaan yang lebih ketat.")
                        status_color = "red"
                        status_icon = "📈"
                else:
                    status_text = "Status tidak dapat ditentukan (E_opt tidak valid)."
                    recommendation_text = "Tidak ada rekomendasi spesifik karena E_opt tidak valid."
                
                # Display using st.metric for better visual hierarchy
                st.metric(
                    label=f"Status Eksploitasi (Upaya)", 
                    value=status_text.split('(')[0].replace("*",""), 
                    delta=f"{exploitation_ratio_effort:.2f} (E/Eopt)", 
                    delta_color="off"
                )
                st.markdown(f"<p style='color:{status_color};'>{status_icon} {status_text}</p>", unsafe_allow_html=True)
                st.info(f"**Rekomendasi Umum:** {recommendation_text}")

                st.markdown(
                    """
                    ---
                    **Catatan Penting:**
                    - Rekomendasi ini bersifat umum dan didasarkan murni pada hasil model produksi surplus yang dipilih.
                    - Keputusan pengelolaan perikanan yang komprehensif harus selalu mempertimbangkan berbagai faktor lain, termasuk aspek biologi spesies target (misalnya, ukuran pertama kali matang gonad, laju pertumbuhan), kondisi ekosistem, dampak sosial-ekonomi bagi nelayan dan masyarakat pesisir, serta kebijakan perikanan yang berlaku.
                    - Akurasi dan kualitas data input (catch dan effort time series) sangat krusial. Pastikan data yang digunakan representatif dan andal.
                    - Model produksi surplus memiliki asumsi dan keterbatasan. Sebaiknya gunakan pendekatan ini sebagai salah satu alat bantu dalam evaluasi stok, bukan satu-satunya dasar pengambilan keputusan.
                    - Dianjurkan untuk melakukan analisis sensitivitas dan evaluasi stok secara berkala dengan metode yang beragam jika memungkinkan.
                    """
                )
            else:
                st.warning(f"⚠️ Parameter biologis (Eopt/CMSY) untuk Model {model_name} tidak dapat dihitung dengan valid atau bernilai nol/negatif. Rekomendasi pengelolaan tidak dapat diberikan.")
        else:
            st.error(f"❌ Visualisasi tidak dapat dibuat untuk model {model_name}. Periksa kembali data input atau parameter model yang dihasilkan.")
    else:
        st.info("Pilih model dari dropdown di atas untuk melihat visualisasi dan rekomendasi.")

    # --- TAHAP 5: KESIMPULAN EVALUASI MODEL & STATUS STOK ---
    st.markdown("### 🚦 Kesimpulan Evaluasi Model & Status Pemanfaatan Stok")
    
    final_best_model_name = None
    final_best_msy = np.nan
    final_best_eopt = np.nan
    final_best_r2 = -np.inf 

    # Ambil hasil dari kalkulasi model sebelumnya
    # (Asumsi schaefer_results dan fox_results sudah ada dari pemanggilan calculate_..._enhanced)
    s_status_calc = schaefer_results.get('status')
    s_metrics = schaefer_results.get('metrics', {})
    s_bio = schaefer_results.get('biological', {})
    s_r2 = s_metrics.get('R2', np.nan)
    s_cmsy = s_bio.get('CMSY', np.nan)
    s_eopt = s_bio.get('Eopt', np.nan)
    s_valid_params = pd.notnull(s_cmsy) and s_cmsy > 0 and pd.notnull(s_eopt) and s_eopt > 0

    f_status_calc = fox_results.get('status')
    f_metrics = fox_results.get('metrics', {})
    f_bio = fox_results.get('biological', {})
    f_r2 = f_metrics.get('R2', np.nan)
    f_cmsy = f_bio.get('CMSY', np.nan)
    f_eopt = f_bio.get('Eopt', np.nan)
    f_valid_params = pd.notnull(f_cmsy) and f_cmsy > 0 and pd.notnull(f_eopt) and f_eopt > 0

    # Pilih model terbaik berdasarkan R2 jika kedua model valid
    if s_status_calc != 'error' and s_valid_params and pd.notnull(s_r2) and s_r2 > final_best_r2:
        final_best_model_name = 'Schaefer'
        final_best_msy, final_best_eopt, final_best_r2 = s_cmsy, s_eopt, s_r2
        
    if f_status_calc != 'error' and f_valid_params and pd.notnull(f_r2) and f_r2 > final_best_r2:
        final_best_model_name = 'Fox'
        final_best_msy, final_best_eopt, final_best_r2 = f_cmsy, f_eopt, f_r2

    if final_best_model_name:
        st.success(f"**Model Acuan untuk Status Pemanfaatan: {final_best_model_name}** (R²={final_best_r2:.3f})")
        st.markdown(f"**Estimasi MSY (CMSY): {final_best_msy:.2f} ton**, **Estimasi Upaya Optimal (E_opt): {final_best_eopt:.2f} hari**")

        if len(yearly_data_for_model) >= 1:
            # Gunakan data tahun terakhir untuk C_aktual dan E_aktual
            current_year_data = yearly_data_for_model.sort_values('tahun', ascending=False).iloc[0]
            c_aktual = current_year_data['catch (ton)']
            e_aktual = current_year_data['effort (hari)']
            tahun_aktual = int(current_year_data['tahun'])

            catch_ratio_to_msy = c_aktual / final_best_msy if final_best_msy > 0 else float('inf')
            effort_ratio_to_eopt = e_aktual / final_best_eopt if final_best_eopt > 0 else float('inf')
            
            status_stok_ikan = "Tidak dapat ditentukan"
            tingkat_exploitasi_desc = "N/A"
            tingkat_upaya_desc = "N/A"
            status_color = '#f0f2f6'  # Warna default netral
            rekomendasi = "Perlu analisis lebih lanjut atau data yang lebih representatif."

            # Logika berdasarkan tabel yang Anda berikan
            if pd.notnull(catch_ratio_to_msy) and pd.notnull(effort_ratio_to_eopt):
                # Kategori 1: Over-exploited (C/Cmsy ≥ 1)
                if catch_ratio_to_msy >= 1.0:
                    tingkat_exploitasi_desc = "Over-exploited (C/Cmsy ≥ 1)"
                    if effort_ratio_to_eopt < 1.0:
                        status_stok_ikan = "Healthy Stock 👍"
                        tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                        status_color = "#28a745"  # Hijau
                        rekomendasi = "Stok sehat namun upaya sudah tinggi relatif terhadap hasil. Pertimbangkan untuk tidak menambah upaya agar tetap optimal dan mencegah overfishing di masa depan."
                    else:  # E/Eopt ≥ 1
                        status_stok_ikan = "Depleting Stock 📉"
                        tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                        status_color = "#dc3545"  # Merah
                        rekomendasi = "Stok mengalami penurunan akibat penangkapan berlebih. Segera kurangi upaya penangkapan secara signifikan untuk pemulihan."
                # Kategori 2: Fully-exploited (0.5 ≤ C/Cmsy < 1)
                elif 0.5 <= catch_ratio_to_msy < 1.0:
                    tingkat_exploitasi_desc = "Fully-exploited (0.5 ≤ C/Cmsy < 1)"
                    if effort_ratio_to_eopt < 1.0:
                        status_stok_ikan = "Recovery Stock ↗️"
                        tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                        status_color = "#17a2b8"  # Biru muda/info
                        rekomendasi = "Stok dalam tahap pemulihan dengan upaya di bawah optimal. Upaya dapat ditingkatkan secara hati-hati menuju E_opt dengan monitoring ketat."
                    else:  # E/Eopt ≥ 1
                        status_stok_ikan = "Overfishing Stock 🎣⚠️"
                        tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                        status_color = "#ffc107"  # Kuning/peringatan
                        rekomendasi = "Upaya penangkapan sudah melebihi batas optimal meskipun hasil tangkapan belum menunjukkan over-eksploitasi penuh. Kurangi upaya untuk menjaga keberlanjutan."
                # Kategori 3: Moderate exploited (C/Cmsy < 0.5) - Menggabungkan dua baris terakhir Anda
                elif catch_ratio_to_msy < 0.5:
                    if catch_ratio_to_msy > 0.2:  # (0.2 < C/Cmsy < 0.5)
                        tingkat_exploitasi_desc = "Moderate exploited (0.2 < C/Cmsy < 0.5)"
                    else:  # (C/Cmsy ≤ 0.2)
                        tingkat_exploitasi_desc = "Low exploited / Potentially Collapsed (C/Cmsy ≤ 0.2)"
                    
                    if effort_ratio_to_eopt < 1.0:  # Underfishing (E/Eopt < 1)
                        status_stok_ikan = "Transitional Recovery Stock / Potensi Besar ⬆️"
                        tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                        status_color = "#007bff"  # Biru primer
                        rekomendasi = "Hasil tangkapan masih rendah, namun upaya juga rendah. Ada potensi besar untuk peningkatan jika stok dapat pulih atau memang belum dimanfaatkan optimal. Perlu investigasi lebih lanjut terhadap kondisi stok."
                    else:  # Overfishing (E/Eopt ≥ 1)
                        if catch_ratio_to_msy <= 0.2:
                            status_stok_ikan = "Collapsed Stock 💔"
                            tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                            status_color = "#6c757d"  # Abu-abu tua/gelap
                            rekomendasi = "Stok kemungkinan besar telah kolaps akibat tekanan penangkapan yang tinggi di masa lalu atau saat ini. Perlu tindakan drastis seperti moratorium dan rencana pemulihan jangka panjang."
                        else:  # (0.2 < C/Cmsy < 0.5) and Overfishing
                            status_stok_ikan = "Overfishing Stock (Risiko Tinggi) 🎣📉"
                            tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                            status_color = "#fd7e14"  # Oranye
                            rekomendasi = "Upaya penangkapan berlebih pada stok yang sudah moderat. Sangat berisiko. Segera kurangi upaya penangkapan."
                else:  # Jika tidak masuk kategori di atas (seharusnya tidak terjadi jika logika C/Cmsy sudah mencakup semua)
                    status_stok_ikan = "Status Tidak Terdefinisi ❓"
                    status_color = '#6c757d'  # Abu-abu
                    rekomendasi = "Kombinasi rasio C/Cmsy dan E/Eopt tidak masuk dalam kategori standar. Perlu review data dan model."

            # Tampilkan status dengan card yang lebih informatif
            text_color = 'white' if status_color not in ['#f0f2f6', '#ffc107', '#17a2b8'] else 'black'
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:{status_color}; color:{text_color}; border-left:8px solid darkgrey; margin-bottom:20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="margin-top:0; margin-bottom:15px; font-weight:bold; color:inherit; font-size:1.3em;">Status Pemanfaatan Stok Ikan (Tahun {tahun_aktual})</h4>
                <p style="font-size:1.5em; font-weight:bold; margin-bottom:15px; color:inherit;">{status_stok_ikan}</p>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:15px;">
                    <div>
                        <p style="margin:0; font-size:0.9em; color:inherit; opacity:0.9;">Hasil Tangkapan Aktual (C):</p>
                        <p style="margin:0; font-size:1.1em; font-weight:bold; color:inherit;">{c_aktual:.2f} ton</p>
                    </div>
                    <div>
                        <p style="margin:0; font-size:0.9em; color:inherit; opacity:0.9;">Upaya Penangkapan Aktual (E):</p>
                        <p style="margin:0; font-size:1.1em; font-weight:bold; color:inherit;">{e_aktual:.2f} hari</p>
                    </div>
                    <div>
                        <p style="margin:0; font-size:0.9em; color:inherit; opacity:0.9;">Rasio C/CMSY:</p>
                        <p style="margin:0; font-size:1.1em; font-weight:bold; color:inherit;">{catch_ratio_to_msy:.2f} <span style="font-size:0.8em; opacity:0.8;"><i>({tingkat_exploitasi_desc})</i></span></p>
                    </div>
                    <div>
                        <p style="margin:0; font-size:0.9em; color:inherit; opacity:0.9;">Rasio E/E_opt:</p>
                        <p style="margin:0; font-size:1.1em; font-weight:bold; color:inherit;">{effort_ratio_to_eopt:.2f} <span style="font-size:0.8em; opacity:0.8;"><i>({tingkat_upaya_desc})</i></span></p>
                    </div>
                </div>
                <hr style="border-top: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
                <p style="margin-bottom:0; font-weight:bold; color:inherit;">Rekomendasi Umum:</p>
                <p style="margin-top:5px; color:inherit; opacity:0.95;">{rekomendasi}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Tidak cukup data tahunan untuk menghitung status pemanfaatan stok berdasarkan data terkini.")
    else:
        st.error("Tidak ada model (Schaefer atau Fox) yang menghasilkan parameter MSY dan E-opt yang valid (positif) untuk penentuan status pemanfaatan.")
    return schaefer_results, fox_results
    




# Konfigurasi layout Streamlit
st.set_page_config(
    page_title="Sistok App",
    page_icon="🐟",
    layout="wide"
)
# # Inisialisasi OpenAI Client
# client = OpenAI(
#     api_key=st.secrets.DEEPSEEK,
#     base_url="https://api.deepseek.com"
           
#                 )

# Data ASLIII
# # ID file Googel Drive
# file_id = '1eACQIHOn3oS96V8rHzN6VlMuKtNX5raz'
# drive_url = f'https://drive.google.com/uc?id={file_id}'


# Data DEMOO
# ID file Googel Drive
file_id = '1wXxn-GJtVfEaZJTH-IPe4svur9uLbZIs' 
drive_url = f'https://drive.google.com/uc?id={file_id}'








# Fungsi untuk memuat data dari database atau file CSV
@st.cache_data
def load_data():
    try:
        # Download file CSV
        # ASLI
        file_path = 'data_bersih.csv'  
        #DEMO
        file_path = 'data/data_bersih_demo.csv'
        gdown.download(drive_url, file_path, quiet=False)
        # Baca file CCSV
        df = pd.read_csv(drive_url,  low_memory=False) #kalo mau gunain demo pake yg file_path
        # Konversi tanggal ke tipe datetime
        df['tanggal_berangkat'] = pd.to_datetime(df['tanggal_berangkat'], errors='coerce')
        df['tanggal_kedatangan'] = pd.to_datetime(df['tanggal_kedatangan'], errors='coerce')
        df['tahun'] = df['tanggal_kedatangan'].dt.year

        
        return df

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file 'data_bersih.csv' ada di folder './data/'." )
        return pd.DataFrame()

# Fungsi filter data
def filter_data(df, pelabuhan_kedatangan_id, nama_ikan_id, start_year, end_year, time_frame):

    # Filter data berdasarkan pelabuhan kedatangan
    if pelabuhan_kedatangan_id:
        df = df[df['pelabuhan_kedatangan_id'] == pelabuhan_kedatangan_id]

    # Filter data berdasarkan nama ikan
    if nama_ikan_id:
        df = df[df['nama_ikan_id'].isin(nama_ikan_id)]

    # Filter berdasarkan tahun
    if start_year:
        df = df[df['tahun'] >= start_year]
    if end_year:
        df = df[df['tahun'] <= end_year]

    # Filter berdasarkan time frame
    if time_frame == 'Daily':
        df['time_period'] = df['tanggal_kedatangan'].dt.date
    elif time_frame == 'Weekly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('W').astype(str)
    elif time_frame == 'Monthly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('M').astype(str)
    elif time_frame == 'Yearly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('Y').astype(str)


    return df

# Function to get OpenAI chat response

def analyze_fishing_data(query, filtered_data):
    """
    Fungsi untuk menganalisis data perikanan berdasarkan query pengguna
    """
    query = query.lower()
    response = ""
    
    try:
        # Analisis total tangkapan
        if 'total tangkapan' in query or 'berapa tangkapan' in query:
            # Filter berdasarkan jenis ikan jika disebutkan
            for fish in filtered_data['nama_ikan_id'].unique():
                if fish.lower() in query:
                    specific_data = filtered_data[filtered_data['nama_ikan_id'].str.lower() == fish.lower()]
                    
                    # Filter tahun jika disebutkan
                    for year in filtered_data['tahun'].unique():
                        if str(year) in query:
                            year_data = specific_data[specific_data['tahun'] == year]
                            total = year_data['berat'].sum()
                            return f"Total tangkapan {fish} pada tahun {year} adalah {total:,.2f} Kg"
                    
                    # Jika tahun tidak disebutkan, tampilkan semua tahun
                    yearly_data = specific_data.groupby('tahun')['berat'].sum()
                    response = f"Total tangkapan {fish} per tahun:\n"
                    for year, total in yearly_data.items():
                        response += f"Tahun {year}: {total:,.2f} Kg\n"
                    return response

        # Analisis alat tangkap
        elif 'alat tangkap' in query or 'jenis alat' in query:
            alat_tangkap = filtered_data.groupby('jenis_api')['berat'].sum().sort_values(ascending=False)
            response = "Alat tangkap yang digunakan (berdasarkan total tangkapan):\n"
            for alat, total in alat_tangkap.items():
                response += f"{alat}: {total:,.2f} Kg\n"
            return response

        # Analisis tren tahunan
        elif 'tren' in query or 'perkembangan' in query:
            yearly_trend = filtered_data.groupby('tahun')['berat'].sum()
            max_year = yearly_trend.idxmax()
            min_year = yearly_trend.idxmin()
            
            response = "Analisis tren tangkapan:\n"
            response += f"Tahun dengan tangkapan tertinggi: {max_year} ({yearly_trend[max_year]:,.2f} Kg)\n"
            response += f"Tahun dengan tangkapan terendah: {min_year} ({yearly_trend[min_year]:,.2f} Kg)\n"
            
            # Hitung pertumbuhan year-over-year
            yoy_growth = yearly_trend.pct_change() * 100
            response += "\nPertumbuhan year-over-year:\n"
            for year, growth in yoy_growth.items():
                if not pd.isna(growth):
                    response += f"{year}: {growth:,.1f}%\n"
            return response

        # Analisis nilai produksi
        elif 'nilai produksi' in query or 'nilai ekonomi' in query:
            if 'tahun' in query:
                for year in filtered_data['tahun'].unique():
                    if str(year) in query:
                        year_data = filtered_data[filtered_data['tahun'] == year]
                        total_value = year_data['nilai_produksi'].sum()
                        return f"Total nilai produksi tahun {year}: Rp {total_value:,.2f}"
            
            total_value = filtered_data['nilai_produksi'].sum()
            avg_value = filtered_data.groupby('tahun')['nilai_produksi'].mean()
            response = f"Total nilai produksi: Rp {total_value:,.2f}\n"
            response += "Rata-rata nilai produksi per tahun:\n"
            for year, value in avg_value.items():
                response += f"Tahun {year}: Rp {value:,.2f}\n"
            return response

        # Analisis jenis ikan
        elif 'jenis ikan' in query or 'ikan apa' in query:
            top_fish = filtered_data.groupby('nama_ikan_id')['berat'].sum().sort_values(ascending=False).head(5)
            response = "5 jenis ikan dengan tangkapan terbanyak:\n"
            for fish, total in top_fish.items():
                response += f"{fish}: {total:,.2f} Kg\n"
            return response

        # Default response
        else:
            return """Saya dapat membantu Anda menganalisis data perikanan. Anda dapat bertanya tentang:
                1 . Total tangkapan (per jenis ikan/tahun)
                2. Alat tangkap yang digunakan
                3. Tren tangkapan tahunan
                4. Nilai produksi
                5. Jenis ikan dominan

                Contoh: 'Berapa total tangkapan cumi tahun 2022?' atau 'Apa saja alat tangkap yang digunakan?'"""

    except Exception as e:
        return f"Maaf, terjadi kesalahan dalam menganalisis data: {str(e)}"

# Ganti fungsi get_openai_response dengan fungsi ini
def get_openai_response(query, filtered_data):
    return analyze_fishing_data(query, filtered_data)

# st.write('Kolom yang ada:', data.columns)


# CSS
st.markdown(
    """
    <style>
    .metric-box{
      border: 1px solid #ccc;
      padding: 10px;
      border-radius: 5px;
    
      margin: 5px;
      text-align: center;
    }
    </style>
""", unsafe_allow_html=True
)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown("<h1 style='text-align: center; '>SISTOK</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Fish Stock Analysis Tools</h2>", unsafe_allow_html=True)



# Menu
menu = option_menu(None, ['Dashboard', 'Analysis', 'About'],
    icons= ['house', 'graph-up', 'book'],
    menu_icon='cast', default_index=0, orientation='horizontal')

# # Sidebar untuk navigasi
# menu = st.sidebar.radio('Navigasi', ['Dashboard', 'Analysis', 'About'])

# Memuat data
data = load_data()

if menu == 'Dashboard':
    # Custom CSS for more elegant styling
    st.markdown("""
    <style>
        /* Main page styling */
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            color: #1E88E5;
            padding-bottom: 20px;
            text-align: center;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 30px;
        }
        
        /* Metrics styling */
        .metric-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }
        
        .metric-box {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease;
            color: #333;
        }
        
        .metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        .metric-icon {
            font-size: 28px;
            margin-bottom: 10px;
            color: #1E88E5;
        }
        
        .metric-title {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #1E88E5;
        }
        
        /* Chart container styling */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Filter panel styling */
        .sidebar .stSelectbox, .sidebar .stMultiSelect, .sidebar .stNumberInput {
            margin-bottom: 20px;
        }
        
        /* Chat styling */
        .chat-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .user-message {
            background-color: #DCF8C6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .assistant-message {
            background-color: #ECECEC;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Data preview styling */
        .preview-header {
            font-weight: 600;
            color: #1E88E5;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* Warning and success message styling */
        .custom-warning {
            background-color: #FFF3CD;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #FFD166;
            margin-bottom: 20px;
        }
        
        .custom-success {
            background-color: #D4EDDA;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #8BBF9F;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    

    # Sidebar Filtering with improved styling
    st.sidebar.markdown("<h3 style='text-align: center; color: #1E88E5;'>Filter Data</h3>", unsafe_allow_html=True)
    
    # Add a separator line
    st.sidebar.markdown("<hr style='margin: 15px 0px; border: none; height: 1px; background-color: #f0f2f6;'>", unsafe_allow_html=True)
    
    pelabuhan = st.sidebar.selectbox("Pilih Pelabuhan", options=[None] + list(data['pelabuhan_kedatangan_id'].unique()))

    jenis_ikan = st.sidebar.multiselect("Pilih Jenis Ikan", options=list(data['nama_ikan_id'].unique()), default=[])

    start_year = st.sidebar.number_input('Start Year', min_value=int(data['tahun'].min()), max_value=int(data['tahun'].max()), value=int(data['tahun'].min()), step=1)
    
    end_year = st.sidebar.number_input('End Year', min_value=start_year, max_value=int(data['tahun'].max()), value=int(data['tahun'].max()), step=1)

    time_frame = st.sidebar.selectbox('Time Frame', ['Daily', 'Weekly', 'Monthly', 'Yearly'])

    # Filter data
    filtered_data = filter_data(data, pelabuhan, jenis_ikan, start_year, end_year, time_frame)

    # Chatbot with improved styling
    st.sidebar.markdown("<hr style='margin: 25px 0px; border: none; height: 1px; background-color: #f0f2f6;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center; color: #1E88E5;'>Ask AI</h3>", unsafe_allow_html=True)

    # Chat input with better styling
    user_input = st.sidebar.text_input('Ask about the Data:', key='chat_input', placeholder="Type your question here...")

    # Send Button with improved styling
    if st.sidebar.button('Send', key='send_button', use_container_width=True):
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})

            # Get bot response
            with st.spinner('Thinking...'):
                bot_response = get_openai_response(user_input, filtered_data)
            # Add bot response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})

    # Display chat history with improved styling
    st.sidebar.markdown("<h4 style='color: #666; margin-top: 20px;'>Riwayat Chat</h4>", unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.sidebar.markdown(f"""
            <div class='user-message'>
                <b>Anda:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"""
            <div class='assistant-message'>
                <b>Assistant:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)

    # Clear chat history button with better styling
    if st.sidebar.button("Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Data completeness check with improved styling
    if 2024 in range(start_year, end_year+1):
        if not filtered_data.empty:
            data_tahun_2024 = filtered_data[filtered_data['tahun'] == 2024]
            if data_tahun_2024.empty or data_tahun_2024['berat'].sum() == 0:
                st.markdown("""
                <div class='custom-warning'>
                    ⚠️ Data tahun 2024 belum lengkap. Mohon diperhatikan!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='custom-success'>
                    ✅ Data tahun 2024 sudah lengkap.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-warning'>
                ⚠️ Data tidak tersedia. Silahkan periksa kembali filter Anda.
            </div>
            """, unsafe_allow_html=True)

    # Rename column
    columns_to_rename = {
        'nilai_produksi': 'Nilai Produksi',
        'jumlah_hari': 'Jumlah Hari',
        'pelabuhan_kedatangan_id': 'Pelabuhan Kedatangan',
        'pelabuhan_keberangkatan_id': 'Pelabuhan Keberangkatan',
        'kelas_pelabuhan': 'Port Class',
        'provinsi': 'Provinsi',
        'tanggal_berangkat': 'Tanggal Berangkat',
        'tanggal_kedatangan': 'Tanggal Kedatangan',
    }
    # Rename columns that exist in dataframe
    filtered_data = filtered_data.rename(columns={k: v for k, v in columns_to_rename.items() if k in filtered_data.columns})

    # Compute top analytics
    if not filtered_data.empty:
        total_tangkapan = float(pd.Series(filtered_data['berat']).sum())
        total_nilai_produksi = float(pd.Series(filtered_data['Nilai Produksi']).sum())
        total_hari = filtered_data['Jumlah Hari'].sum()
        total_ikan = filtered_data['nama_ikan_id'].nunique()
    else:
        total_tangkapan = total_nilai_produksi = total_hari = total_ikan = 0

    # Display top analytics with improved styling
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    
    total1, total2, total3, total4 = st.columns(4, gap='small')
    
    with total1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>🐟</div>
            <div class='metric-title'>Total Tangkapan</div>
            <div class='metric-value'>{total_tangkapan:,.0f} Kg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>💵</div>
            <div class='metric-title'>Nilai Produksi</div>
            <div class='metric-value'>{total_nilai_produksi:,.0f} IDR</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total3:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>📆</div>
            <div class='metric-title'>Total Hari</div>
            <div class='metric-value'>{total_hari}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total4:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>🎣</div>
            <div class='metric-title'>Jenis Ikan</div>
            <div class='metric-value'>{total_ikan}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Improved data preview section
    with st.expander('PREVIEW DATASET', expanded=False):
        st.markdown("<p class='preview-header'>Select columns to display:</p>", unsafe_allow_html=True)
        showData = st.multiselect('Filter:', filtered_data.columns, default=filtered_data.columns)
        st.dataframe(filtered_data[showData], use_container_width=True)  
    
    # Graphs with enhanced styling
    # Graph 1: Yearly Catch Data
    tangkapan_tahunan = filtered_data.groupby('tahun').agg({'berat': 'sum'}).reset_index()
    fig_tangkapan = px.line(
        tangkapan_tahunan, 
        x='tahun', 
        y='berat', 
        title='TOTAL BERAT TANGKAPAN PER TAHUN',
        markers=True,
        color_discrete_sequence=['#1E88E5'],
    )
    fig_tangkapan.update_layout(
        xaxis=dict(tickmode='linear'),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#EEEEEE'),
        xaxis_title="Tahun",
        yaxis_title="Berat (Kg)",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
        hovermode='x unified',
    )
    fig_tangkapan.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
    )

    # Graph 2: Top 10 Fish Types
    tangkapan_dominan = (
        filtered_data.groupby('nama_ikan_id').agg({'berat': 'sum'})
        .reset_index().sort_values(by='berat', ascending=False).head(10)
    )
    fig_tangkapan_dominan = px.bar(
        tangkapan_dominan,
        x='berat', 
        y='nama_ikan_id',
        orientation='h',
        title="JENIS TANGKAPAN TERBANYAK",
        color_discrete_sequence=['#1E88E5'],
    )
    fig_tangkapan_dominan.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True, 
            gridcolor='#EEEEEE',
            categoryorder='total ascending',
            title="Jenis Ikan",
        ),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#EEEEEE',
            title="Berat (Kg)",
        ),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
    )
    fig_tangkapan_dominan.update_traces(
        marker_color='#1E88E5',
        hovertemplate='<b>%{y}</b><br>Berat: %{x:,.0f} Kg<extra></extra>'
    )
    
    # Pie chart with better styling
    alat_tangkap_dominan = filtered_data.groupby('jenis_api').agg({'berat': 'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
    fig_alat_tangkap = px.pie(
        alat_tangkap_dominan, 
        names='jenis_api', 
        values='berat', 
        title='ALAT TANGKAP DOMINAN',
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_alat_tangkap.update_layout(
        legend_title='jenis_api',
        legend_y=0.9,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
    )
    fig_alat_tangkap.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        textfont_size=12,
    )

    # Display charts in a more visually appealing way
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_tangkapan, use_container_width=True)
    with right:
        st.plotly_chart(fig_tangkapan_dominan, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display pie chart in its own container
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig_alat_tangkap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
          

elif menu == 'Analysis':
    # Header
    st.markdown("""
    <div style="background-color:#1E3A8A; padding:10px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; text-align:center;">📊 Analysis Dashboard</h1>
        <p style="color:#E5E7EB; text-align:center; font-size:1.2em;">Analisis data perikanan menggunakan model statistik dan visualisasi</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload dan download
    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div style="background-color:#F3F4F6; padding:15px; border-radius:5px; border-left:5px solid #1E40AF;">
                <h3 style="color:#1E3A8A;">Upload Data CSV</h3>
                <p style="color:#4B5563;">Unggah file CSV Anda untuk analisis data perikanan.</p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                'Pilih file CSV untuk dianalisis',
                type=['csv'],
                help='Limit: 200MB per file'
            )

        with col2:
            st.markdown("""
            <div style="background-color:#F3F4F6; padding:15px; border-radius:5px; border-left:5px solid #059669;">
                <h3 style="color:#065F46;">Download Sample Data</h3>
                <p style="color:#4B5563;">Download format CSV sampel.</p>
            </div>
            """, unsafe_allow_html=True)

            try:
                with open('./data/data_kembung_karangantu.csv', 'r') as file:
                    sample_csv_content = file.read()

                st.download_button(
                    label='📥 Download Sample CSV',
                    data=sample_csv_content,
                    file_name='sample_data.csv',
                    mime='text/csv',
                    help='Klik untuk mengunduh data sampel'
                )
            except FileNotFoundError:
                st.error('Sample data tidak ditemukan.')

    st.markdown("<hr style='border: 1px solid #E5E7EB; margin: 20px 0;'>", unsafe_allow_html=True)

    # Proses file yang diupload
    if uploaded_file is not None:
        user_data_raw = pd.read_csv(uploaded_file)
        user_data_processed = user_data_raw.copy()
        st.success("File berhasil diupload")

        # --- TAHAP AWAL : VALIDASI DAN PRA-PEMROSESAN DATA UNGGAHAN----
        st.markdown("### Validasi & Pra-pemrosesan Data Awal Unggahan")
        with st.expander("Lihat detail Validasi & Pra Pemrosesan awal", expanded=False):
            st.write("Data awal yang diunggah ( 5 baris pertama):")
            st.dataframe(user_data_processed.head())

            required_cols = ['tahun', 'jenis_api', 'berat', 'Jumlah Hari'] # Kolom 'Nilai Produksi' opsional untuk beberapa chart awal
            missing_cols = [col for col in required_cols if col not in user_data_processed.columns]
            if missing_cols:
                st.error(f"Kolom berikut WAJIB ada di file CSV Anda untuk analisis penuh: {', '.join(missing_cols)}. Beberapa chart mungkin tidak tampil atau analisis lanjutan gagal.")
                    # Tidak st.stop() di sini agar chart awal yang tidak butuh semua kolom tetap bisa tampil

            # Konversi tipe data & penanganan NaN untuk kolom inti EDA
            if 'tahun' in user_data_processed.columns:
                user_data_processed['tahun'] = pd.to_numeric(user_data_processed['tahun'], errors='coerce').astype('Int64')
            if 'jenis_api' in user_data_processed.columns:
                user_data_processed['jenis_api'] = user_data_processed['jenis_api'].astype(str).str.strip()
            if 'berat' in user_data_processed.columns:
                user_data_processed['berat'] = pd.to_numeric(user_data_processed['berat'], errors='coerce')
            if 'Jumlah Hari' in user_data_processed.columns:
                user_data_processed['Jumlah Hari'] = pd.to_numeric(user_data_processed['Jumlah Hari'], errors='coerce')
           


            # Drop baris jika kolom esensial untuk agregasi tahunan (tahun, berat) adalah NaN
            user_data_processed.dropna(subset=['tahun', 'berat'], inplace=True)
                
            if user_data_processed.empty:
                st.error("Setelah pra-pemrosesan dasar, tidak ada data valid yang tersisa. Proses dihentikan.")
                st.stop()
                
            st.write("Data setelah konversi tipe dasar (5 baris pertama):")
            st.dataframe(user_data_processed.head())

        

        num_rows = user_data_processed.shape[0]
        num_cols = user_data_processed.shape[1]

        st.markdown(f"""
        <div style="background-color:#F9FAFB; padding:15px; border-radius:8px; margin:15px 0;">
            <h3 style="color:#1F2937;">Dataset Overview</h3>
            <div style="display:flex; gap:20px;">
                <div style="flex:1; background:white; padding:15px; border-left:4px solid #3B82F6;">
                    <h4 style="color:#3B82F6;">Jumlah Data</h4>
                    <p style="font-size:1.5em;">{num_rows:,}</p>
                </div>
                <div style="flex:1; background:white; padding:15px; border-left:4px solid #8B5CF6;">
                    <h4 style="color:#8B5CF6;">Jumlah Kolom</h4>
                    <p style="font-size:1.5em;">{num_cols}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander('📋 Lihat Dataset'):
            st.dataframe(user_data_processed, height=300)

        # Contoh analisis
        if 'tahun' in user_data_processed.columns and 'berat' in user_data_processed.columns:
            data_per_year_eda = user_data_processed.groupby('tahun').agg(
                berat_total_kg = ('berat', 'sum'),
                nilai_produksi_total = ('Nilai Produksi', 'sum') if 'Nilai Produksi' in user_data_processed.columns else pd.NamedAgg(column='tahun', aggfunc=lambda x: 0)
            ).reset_index()
            data_per_year_eda.rename(columns={'berat_total_kg': 'berat'}, inplace=True)


            st.markdown("""
            <div style="background-color:#EFF6FF; padding:15px; border-radius:10px; margin:20px 0;">
                <h2 style="color:#1E3A8A; text-align:center;">Analisis Data Perikanan</h2>
                <p style="color:#4B5563; text-align:center;">Visualisasi data tangkapan dan alat tangkap</p>
            </div>
            """, unsafe_allow_html=True)


            if 'Nilai Produksi' in user_data_processed.columns and not data_per_year_eda.empty:
                data_per_year_eda['Harga rata-rata nilai produksi'] = np.where(data_per_year_eda['berat'] > 0, data_per_year_eda['nilai_produksi_total'] / data_per_year_eda['berat'], 0)
                data_per_year_eda['Produksi (Ton)'] = data_per_year_eda['berat'] / 1000.0
                # Hitung ulang Nilai Produksi dalam Ton * harga rata2 (jika ada)
                data_per_year_eda['Nilai Produksi (dari Ton)'] = data_per_year_eda['Produksi (Ton)'] * data_per_year_eda['Harga rata-rata nilai produksi']
            else:
                data_per_year_eda['Harga rata-rata nilai produksi'] = 0
                data_per_year_eda['Produksi (Ton)'] = data_per_year_eda['berat'] / 1000.0
                data_per_year_eda['Nilai Produksi (dari Ton)'] = 0
            
            # Membuat grafik garis untuk total berat tangkapan per tahun
            fig_data_per_year = px.line(
                data_per_year_eda, 
                x='tahun', 
                y='berat', 
                orientation='v',
                title='Total Berat Tangkapan Per Tahun', 
                template='plotly_white'  
            )
            
            # Memperbarui layout grafik
            fig_data_per_year.update_layout(
                title={
                    'text': '<b>TOTAL BERAT TANGKAPAN PER TAHUN</b>',
                    'font': {'size': 20, 'color': '#1E3A8A'},
                    'y': 0.95
                },
                xaxis=dict(
                    tickmode='linear',
                    title='Tahun',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    gridcolor='#E5E7EB'
                ),
                yaxis=dict(
                    title='Berat (kg)',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    gridcolor='#E5E7EB',
                    showgrid=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            # Perbarui garis dengan warna dan marker
            fig_data_per_year.update_traces(
                line=dict(color='#2563EB', width=3),
                marker=dict(size=8, color='#1E40AF'),
                hovertemplate='<b>Tahun:</b> %{x}<br><b>Berat:</b> %{y:,.0f} kg<extra></extra>'
            )

            # Grafik jenis API dominan

            if 'jenis_api' in user_data_processed.columns:

                api_dominan_eda = user_data_processed.groupby('jenis_api').agg({'berat':'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
            
                # Mengubah tampilan grafik bar
                fig_api_dominan = px.bar(
                    api_dominan_eda,
                    x='berat', 
                    y='jenis_api',
                    orientation='h',
                    title="Jenis API Dominan",
                    template='plotly_white',  # Menggunakan template putih
                    color_discrete_sequence=['#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE'] * 2  # Palet warna biru gradient
                )
            
                fig_api_dominan.update_layout(
                    title={
                        'text': '<b>JENIS API DOMINAN</b>',
                        'font': {'size': 20, 'color': '#1E3A8A'},
                        'y': 0.95
                    },
                    xaxis=dict(
                        title='Berat (kg)',
                        title_font={'size': 14, 'color': '#4B5563'},
                        tickfont={'size': 12},
                        gridcolor='#E5E7EB',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title='Jenis API',
                        title_font={'size': 14, 'color': '#4B5563'},
                        tickfont={'size': 12},
                        categoryorder='total ascending',
                        gridcolor='#E5E7EB'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hoverlabel=dict(bgcolor='white', font_size=12),
                    margin=dict(l=60, r=40, t=80, b=60)
                )
            
                # Update traces untuk custom hover
                fig_api_dominan.update_traces(
                    hovertemplate='<b>%{y}</b><br>Berat: %{x:,.0f} kg<extra></extra>'
                )
            
            # Menampilkan grafik dalam layout yang lebih baik
            st.markdown("""
            <style>
            .chart-container {
                background-color: #F9FAFB;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display charts in columns
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(fig_data_per_year, use_container_width=True)
            with right:
                st.plotly_chart(fig_api_dominan, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Tampilkan data produksi dalam card yang lebih baik
            st.markdown("""
            <div style="background-color:#F0FDF4; padding:15px; border-radius:10px; margin:15px 0; border:1px solid #D1FAE5;">
                <h3 style="color:#065F46; text-align:center; margin-bottom:15px;">📊 DATA PRODUKSI DAN NILAI PRODUKSI PER TAHUN</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Style DataTable
            st.markdown("""
            <style>
                .dataframe {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                }
                .dataframe th {
                    background-color: #065F46;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }
                .dataframe td {
                    padding: 10px;
                    border-bottom: 1px solid #D1FAE5;
                }
                .dataframe tr:nth-child(even) {
                    background-color: #ECFDF5;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                data_per_year_eda[['tahun', 'Produksi (Ton)', 'Harga rata-rata nilai produksi', 'nilai_produksi_total']]
                .reset_index(drop=True), 
                use_container_width=True
            )
            
        

# Ganti bagian expander dengan sistem tab yang lebih elegan
# Tambahkan custom CSS untuk tab dan perbaikan tampilan visual
        st.markdown("""
        <style>
            /* Styling untuk tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background-color: #F3F4F6;
                border-radius: 10px 10px 0px 0px;
                padding: 5px 5px 0px 5px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 45px;
                white-space: pre-wrap;
                background-color: #F3F4F6;
                border-radius: 10px 10px 0px 0px;
                gap: 1px;
                padding-left: 20px;
                padding-right: 20px;
                font-weight: 600;
                color: #4B5563;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1E40AF;
                color: white;
            }
            .stTabs [data-baseweb="tab-highlight"] {
                display: none;
            }
            /* Content container styling */
            .tab-content {
                background-color: white;
                border-radius: 0px 0px 10px 10px;
                padding: 20px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }
            /* Charts container */
            .chart-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-top: 15px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            /* Table styling */
            .styled-table {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                border: 1px solid #E5E7EB;
                margin-bottom: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            /* Header styling */
            .header-section {
                background-color: #EFF6FF;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 5px solid #1E40AF;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .header-title {
                color: #1E40AF;
                font-size: 1.6em;
                margin-bottom: 8px;
                font-weight: bold;
            }
            .header-subtitle {
                color: #4B5563;
                font-size: 1.1em;
                line-height: 1.4;
            }
            /* Tabel dataframe styling */
            .dataframe-container {
                padding: 0px !important;
                max-height: 350px;
                overflow-y: auto;
            }
            .dataframe-container td, .dataframe-container th {
                font-size: 13px !important;
                padding: 5px 10px !important;
            }
            /* Tombol filter section */
            .filter-section {
                background-color: #F9FAFB;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #E5E7EB;
            }
            /* Card summary styling */
            .summary-card {
                background-color: #F0F9FF;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #0284C7;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            .summary-title {
                color: #0C4A6E;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 5px;
            }
            .summary-value {
                color: #0369A1;
                font-size: 1.8em;
                font-weight: 700;
            }
            .summary-label {
                color: #64748B;
                font-size: 0.85em;
            }
            /* Top 5 alat tangkap styling - untuk mengurangi jumlah tampilan di chart */
            .top-note {
                color: #64748B;
                font-size: 0.85em;
                font-style: italic;
                margin-top: 5px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

        # Header section untuk analisis alat tangkap
        st.markdown("""
        <div class="header-section">
            <div class="header-title">Analisis Alat Tangkap</div>
            <div class="header-subtitle">Data hasil tangkapan dan jumlah trip berdasarkan jenis alat penangkapan ikan</div>
        </div>
        """, unsafe_allow_html=True)

        # Membuat tabs untuk hasil tangkapan dan jumlah trip
        tab_hasil_eda, tab_trip_eda = st.tabs(["📊 Hasil Tangkapan per Alat", "🚢 Jumlah Trip per Alat"])

        # Tab 1: Hasil Tangkapan per Alat Tangkap
        with tab_hasil_eda:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            if {'jenis_api', 'tahun', 'berat'}.issubset(user_data_processed.columns):
                # Tambahkan section filter tahun (optional)
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
                with col_filter1:
                    tahun_list = sorted(user_data_processed['tahun'].unique().tolist())
                    selected_years = st.multiselect("Pilih Tahun", tahun_list, default=tahun_list)
                
                with col_filter2:
                    # Filter untuk top alat tangkap
                    show_top = st.checkbox("Tampilkan 5 Alat Tangkap Teratas", value=True)
                
                # Jika tidak ada tahun yang dipilih, gunakan semua tahun
                if not selected_years:
                    selected_years = tahun_list
                    
                # Filter data berdasarkan tahun yang dipilih
                filtered_data = user_data_processed[user_data_processed['tahun'].isin(selected_years)]
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Hitung summary card untuk hasil tangkapan
                total_tangkapan = filtered_data['berat'].sum()
                avg_per_trip = total_tangkapan / filtered_data['Jumlah Hari'].sum() if filtered_data['Jumlah Hari'].sum() > 0 else 0
                
                # Tampilkan summary cards
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-title">Total Hasil Tangkapan</div>
                        <div class="summary-value">{total_tangkapan:,.1f} kg</div>
                        <div class="summary-label">Dari {len(selected_years)} tahun terpilih</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum2:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #7E22CE;">
                        <div class="summary-title" style="color: #581C87;">Rata-rata per Trip</div>
                        <div class="summary-value" style="color: #7E22CE;">{avg_per_trip:,.1f} kg</div>
                        <div class="summary-label">Per hari trip</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum3:
                    unique_alat = filtered_data['jenis_api'].nunique()
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #16A34A;">
                        <div class="summary-title" style="color: #166534;">Jumlah Jenis Alat</div>
                        <div class="summary-value" style="color: #16A34A;">{unique_alat}</div>
                        <div class="summary-label">Jenis alat tangkap digunakan</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Layout dengan dua kolom
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown('<div class="styled-table">', unsafe_allow_html=True)
                    tangkapan_per_tahun = filtered_data.groupby(['jenis_api', 'tahun']).agg({'berat': 'sum'}).reset_index()
                    tangkapan_pivot = tangkapan_per_tahun.pivot(index='jenis_api', columns='tahun', values='berat').fillna(0)

                    # Tambahkan kolom total untuk tiap alat tangkap
                    tangkapan_pivot['Total'] = tangkapan_pivot.sum(axis=1)
                    
                    # Sort berdasarkan total (descending)
                    tangkapan_pivot = tangkapan_pivot.sort_values(by='Total', ascending=False)

                    # Tambahkan baris jumlah total untuk tiap alat tangkap
                    tangkapan_pivot.loc['Jumlah'] = tangkapan_pivot.sum()

                    # Reset index untuk tampilkan tabel
                    tangkapan_pivot = tangkapan_pivot.reset_index()
                    
                    # Format angka dengan pemisah ribuan
                    for col in tangkapan_pivot.columns:
                        if col != 'jenis_api':
                            tangkapan_pivot[col] = tangkapan_pivot[col].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)

                    # Tampilkan tabel dengan judul
                    st.markdown('<h3 style="color:#1E40AF; font-size:1.2em;">Hasil Tangkapan per Alat Tangkap (kg)</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(tangkapan_pivot, use_container_width=True, height=300)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Filter data untuk grafik (tanpa baris Jumlah)
                    chart_data = tangkapan_pivot[tangkapan_pivot['jenis_api'] != 'Jumlah'].copy()
                    
                    # Konversi string ke float untuk kolom Total terlebih dahulu
                    chart_data['Total_num'] = chart_data['Total'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
                    
                    # PERBAIKAN: Lakukan sorting dan filter berdasarkan Total_num, bukan Total
                    if show_top and len(chart_data) > 5:
                        chart_data = chart_data.sort_values(by='Total_num', ascending=False).head(5)
                        top_note = True
                    else:
                        top_note = False
                    
                    # Membuat grafik batang hasil tangkapan dengan tampilan lebih baik
                    fig_tangkapan_total = px.bar(
                        chart_data,
                        x='jenis_api',
                        y='Total_num',
                        title="<b>Hasil Tangkapan per Alat Tangkap</b>",
                        color='jenis_api',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={'jenis_api': 'Jenis Alat', 'Total_num': 'Total Tangkapan (kg)'},
                        template='plotly_white'
                    )
                    
                    fig_tangkapan_total.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.5)',
                        paper_bgcolor='rgba(255,255,255,0)',
                        title_font=dict(size=18, color='#1E40AF'),
                        legend_title_text='',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)',
                            tickformat=",.0f"
                        ),
                        margin=dict(l=50, r=30, t=80, b=50),
                    )
                    
                    # Tambahkan label nilai di atas setiap batang
                    fig_tangkapan_total.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Total: %{y:,.1f} kg<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_tangkapan_total, use_container_width=True)
                    
                    # Tampilkan catatan jika hanya menampilkan top 5
                    if top_note:
                        st.markdown('<div class="top-note">* Menampilkan 5 alat tangkap dengan hasil tertinggi</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tambahkan grafik pie chart untuk distribusi
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # PERBAIKAN: Gunakan pie_data yang sudah difilter dengan nilai numerik untuk pie chart
                    pie_data = chart_data.copy()
                    if len(pie_data) > 7 and not show_top:
                        # Ambil top 6, gabungkan sisanya sebagai "Lainnya"
                        pie_top = pie_data.nlargest(6, 'Total_num')
                        pie_others = pd.DataFrame({
                            'jenis_api': ['Lainnya'],
                            'Total_num': [pie_data['Total_num'].sum() - pie_top['Total_num'].sum()]
                        })
                        pie_data = pd.concat([pie_top, pie_others])
                    
                    fig_pie = px.pie(
                        pie_data,
                        values='Total_num',
                        names='jenis_api',
                        title='<b>Distribusi Hasil Tangkapan (%)</b>',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig_pie.update_layout(
                        title_font=dict(size=18, color='#1E40AF'),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=80, b=20),
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>%{value:,.1f} kg<br>%{percent}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Data tidak memiliki semua kolom yang diperlukan (jenis_api, tahun, berat).")
                
            st.markdown('</div>', unsafe_allow_html=True)

       # Tab 2: Jumlah Trip per Alat Tangkap
        with tab_trip_eda:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            if {'jenis_api', 'tahun', 'Jumlah Hari'}.issubset(user_data_processed.columns):
                # Tambahkan section filter tahun (optional)
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
                with col_filter1:
                    tahun_list = sorted(user_data_processed['tahun'].unique().tolist())
                    selected_years_trip = st.multiselect("Pilih Tahun", tahun_list, default=tahun_list, key="trip_years")
                
                with col_filter2:
                    # Filter untuk top alat tangkap
                    show_top_trip = st.checkbox("Tampilkan 5 Alat Tangkap Teratas", value=True, key="top_trip")
                
                # Jika tidak ada tahun yang dipilih, gunakan semua tahun
                if not selected_years_trip:
                    selected_years_trip = tahun_list
                    
                # Filter data berdasarkan tahun yang dipilih
                filtered_data_trip = user_data_processed[user_data_processed['tahun'].isin(selected_years_trip)]
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Hitung summary card untuk jumlah trip
                total_trip = filtered_data_trip['Jumlah Hari'].sum()
                
                # Hitung rata-rata hasil per hari
                if 'berat' in filtered_data_trip.columns:
                    total_tangkapan_trip = filtered_data_trip['berat'].sum()
                    avg_tangkapan_per_trip = total_tangkapan_trip / total_trip if total_trip > 0 else 0
                else:
                    avg_tangkapan_per_trip = 0
                
                # Tampilkan summary cards
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #EA580C;">
                        <div class="summary-title" style="color: #9A3412;">Total Hari Trip</div>
                        <div class="summary-value" style="color: #EA580C;">{int(total_trip):,}</div>
                        <div class="summary-label">Hari</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum2:
                    # Hitung jumlah total trip berbeda (perjalanan, bukan hari)
                    jumlah_trip_unik = len(filtered_data_trip)
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #0891B2;">
                        <div class="summary-title" style="color: #155E75;">Jumlah Trip Tercatat</div>
                        <div class="summary-value" style="color: #0891B2;">{jumlah_trip_unik:,}</div>
                        <div class="summary-label">Perjalanan</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum3:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #4F46E5;">
                        <div class="summary-title" style="color: #3730A3;">Rata-rata Hasil per Hari</div>
                        <div class="summary-value" style="color: #4F46E5;">{avg_tangkapan_per_trip:,.1f} kg</div>
                        <div class="summary-label">Hasil tangkapan per hari trip</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Layout dengan dua kolom
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown('<div class="styled-table">', unsafe_allow_html=True)
                    effort_per_tahun = filtered_data_trip.groupby(['jenis_api', 'tahun']).agg({'Jumlah Hari': 'sum'}).reset_index()
                    effort_pivot = effort_per_tahun.pivot(index='jenis_api', columns='tahun', values='Jumlah Hari').fillna(0)

                    # Tambahkan kolom total untuk tiap alat tangkap
                    effort_pivot['Total'] = effort_pivot.sum(axis=1)
                    
                    # Sort berdasarkan total (descending)
                    effort_pivot = effort_pivot.sort_values(by='Total', ascending=False)
                    
                    # Tambahkan baris jumlah total untuk tiap alat tangkap
                    effort_pivot.loc['Jumlah'] = effort_pivot.sum()

                    # Reset index untuk tampilkan tabel
                    effort_pivot = effort_pivot.reset_index()
                    
                    # Format angka dengan pemisah ribuan
                    for col in effort_pivot.columns:
                        if col != 'jenis_api':
                            effort_pivot[col] = effort_pivot[col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                    
                    # Tampilkan tabel dengan judul
                    st.markdown('<h3 style="color:#1E40AF; font-size:1.2em;">Jumlah Trip per Alat Tangkap (hari)</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(effort_pivot, use_container_width=True, height=300)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Filter data untuk grafik (tanpa baris Jumlah)
                    chart_data = effort_pivot[effort_pivot['jenis_api'] != 'Jumlah'].copy()
                    
                    # PERBAIKAN: Konversi string ke float untuk kolom Total TERLEBIH DAHULU
                    chart_data['Total_num'] = chart_data['Total'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
                    
                    # PERBAIKAN: Lalu ambil top 5 jika opsi dipilih berdasarkan nilai numerik
                    if show_top_trip and len(chart_data) > 5:
                        chart_data = chart_data.sort_values(by='Total_num', ascending=False).head(5)
                        top_trip_note = True
                    else:
                        top_trip_note = False
                    
                    # Membuat grafik batang jumlah trip dengan tampilan lebih baik
                    fig_trip_per_alat = px.bar(
                        chart_data,
                        x='jenis_api',
                        y='Total_num',
                        title="<b>Jumlah Trip per Alat Tangkap</b>",
                        color='jenis_api',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        labels={'jenis_api': 'Jenis Alat', 'Total_num': 'Total Trip (hari)'},
                        template='plotly_white'
                    )
                    
                    fig_trip_per_alat.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.5)',
                        paper_bgcolor='rgba(255,255,255,0)',
                        title_font=dict(size=18, color='#1E40AF'),
                        legend_title_text='',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)',
                            tickformat=",.0f"
                        ),
                        margin=dict(l=50, r=30, t=80, b=50),
                    )
                    
                    # Tambahkan label nilai di atas setiap batang
                    fig_trip_per_alat.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Total: %{y:,.0f} hari<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_trip_per_alat, use_container_width=True)
                    
                    # Tampilkan catatan jika hanya menampilkan top 5
                    if top_trip_note:
                        st.markdown('<div class="top-note">* Menampilkan 5 alat tangkap dengan jumlah trip tertinggi</div>', unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tambahkan grafik pie chart untuk distribusi
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # PERBAIKAN: Gunakan data yang sudah difilter dengan benar untuk pie chart
                    pie_data_trip = chart_data.copy()
                    if len(pie_data_trip) > 7 and not show_top_trip:
                        # Ambil top 6, gabungkan sisanya sebagai "Lainnya" berdasarkan nilai numerik
                        pie_top = pie_data_trip.nlargest(6, 'Total_num')
                        pie_others = pd.DataFrame({
                            'jenis_api': ['Lainnya'],
                            'Total_num': [pie_data_trip['Total_num'].sum() - pie_top['Total_num'].sum()]
                        })
                        pie_data_trip = pd.concat([pie_top, pie_others])
                    
                    fig_pie = px.pie(
                        pie_data_trip,
                        values='Total_num',
                        names='jenis_api',
                        title='<b>Distribusi Jumlah Trip (%)</b>',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    fig_pie.update_layout(
                        title_font=dict(size=18, color='#1E40AF'),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=80, b=20),
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>%{value:,.0f} hari<br>%{percent}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Data tidak memiliki semua kolom yang diperlukan (jenis_api, tahun, Jumlah Hari).")
            
            st.markdown('</div>', unsafe_allow_html=True)

       
            
            st.markdown('</div>', unsafe_allow_html=True)

                # Header untuk analisis lanjutan dengan desain modern
        st.markdown("""
            <div style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); 
                padding: 25px; border-radius: 12px; margin: 30px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="background-color: rgba(255,255,255,0.2); color: white; padding: 12px; 
                        border-radius: 50%; margin-right: 15px; font-size: 24px;">📈</span>
                    <h2 style="color: white; margin: 0; font-weight: 600; font-size: 28px;">ANALISIS LANJUTAN: MODEL PRODUKSI SURPLUS</h2>
                </div>
                <p style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 60px; font-size: 16px; line-height: 1.5;">
                    Model Produksi Surplus adalah metode analisis stok ikan yang mengukur hubungan antara kelimpahan stok 
                    dan upaya penangkapan. Model ini membantu dalam menentukan tingkat eksploitasi optimal dan berperan 
                    penting dalam menjaga keberlanjutan sumber daya perikanan.
                </p>
            </div>

            <style>
                /* Perbaikan tampilan sidebar */
                [data-testid=stSidebar] {
                    background-color: #F1F5F9;
                }
                
                /* Styling untuk bagian tab/expander */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 8px;
                }
                
                .stTabs [data-baseweb="tab"] {
                    background-color: #F1F5F9;
                    border-radius: 4px 4px 0px 0px;
                    padding: 10px 20px;
                    font-weight: 500;
                }
                
                .stTabs [aria-selected="true"] {
                    background-color: #3B82F6 !important;
                    color: white !important;
                }
                
                /* Styling untuk expander */
                .streamlit-expanderHeader {
                    background-color: #F1F5F9;
                    border-radius: 8px;
                    font-weight: 600;
                    color: #1E3A8A;
                }
                
                .streamlit-expanderContent {
                    background-color: white;
                    border-radius: 0px 0px 8px 8px;
                    border: 1px solid #E2E8F0;
                    border-top: none;
                }
            </style>
            """, unsafe_allow_html=True)
        
        

        
        # ------ TAHAP 1: IDENTIFIKASI ALAT TANGKAP KONSISTEN ------
        data_untuk_spm = pd.DataFrame() # Inisialisasi
        with st.expander("Filter Alat Tangkap Konsisten", expanded=False):
            # Inisialisasi parameter analisis
            min_year_analysis = int(user_data_processed['tahun'].min())
            max_year_analysis = int(user_data_processed['tahun'].max())
            analysis_period_years = list(range(min_year_analysis, max_year_analysis + 1))
            
            # Filter data valid untuk SPM
            data_spm_effort_valid = user_data_processed[
                (user_data_processed['Jumlah Hari'] > 0) & 
                (user_data_processed['berat'] > 0)
            ]
            
            if data_spm_effort_valid.empty:
                st.error("Tidak ada data dengan 'Jumlah Hari' > 0 dan 'berat' > 0 setelah pra-pemrosesan awal. Tidak dapat melanjutkan identifikasi alat tangkap konsisten")
                st.stop()

            # Identifikasi alat tangkap yang beroperasi konsisten setiap tahun
            operational_years_spm = data_spm_effort_valid.groupby("jenis_api")['tahun'].apply(set)
            gears_konsisten_spm = [
                gear for gear, year_set in operational_years_spm.items()
                if set(analysis_period_years).issubset(year_set)
            ]

            if not gears_konsisten_spm:
                st.error(f"Tidak ada alat tangkap yang beroperasi secara konsisten (memiliki data effort > 0 dan berat > 0) setiap tahun dari {min_year_analysis} hingga {max_year_analysis} untuk Model SPM.")
                st.stop()

            st.success(f"Ditemukan {len(gears_konsisten_spm)} alat tangkap konsisten untuk Model SPM:")
            st.json(gears_konsisten_spm)

            # Filter data untuk alat tangkap konsisten
            data_untuk_spm = user_data_processed[user_data_processed['jenis_api'].isin(gears_konsisten_spm)].copy()

            # Hitung persentase coverage
            total_catch_awal_kg_spm = user_data_processed['berat'].sum()
            total_catch_spm_kg = data_untuk_spm['berat'].sum()
            persentase_coverage_spm = (total_catch_spm_kg / total_catch_awal_kg_spm) * 100 if total_catch_awal_kg_spm > 0 else 0
            
            st.metric(
                label="Persentase Total Tangkapan (kg) yang Diwakili Alat Tangkap untuk SPM", 
                value=f"{persentase_coverage_spm:.2f}%"
            )

            if persentase_coverage_spm < 50:
                st.warning("Alat tangkap konsisten untuk SPM hanya mewakili kurang dari 50% total tangkapan. Hasil model mungkin sangat terbatas.")

            if data_untuk_spm.empty:
                st.error("Tidak ada data yang tersisa untuk SPM setelah filter alat tangkap konsisten. Proses dihentikan.")
                st.stop()

        # ------ TAHAP 2: STANDARISASI UPAYA & AGREGASI DATA TAHUNAN ------

        with st.expander("Lihat Detail Standarisasi & Agregasi untuk Model SPM", expanded=True):
            try:
                # Persiapan data dengan konversi unit
                data_untuk_spm['catch (ton)'] = data_untuk_spm['berat'] / 1000.0
                data_untuk_spm.rename(columns={'Jumlah Hari': 'effort (hari)'}, inplace=True)

                # --- TABEL 1: UPAYA TAHUNAN PER ALAT TANGKAP ---
                st.markdown("#### Tabel 1: Upaya Tahunan per Alat Tangkap (hari)")
                effort_pivot_table = data_untuk_spm.groupby(['tahun', 'jenis_api'])['effort (hari)'].sum().unstack(fill_value=0)
                st.dataframe(effort_pivot_table.style.format("{:,.0f}"))

                # --- TABEL 2: TANGKAPAN TAHUNAN PER ALAT TANGKAP ---
                st.markdown("#### Tabel 2: Tangkapan Tahunan per Alat Tangkap")
                
                # Tangkapan dalam kg
                catch_pivot_table_kg = data_untuk_spm.groupby(['tahun', 'jenis_api'])['berat'].sum().unstack(fill_value=0)
                st.markdown("##### Tangkapan (kg)")
                st.dataframe(catch_pivot_table_kg.style.format("{:,.0f}"))
                
                # Tangkapan dalam ton
                catch_pivot_table_ton = data_untuk_spm.groupby(['tahun', 'jenis_api'])['catch (ton)'].sum().unstack(fill_value=0)
                st.markdown("##### Tangkapan (ton)")
                st.dataframe(catch_pivot_table_ton.style.format("{:,.2f}"))

                # --- PERHITUNGAN CPUE DAN FPI ---
                # Agregasi data tahunan per alat tangkap untuk CPUE
                annual_gear_data = data_untuk_spm[data_untuk_spm['effort (hari)'] > 0].groupby(['tahun', 'jenis_api']).agg(
                    total_catch_ton=('catch (ton)', 'sum'),
                    total_effort_hari=('effort (hari)', 'sum')
                ).reset_index()

                if annual_gear_data.empty or annual_gear_data['total_effort_hari'].sum() == 0:
                    st.error("Tidak ada data effort yang valid (>0) untuk menghitung CPUE per alat tangkap.")
                    st.stop()

                # Hitung CPUE tahunan per alat tangkap
                annual_gear_data['cpue_tahunan'] = annual_gear_data['total_catch_ton'] / annual_gear_data['total_effort_hari']
                
                # --- TABEL 3: CPUE TAHUNAN PER ALAT TANGKAP ---
                st.markdown("#### Tabel 3: CPUE Tahunan per Alat Tangkap (ton/hari)")
                cpue_pivot_table = annual_gear_data.pivot(index='tahun', columns='jenis_api', values='cpue_tahunan').fillna(0)
                st.dataframe(cpue_pivot_table.style.format("{:.4f}"))

                # --- TABEL 4: PERHITUNGAN FPI ---
                st.markdown("#### Tabel 4: Rata-rata CPUE per Alat Tangkap dan Fishing Power Index (FPI)")
                
                # Hitung rata-rata CPUE per alat tangkap
                valid_cpue_data = annual_gear_data[np.isfinite(annual_gear_data['cpue_tahunan'])]
                if valid_cpue_data.empty:
                    st.error("Tidak ada data CPUE yang valid untuk menghitung rata-rata CPUE per alat tangkap.")
                    st.stop()

                avg_cpue_per_gear = valid_cpue_data.groupby('jenis_api')['cpue_tahunan'].mean().reset_index()
                avg_cpue_per_gear.rename(columns={'cpue_tahunan': 'CPUE_rata_rata'}, inplace=True)

                st.write("CPUE Rata-rata per Alat Tangkap:")
                st.dataframe(avg_cpue_per_gear.style.format({'CPUE_rata_rata': "{:.4f}"}))

                # Tentukan alat tangkap referensi dan hitung FPI
                if avg_cpue_per_gear['CPUE_rata_rata'].max() <= 0:
                    st.warning("CPUE rata-rata tertinggi adalah nol atau negatif. FPI mungkin tidak bermakna.")
                    avg_cpue_per_gear['FPI'] = 0.0
                else:
                    cpue_referensi = avg_cpue_per_gear['CPUE_rata_rata'].max()
                    alat_referensi_df = avg_cpue_per_gear[avg_cpue_per_gear['CPUE_rata_rata'] == cpue_referensi]
                    alat_referensi_nama = alat_referensi_df['jenis_api'].iloc[0] if not alat_referensi_df.empty else "Tidak Ditemukan"
                    
                    st.write(f"Alat Tangkap Referensi (CPUE Rata-rata Tertinggi): **{alat_referensi_nama}** (CPUE: {cpue_referensi:.4f} ton/hari)")
                    
                    # Hitung FPI
                    avg_cpue_per_gear['FPI'] = avg_cpue_per_gear['CPUE_rata_rata'] / cpue_referensi

                st.write("Fishing Power Index (FPI) per Alat Tangkap:")
                st.dataframe(
                    avg_cpue_per_gear[['jenis_api', 'CPUE_rata_rata', 'FPI']]
                    .sort_values(by='FPI', ascending=False)
                    .style.format({'CPUE_rata_rata': "{:.4f}", 'FPI': "{:.3f}"})
                )

                # --- PERSIAPAN DATA AKHIR UNTUK MODEL SPM ---
                # Gabungkan FPI ke data utama
                data_untuk_spm_with_fpi = pd.merge(
                    data_untuk_spm, 
                    avg_cpue_per_gear[['jenis_api', 'FPI']], 
                    on='jenis_api', 
                    how='left'
                )
                
                # Hitung effort terstandarisasi
                data_untuk_spm_with_fpi['effort_std'] = data_untuk_spm_with_fpi['effort (hari)'] * data_untuk_spm_with_fpi['FPI']

                # Agregasi data tahunan untuk model
                yearly_data_for_model = data_untuk_spm_with_fpi.groupby('tahun').agg(
                    catch_total_ton=('catch (ton)', 'sum'),
                    effort_total_std=('effort_std', 'sum')
                ).reset_index()
                
                yearly_data_for_model.rename(columns={
                    'catch_total_ton': 'catch (ton)', 
                    'effort_total_std': 'effort (hari)'
                }, inplace=True)

                # Filter data dan validasi
                yearly_data_for_model = yearly_data_for_model[yearly_data_for_model['effort (hari)'] > 1e-6]
                
                if yearly_data_for_model.empty or len(yearly_data_for_model) < 4:
                    st.error(f"Jumlah data tahunan ({len(yearly_data_for_model)}) setelah standarisasi untuk SPM terlalu sedikit. Minimal disarankan 4-5 titik.")
                    st.stop()
                
                # Hitung CPUE final
                yearly_data_for_model['CPUE'] = yearly_data_for_model['catch (ton)'] / yearly_data_for_model['effort (hari)']
                yearly_data_for_model.dropna(subset=['CPUE'], inplace=True)
                yearly_data_for_model = yearly_data_for_model[np.isfinite(yearly_data_for_model['CPUE'])]
                
                if yearly_data_for_model.empty or len(yearly_data_for_model) < 4:
                    st.error("Setelah perhitungan CPUE dan penghapusan NaN/Inf, data tahunan untuk SPM tidak cukup.")
                    st.stop()

                # --- TAMPILKAN DATA FINAL ---
                st.markdown("#### Data Tahunan Agregat Terstandarisasi Siap untuk Model SPM:")
                st.dataframe(
                    yearly_data_for_model[['tahun', 'catch (ton)', 'effort (hari)', 'CPUE']]
                    .style.format({
                        'catch (ton)': '{:.2f}', 
                        'effort (hari)': '{:.1f}', 
                        'CPUE': '{:.4f}'
                    })
                )

                # ----- PEMANGGILAN FUNGSI ANALISIS UTAMA SPM -----
                if len(yearly_data_for_model) >= 3: # Minimal data untuk model
                    schaefer_run_results, fox_run_results = run_enhanced_surplus_production_analysis(yearly_data_for_model)
                    if schaefer_run_results is None or fox_run_results is None:
                        st.error("Analisis gagal karena salah satu model tidak menghasilkan output.")
                    # Fungsi di atas akan menampilkan semua output model, termasuk kesimpulan status
                else:
                    st.warning(f"Data tahunan agregat ({len(yearly_data_for_model)} baris) tidak cukup untuk analisis model surplus produksi (min. 3).")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam analisis: {e}")
                st.error("Pastikan format file CSV Anda benar dan semua kolom yang diperlukan ada (tahun, jenis_api, berat, 'Jumlah Hari').")
                import traceback
                st.error(f"Detail Error (untuk debugging): {traceback.format_exc()}")



                


                
                

              
    else:
        st.info('Please upload a CSV file to proceed.')
        



elif menu == 'About':
    # Header with custom styling
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight:bold;
        color:#1E88E5;
        margin-bottom:0px;
    }
    .sub-font {
        font-size:20px;
        color:#424242;
        margin-top:0px;
    }
    .section-header {
        background-color:#f0f8ff;
        padding:10px;
        border-radius:5px;
        margin-top:30px;
        border-left:5px solid #1E88E5;
    }
    .card {
        background-color:white;
        padding:20px;
        border-radius:10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-bottom:20px;
        border-top:4px solid #1E88E5;
    }
    .model-card {
        background-color:#f8f9fa;
        padding:15px;
        border-radius:8px;
        margin-bottom:15px;
    }
    .feature-item {
        margin-bottom:10px;
    }
    </style>
    
    <p class="big-font">SISTOK</p>
    <p class="sub-font">Sistem Informasi Stok Perikanan</p>
    """, unsafe_allow_html=True)
    
    # Main Description with card styling
    st.markdown("""
    <div class="card">
    <h3>What is SISTOK?</h3>
    <p>SISTOK is a comprehensive web-based application designed to help fisheries researchers, managers, and stakeholders analyze and understand fish stock data effectively. This tool provides various analytical capabilities to support sustainable fisheries management decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features with better visual organization
    st.markdown('<div class="section-header"><h2>✨ Key Features</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📊 Data Management</h3>
        <div class="feature-item">• <b>CSV file upload</b> with intelligent parsing</div>
        <div class="feature-item">• <b>Real-time data processing</b> for immediate insights</div>
        <div class="feature-item">• <b>Interactive data filtering</b> capabilities</div>
        </div>
        
        <div class="card">
        <h3>📈 Visualization</h3>
        <div class="feature-item">• <b>Dynamic charts and graphs</b> for data exploration</div>
        <div class="feature-item">• <b>Catch statistics visualization</b> with multiple views</div>
        <div class="feature-item">• <b>Temporal trend analysis</b> for pattern identification</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🎯 Advanced Analytics</h3>
        <div class="feature-item">• <b>Surplus Production Models</b> (Schaefer & Fox)</div>
        <div class="feature-item">• <b>CPUE Analysis</b> with standardization options</div>
        <div class="feature-item">• <b>Fishing effort standardization</b> across gear types</div>
        </div>
        
        <div class="card">
        <h3>📆 Time Series Analysis</h3>
        <div class="feature-item">• <b>Multiple time frame options</b> for different perspectives</div>
        <div class="feature-item">• <b>Trend identification</b> using advanced algorithms</div>
        <div class="feature-item">• <b>Seasonal pattern analysis</b> for temporal insights</div>
        </div>
        """, unsafe_allow_html=True)

    # How to use - redesigned with modern expander
    st.markdown('<div class="section-header"><h2>🔍 How to Use SISTOK</h2></div>', unsafe_allow_html=True)

    with st.expander("**STEP-BY-STEP GUIDE**"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
            <h3>1. Data Upload</h3>
            <div class="feature-item">• Navigate to the <b>Analysis</b> section</div>
            <div class="feature-item">• Upload your CSV file containing fishing data</div>
            <div class="feature-item">• Ensure your data includes required columns (date, catch, effort)</div>
            </div>
            
            <div class="card">
            <h3>2. Data Exploration</h3>
            <div class="feature-item">• Use the <b>Dashboard</b> to view data overview</div>
            <div class="feature-item">• Apply filters to focus on specific time periods, ports, or fish species</div>
            <div class="feature-item">• Examine catch trends and patterns through visualizations</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
            <h3>3. Analysis</h3>
            <div class="feature-item">• Calculate CPUE for different fishing gears</div>
            <div class="feature-item">• Apply surplus production models for stock assessment</div>
            <div class="feature-item">• Estimate MSY and optimal fishing effort levels</div>
            </div>
            
            <div class="card">
            <h3>4. Results Interpretation</h3>
            <div class="feature-item">• Review visualizations and statistical metrics</div>
            <div class="feature-item">• Reliability Model Evaluation</div>
            <div class="feature-item">• Make informed fisheries management decisions</div>
            </div>
            """, unsafe_allow_html=True)

    # Access Steps - newly designed section
    st.markdown('<div class="section-header"><h2>🚪 Access Steps</h2></div>', unsafe_allow_html=True)
    
    with st.expander("**HOW TO ACCESS SISTOK**"):
        st.markdown("""
        <div class="card">
        <h3>1. Access the Web Application</h3>
        <div class="feature-item">• Open your web browser and navigate to the SISTOK application URL</div>
        <div class="feature-item">• For local access: <code>https://sistok-tools-v1.streamlit.app/</code></div>
        <div class="feature-item">• For remote access: [Application URL]</div>
        </div>
        

        
        <div class="card">
        <h3>2. Navigation</h3>
        <div class="feature-item">• Use the sidebar menu to navigate between different modules</div>
        <div class="feature-item">• Select <b>Dashboard</b> for overview statistics and trends</div>
        <div class="feature-item">• Select <b>Analysis</b> for detailed data processing workflows</div>
        <div class="feature-item">• Select <b>About</b> for application description and others</div>
        </div>
        """, unsafe_allow_html=True)

    # Graphical Representation/Models - redesigned with tabs
    st.markdown('<div class="section-header"><h2>📊 Model Representations</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Schaefer Model", "Fox Model"])
    
    with tab1:
        st.markdown("""
        <div class="model-card">
        <h3 style="color:#1E88E5;">Schaefer Model</h3>
        <p>The Schaefer model is based on the logistic growth equation and assumes a linear relationship between CPUE (Catch Per Unit Effort) and effort (E).</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Mathematical Formulation")
            st.latex(r'''CPUE = a - b \times E''')
            st.latex(r'''Y = a \times E - b \times E^2''')
            st.latex(r'''MSY = \frac{a^2}{4b}''')
            st.latex(r'''E_{MSY} = \frac{a}{2b}''')
            
        with col2:
            st.markdown("#### Model Characteristics")
            st.markdown("""
            <div class="feature-item">• CPUE decreases <b>linearly</b> with increasing fishing effort</div>
            <div class="feature-item">• Yield (Y) follows a <b>parabolic relationship</b> with effort</div>
            <div class="feature-item">• MSY occurs at <b>half the effort level</b> that would drive the stock to zero</div>
            <div class="feature-item">• <b>More conservative</b> than the Fox model at higher effort levels</div>
            """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown("""
        <div class="model-card">
        <h3 style="color:#1E88E5;">Fox Model</h3>
        <p>The Fox model assumes an exponential relationship between CPUE and effort, allowing for more sustainable harvesting at lower population levels.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Mathematical Formulation")
            st.latex(r'''CPUE = a \times e^{-b \times E}''')
            st.latex(r'''Y = a \times E \times e^{-b \times E}''')
            st.latex(r'''MSY = \frac{a}{b \times e} \times e^{-1}''')
            st.latex(r'''E_{MSY} = \frac{1}{b}''')
            
        with col2:
            st.markdown("#### Model Characteristics")
            st.markdown("""
            <div class="feature-item">• CPUE decreases <b>exponentially</b> with increasing fishing effort</div>
            <div class="feature-item">• Better suited for stocks that can sustain fishing pressure at <b>low population levels</b></div>
            <div class="feature-item">• MSY occurs at a <b>higher effort level</b> compared to the Schaefer model</div>
            <div class="feature-item">• Often provides a <b>better fit</b> for certain fisheries data</div>
            """, unsafe_allow_html=True)

    # Terminology - redesigned with modern styling
    st.markdown('<div class="section-header"><h2>📚 Key Terminology</h2></div>', unsafe_allow_html=True)
    
    with st.expander("**FISHERIES MANAGEMENT TERMS**"):
        col1, col2 = st.columns(2)
        
        terms = {
            "CPUE (Catch Per Unit Effort)": "A measure of the abundance of a target species, calculated as the total catch divided by the total fishing effort.",
            "MSY (Maximum Sustainable Yield)": "The largest yield (or catch) that can be taken from a species' stock over an indefinite period without depleting the stock.",
            "Fishing Effort": "The amount of fishing gear of a specific type used on the fishing grounds over a given unit of time.",
            "Biomass": "The total weight of fish in a stock or population.",
            "Carrying Capacity (K)": "The maximum population size of the species that the environment can sustain indefinitely.",
            "Growth Rate (r)": "The intrinsic rate of population increase in the absence of density-dependent factors.",
            "Stock Assessment": "The process of collecting and analyzing biological and statistical information to determine the changes in the abundance of fishery stocks in response to fishing.",
            "Overfishing": "Fishing activity that leads to a reduction in stock levels below acceptable levels.",
            "Surplus Production": "The total weight (biomass) produced by a fish population through growth and reproduction in excess of what is needed to maintain the population size."
        }
        
        # Split terms into two columns
        terms_list = list(terms.items())
        mid_point = len(terms_list) // 2
        
        with col1:
            for term, definition in terms_list[:mid_point]:
                st.markdown(f"""
                <div class="card">
                <h4>{term}</h4>
                <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            for term, definition in terms_list[mid_point:]:
                st.markdown(f"""
                <div class="card">
                <h4>{term}</h4>
                <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)

    # Model Evaluation - redesigned with tabs for each metric
    st.markdown('<div class="section-header"><h2>📏 Model Evaluation Metrics</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
    <p>SISTOK uses multiple statistical metrics to evaluate and compare the performance of fisheries models. 
    Each metric provides unique insights into how well a model fits the observed data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    metric_tabs = st.tabs(["RMSE", "MAE", "MAPE", "R²"])
    
    with metric_tabs[0]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Root Mean Square Error")
            st.latex(r'''RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>RMSE measures the average magnitude of the errors in predictions, with a higher penalty for large errors.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Expressed in the same units as the dependent variable</div>
            <div class="feature-item">• Gives higher weight to larger errors due to squaring</div>
            <div class="feature-item">• Useful when large errors are particularly undesirable</div>
            </div>
            """, unsafe_allow_html=True)
    
    with metric_tabs[1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Mean Absolute Error")
            st.latex(r'''MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>MAE measures the average magnitude of errors in predictions without considering their direction.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Expressed in the same units as the dependent variable</div>
            <div class="feature-item">• Treats all errors equally (no squaring)</div>
            <div class="feature-item">• More robust to outliers than RMSE</div>
            </div>
            """, unsafe_allow_html=True)
            
    with metric_tabs[2]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Mean Absolute Percentage Error")
            st.latex(r'''MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>MAPE expresses prediction accuracy as a percentage of error, making it easy to interpret across different scales.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Provides error in percentage terms (scale-independent)</div>
            <div class="feature-item">• Easier to communicate to non-technical stakeholders</div>
            <div class="feature-item">• Not suitable when actual values are close to zero</div>
            </div>
            """, unsafe_allow_html=True)
            
    with metric_tabs[3]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Coefficient of Determination")
            st.latex(r'''R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>R² indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Values range from 0 to 1 (or 0% to 100%)</div>
            <div class="feature-item">• Higher values indicate better model fit</div>
            <div class="feature-item">• R² = 1: model explains all variability in the response data</div>
            <div class="feature-item">• R² = 0: model explains none of the variability</div>
            </div>
            """, unsafe_allow_html=True)

    # Technical Requirements - redesigned with icon and card styling
    st.markdown('<div class="section-header"><h2>💻 Technical Requirements</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>Data Format Specifications</h3>
    <p>Your data should be in CSV format with the following columns:</p>
    <div class="feature-item">• <b>Date columns</b>: tanggal_berangkat, tanggal_kedatangan</div>
    <div class="feature-item">• <b>Catch data</b>: berat</div>
    <div class="feature-item">• <b>Effort data</b>: Jumlah hari</div>
    <div class="feature-item">• <b>Fishing gear</b>: jenis_api</div>
    <div class="feature-item">• Additional metadata as needed</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact Information - redesigned with better styling
    st.markdown('<div class="section-header"><h2>📬 Contact & Support</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>Developer Contact</h3>
        <div class="feature-item">📧 <b>Email</b>: tandrysimamora@gmail.com</div>
        <div class="feature-item">📱 <b>Phone</b>: +62 822 6160-6428</div>
        <div class="feature-item">🌐 <b>Website</b>: https://tndry.github.io/portfolio-website/</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:  
        st.markdown("""
        <div class="card">
        <h3>Support Resources</h3>
        <div class="feature-item">📚 <b>Documentation</b>: [Link to documentation]</div>
        <div class="feature-item">❓ <b>FAQ</b>: [Link to FAQ page]</div>
        <div class="feature-item">🎓 <b>Tutorials</b>: [Link to tutorials]</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Version info - moved to the main content area with better styling
    st.markdown("""
    <div style="margin-top:30px; text-align:center; padding:10px; background-color:#f0f8ff; border-radius:5px;">
    <p style="margin:0; color:#1E88E5; font-weight:bold;">SISTOK v2.0.0</p>
    <p style="margin:0; font-size:12px; color:#616161;">Last updated: April 2025</p>
    </div>
    """, unsafe_allow_html=True)
        

else:
    st.error('Data tidak tersedia. Silahkan periksa kembali file Anda.')




    