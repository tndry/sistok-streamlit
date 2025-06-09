

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

def create_model_comparison_table(schaefer_results, fox_results):
    """Create an elegant comparison table for both models"""
    
    # Status indicators
    def get_status_indicator(status):
        if status == 'success':
            return "✅"
        elif status == 'warning':
            return "⚠️"
        else: # error or other
            return "❌"
    
    # Helper to format values or return "-"
    def format_val(value, precision):
        if pd.notnull(value) and not np.isinf(value):
            return f"{value:.{precision}f}"
        return "-"

    # Create comparison data
    s_params = schaefer_results.get('parameters', {}) if schaefer_results.get('status') != 'error' else {}
    s_bio = schaefer_results.get('biological', {}) if schaefer_results.get('status') != 'error' else {}
    s_metrics = schaefer_results.get('metrics', {}) if schaefer_results.get('status') != 'error' else {}
    s_status = get_status_indicator(schaefer_results.get('status', 'error'))
    f_params = fox_results.get('parameters', {}) if fox_results.get('status') != 'error' else {}
    f_bio = fox_results.get('biological', {}) if fox_results.get('status') != 'error' else {}
    # For Fox, R2 can be reported from original scale, others might be from log scale or original.
    # Here, using original scale metrics for comparison of goodness-of-fit to observed CPUE/Catch
    f_metrics_display = fox_results.get('metrics_original', {}) if fox_results.get('status') != 'error' else {}
    f_status = get_status_indicator(fox_results.get('status', 'error'))
                    
    comparison_df = pd.DataFrame({
        'Parameter': [
            'Status Model',
            'Intercept (Schaefer: a, Fox: c=ln(Uinf))', # Clarified Fox intercept
            'Slope (Schaefer: b, Fox: d)',
            'E optimal (hari)',
            'MSY/CMSY (ton)',
            'R² (Schaefer: CPUE, Fox: CPUE Original)', # Clarified R2 source
            'RMSE (Schaefer: CPUE, Fox: CPUE Original)',
            'MAE (Schaefer: CPUE, Fox: CPUE Original)',
            'MAPE (%) (Schaefer: CPUE, Fox: CPUE Original)'
        ],
        'Schaefer': [
            s_status,
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
            f_status,
            format_val(f_params.get('c'), 4), # This is log-scale intercept
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
    std_residuals = residuals_data.get('standardized_residual
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

def create_model_visualization(model_results, yearly_data, model_name):
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
    
    fig_cpue.add_trace(go.Scatter(
        x=effort_range, y=cpue_pred, mode='lines', name=f'Kurva Model {model_name}',
        line=dict(color='#ff7f0e', width=3)
    ))
    fig_cpue.add_trace(go.Scatter(
        x=yearly_data['effort (hari)'], y=yearly_data['CPUE'], mode='markers', name='Data Aktual (CPUE)',
        marker=dict(color='#d62728', size=10, symbol='circle')
    ))
    fig_cpue.add_vline(x=eopt, line_dash="dash", line_color="#2ca02c",
                        annotation_text=f"E_opt = {eopt:.1f}", annotation_position="bottom right")
    # Add CPUE_MSY point if calculable and makes sense
    cpue_at_eopt = np.interp(eopt, effort_range, cpue_pred)
    fig_cpue.add_trace(go.Scatter(
        x=[eopt], y=[cpue_at_eopt], mode='markers+text', name='CPUE pada E_opt',
        marker=dict(color='#2ca02c', size=10, symbol='diamond'),
        text=[f" CPUE_opt ({cpue_at_eopt:.2f})"], textposition="top right"
    ))

    fig_cpue.update_layout(
        title=f'CPUE vs Upaya Penangkapan - Model {model_name}',
        xaxis_title='Upaya Penangkapan (hari)', yaxis_title='CPUE (ton/hari)',
        template='plotly_white', hovermode='x unified', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
    )
                    
    return fig_surplus, fig_cpue

def display_diagnostic_summary(model_results, model_name):
                    """Display diagnostic summary with interpretation"""
                    if model_results.get('status') == 'error':
                        st.error(f"❌ Model {model_name}: {model_results.get('error', 'Unknown error')}")
                        return
                    
                    residuals_data = model_results.get('residuals', {})
                    if not residuals_data: # If residuals key exists but is empty dict
                        st.warning(f"Data diagnostik tidak tersedia untuk model {model_name}.")
                        return

                    st.subheader(f"📊 Diagnostik Model {model_name}")
                    
                    # Use columns for a more compact layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Uji Normalitas Residual (Shapiro-Wilk):**")
                        normality_p = residuals_data.get('normality_p', np.nan)
                        if pd.notnull(normality_p) and np.isfinite(normality_p):
                            if normality_p > 0.05:
                                st.success(f"✅ Residual kemungkinan terdistribusi normal (p = {normality_p:.4f})")
                            else:
                                st.warning(f"⚠️ Residual kemungkinan tidak terdistribusi normal (p = {normality_p:.4f})")
                        else:
                            st.info("ℹ️ Uji normalitas tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

                        st.markdown("**Uji Autokorelasi (Durbin-Watson):**")
                        dw_stat = residuals_data.get('dw_statistic', np.nan)
                        if pd.notnull(dw_stat) and np.isfinite(dw_stat):
                            interpretation = "Tidak ada autokorelasi signifikan"
                            emoji = "✅"
                            if not (1.5 <= dw_stat <= 2.5): # General rule of thumb
                                interpretation = "Kemungkinan ada autokorelasi"
                                emoji = "⚠️"
                                if dw_stat < 1.5: interpretation += " positif."
                                elif dw_stat > 2.5: interpretation += " negatif."
                            st.markdown(f"{emoji} {interpretation} (DW = {dw_stat:.3f})")
                        else:
                            st.info("ℹ️ Uji Durbin-Watson tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

                    with col2:
                        st.markdown("**Uji Heteroskedastisitas (Pearson |resid| vs fitted):**")
                        hetero_p = residuals_data.get('hetero_p', np.nan)
                        if pd.notnull(hetero_p) and np.isfinite(hetero_p):
                            hetero_corr = residuals_data.get('hetero_correlation', np.nan)
                            corr_text = f", korelasi={hetero_corr:.3f}" if pd.notnull(hetero_corr) else ""
                            if hetero_p > 0.05:
                                st.success(f"✅ Varians residual kemungkinan homoskedastik (p = {hetero_p:.4f}{corr_text})")
                            else:
                                st.warning(f"⚠️ Varians residual kemungkinan heteroskedastik (p = {hetero_p:.4f}{corr_text})")
                        else:
                            st.info("ℹ️ Uji heteroskedastisitas tidak dapat dilakukan (data tidak cukup atau hasil tidak valid).")

                        st.markdown("**Status Validitas Model Biologis:**")
                        status = model_results.get('status', 'error')
                        if status == 'success':
                            st.success("✅ Model menghasilkan parameter biologis yang valid (Eopt & MSY > 0).")
                        elif status == 'warning':
                            st.warning("⚠️ Parameter biologis mungkin tidak sepenuhnya valid atau model menunjukkan beberapa ketidaksesuaian (misal Eopt/MSY <= 0 atau koefisien model tidak ideal).")
                        else: # error
                            st.error(f"❌ Model gagal menghasilkan parameter biologis yang valid. ({model_results.get('error', '')})")

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
                    missing_cols = [col for col in required_cols if col not in yearly_data_for_model.columns]
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
                    
                    # Display model comparison table
                    st.subheader("📋 Perbandingan Model")
                    comparison_df = create_model_comparison_table(schaefer_results, fox_results)
                    
                    df_to_style = comparison_df.set_index('Parameter')

                    def style_status_cells(val):
                        s_val = str(val) # Ensure it's a string
                        if '✅' in s_val: return 'background-color: #e8f5e8; color: black; font-weight: bold;'
                        elif '⚠️' in s_val: return 'background-color: #fff3cd; color: black; font-weight: bold;'
                        elif '❌' in s_val: return 'background-color: #f8d7da; color: black; font-weight: bold;'
                        return '' # Default no style

                    # Apply styling only to the 'Status Model' row for Schaefer and Fox columns
                    # Pandas Styler can be tricky for cell-specific styling based on row AND column.
                    # Using applymap on a subset is cleaner if applicable.
                    # Here, we can format the DataFrame to HTML for more control or use a simpler Styler.
                    # For simplicity with Styler, we can make it apply to all cells in Schaefer/Fox columns and let the conditions handle specificity.
                    
                    styled_df = df_to_style.style.apply(
                        lambda x: x.map(style_status_cells) if x.name in ['Schaefer', 'Fox'] else x, axis=0
                    )
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Model diagnostics tabs
                    st.markdown("---")
                    st.subheader("🔍 Diagnostik dan Analisis Residual Rinci")
                    tab_schaefer, tab_fox = st.tabs(["Diagnostik Schaefer", "Diagnostik Fox"])

                    with tab_schaefer:
                        display_diagnostic_summary(schaefer_results, 'Schaefer')
                        if schaefer_results.get('status') != 'error':
                            fig_residuals_s = create_residual_plots(schaefer_results, 'Schaefer')
                            if fig_residuals_s:
                                st.plotly_chart(fig_residuals_s, use_container_width=True)
                            else:
                                st.info("Plot residual untuk Schaefer tidak dapat ditampilkan.")
                    
                    with tab_fox:
                        display_diagnostic_summary(fox_results, 'Fox')
                        if fox_results.get('status') != 'error':
                            fig_residuals_f = create_residual_plots(fox_results, 'Fox')
                            if fig_residuals_f:
                                st.plotly_chart(fig_residuals_f, use_container_width=True)
                            else:
                                st.info("Plot residual untuk Fox tidak dapat ditampilkan.")
                        
                    # Model selection for visualization
                    st.markdown("---")
                    st.subheader("📈 Visualisasi Model dan Rekomendasi")
                    
                    valid_models_options = {}
                    if schaefer_results.get('status') in ['success', 'warning'] and schaefer_results.get('biological', {}).get('Eopt') is not None:
                        s_r2 = schaefer_results.get('metrics', {}).get('R2', np.nan)
                        s_eopt = schaefer_results.get('biological', {}).get('Eopt', np.nan)
                        if pd.notnull(s_r2) and pd.notnull(s_eopt) and s_eopt > 0 : # Ensure R2 is valid and Eopt is positive
                            valid_models_options[f"Schaefer (R²={s_r2:.3f}, Eopt={s_eopt:.1f})"] = ('Schaefer', schaefer_results)
                    
                    if fox_results.get('status') in ['success', 'warning'] and fox_results.get('biological', {}).get('Eopt') is not None:
                        f_r2_orig = fox_results.get('metrics_original', {}).get('R2', np.nan)
                        f_eopt = fox_results.get('biological', {}).get('Eopt', np.nan)
                        if pd.notnull(f_r2_orig) and pd.notnull(f_eopt) and f_eopt > 0 : # Ensure R2 is valid and Eopt is positive
                            valid_models_options[f"Fox (R² Ori={f_r2_orig:.3f}, Eopt={f_eopt:.1f})"] = ('Fox', fox_results)
                    
                    if not valid_models_options:
                        st.error("❌ Tidak ada model yang valid (dengan Eopt > 0) untuk visualisasi dan rekomendasi.")
                        return
                    
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
                        fig_surplus, fig_cpue = create_model_visualization(model_results_selected, yearly_data_for_model, model_name)
                        
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
                                exploitation_ratio_catch = current_catch / cmsy if cmsy > 0 else np.nan # More for context than primary indicator

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
                                    elif exploitation_ratio_effort <= 1.2: # (0.8 to 1.2)
                                        status_text = "**Fully-exploited (Dimanfaatkan Penuh)**"
                                        recommendation_text = ("Upaya penangkapan saat ini berada pada atau mendekati tingkat optimal. "
                                                            "Pertahankan atau lakukan penyesuaian kecil pada upaya. Fokus pada monitoring stok, "
                                                            "efisiensi, dan keberlanjutan.")
                                        status_color = "orange"
                                        status_icon = "🎣"
                                    else: # > 1.2
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
                                st.metric(label=f"Status Eksploitasi (Upaya)", value=status_text.split('(')[0].replace("*",""), delta=f"{exploitation_ratio_effort:.2f} (E/Eopt)", delta_color="off")
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
                        f_metrics = fox_results.get('metrics_original', {})
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
                                status_color = '#f0f2f6' # Warna default netral
                                rekomendasi = "Perlu analisis lebih lanjut atau data yang lebih representatif."

                                # Logika berdasarkan tabel yang Anda berikan
                                if pd.notnull(catch_ratio_to_msy) and pd.notnull(effort_ratio_to_eopt):
                                    # Kategori 1: Over-exploited (C/Cmsy ≥ 1)
                                    if catch_ratio_to_msy >= 1.0:
                                        tingkat_exploitasi_desc = "Over-exploited (C/Cmsy ≥ 1)"
                                        if effort_ratio_to_eopt < 1.0:
                                            status_stok_ikan = "Healthy Stock 👍"
                                            tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                                            status_color = "#28a745" # Hijau
                                            rekomendasi = "Stok sehat namun upaya sudah tinggi relatif terhadap hasil. Pertimbangkan untuk tidak menambah upaya agar tetap optimal dan mencegah overfishing di masa depan."
                                        else: # E/Eopt ≥ 1
                                            status_stok_ikan = "Depleting Stock 📉"
                                            tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                                            status_color = "#dc3545" # Merah
                                            rekomendasi = "Stok mengalami penurunan akibat penangkapan berlebih. Segera kurangi upaya penangkapan secara signifikan untuk pemulihan."
                                    # Kategori 2: Fully-exploited (0.5 ≤ C/Cmsy < 1)
                                    elif 0.5 <= catch_ratio_to_msy < 1.0:
                                        tingkat_exploitasi_desc = "Fully-exploited (0.5 ≤ C/Cmsy < 1)"
                                        if effort_ratio_to_eopt < 1.0:
                                            status_stok_ikan = "Recovery Stock ↗️"
                                            tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                                            status_color = "#17a2b8" # Biru muda/info
                                            rekomendasi = "Stok dalam tahap pemulihan dengan upaya di bawah optimal. Upaya dapat ditingkatkan secara hati-hati menuju E_opt dengan monitoring ketat."
                                        else: # E/Eopt ≥ 1
                                            status_stok_ikan = "Overfishing Stock 🎣⚠️"
                                            tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                                            status_color = "#ffc107" # Kuning/peringatan
                                            rekomendasi = "Upaya penangkapan sudah melebihi batas optimal meskipun hasil tangkapan belum menunjukkan over-eksploitasi penuh. Kurangi upaya untuk menjaga keberlanjutan."
                                    # Kategori 3: Moderate exploited (C/Cmsy < 0.5) - Menggabungkan dua baris terakhir Anda
                                    elif catch_ratio_to_msy < 0.5:
                                        if catch_ratio_to_msy > 0.2: # (0.2 < C/Cmsy < 0.5)
                                            tingkat_exploitasi_desc = "Moderate exploited (0.2 < C/Cmsy < 0.5)"
                                        else: # (C/Cmsy ≤ 0.2)
                                            tingkat_exploitasi_desc = "Low exploited / Potentially Collapsed (C/Cmsy ≤ 0.2)"
                                        
                                        if effort_ratio_to_eopt < 1.0: # Underfishing (E/Eopt < 1)
                                            status_stok_ikan = "Transitional Recovery Stock / Potensi Besar ⬆️"
                                            tingkat_upaya_desc = "Underfishing (E/Eopt < 1)"
                                            status_color = "#007bff" # Biru primer
                                            rekomendasi = "Hasil tangkapan masih rendah, namun upaya juga rendah. Ada potensi besar untuk peningkatan jika stok dapat pulih atau memang belum dimanfaatkan optimal. Perlu investigasi lebih lanjut terhadap kondisi stok."
                                        else: # Overfishing (E/Eopt ≥ 1)
                                            if catch_ratio_to_msy <= 0.2:
                                                status_stok_ikan = "Collapsed Stock 💔"
                                                tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                                                status_color = "#6c757d" # Abu-abu tua/gelap
                                                rekomendasi = "Stok kemungkinan besar telah kolaps akibat tekanan penangkapan yang tinggi di masa lalu atau saat ini. Perlu tindakan drastis seperti moratorium dan rencana pemulihan jangka panjang."
                                            else: # (0.2 < C/Cmsy < 0.5) and Overfishing
                                                status_stok_ikan = "Overfishing Stock (Risiko Tinggi) 🎣📉"
                                                tingkat_upaya_desc = "Overfishing (E/Eopt ≥ 1)"
                                                status_color = "#fd7e14" # Oranye
                                                rekomendasi = "Upaya penangkapan berlebih pada stok yang sudah moderat. Sangat berisiko. Segera kurangi upaya penangkapan."
                                    else: # Jika tidak masuk kategori di atas (seharusnya tidak terjadi jika logika C/Cmsy sudah mencakup semua)
                                        status_stok_ikan = "Status Tidak Terdefinisi ❓"
                                        status_color = '#6c757d' # Abu-abu
                                        rekomendasi = "Kombinasi rasio C/Cmsy dan E/Eopt tidak masuk dalam kategori standar. Perlu review data dan model."

                                # Tampilkan status dengan card yang lebih informatif
                                st.markdown(f"""
                                <div style="padding:20px; border-radius:10px; background-color:{status_color}; color:{'white' if status_color not in ['#f0f2f6', '#ffc107', '#17a2b8'] else 'black'}; border-left:8px solid {'darkgrey' if status_color == '#f0f2f6' else darken_color(status_color, 0.2) }; margin-bottom:20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
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
                                    <hr style="border-top: 1px solid rgba(255,255,255,0.3) if '{'white' if status_color not in ['#f0f2f6', '#ffc107', '#17a2b8'] else 'black'}' == 'white' else 'rgba(0,0,0,0.1)'; margin: 15px 0;">
                                    <p style="margin-bottom:0; font-weight:bold; color:inherit;">Rekomendasi Umum:</p>
                                    <p style="margin-top:5px; color:inherit; opacity:0.95;">{rekomendasi}</p>
                                </div>
                                """, unsafe_allow_html=True)

                            else:
                                st.warning("Tidak cukup data tahunan untuk menghitung status pemanfaatan stok berdasarkan data terkini.")
                        else:
                            st.error("Tidak ada model (Schaefer atau Fox) yang menghasilkan parameter MSY dan E-opt yang valid (positif) untuk penentuan status pemanfaatan.")