import os
import json
import gradio as gr
import plotly.graph_objects as go
import pandas as pd

def load_metrics():
    """Load and validate metrics data."""
    if not os.path.exists("metrics.json"):
        raise FileNotFoundError("metrics.json not found")
        
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    report = metrics.get("report", {})
    if not report:
        raise ValueError("No 'report' data found in metrics.json")
    
    return metrics

def create_metrics_summary(metrics):
    """Create HTML for metrics summary."""
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Accuracy</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('accuracy', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Precision</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('precision', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Recall</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('recall', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">F1 Score</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('f1', 0):.4f}</div>
        </div>
    </div>
    """

def create_plot():
    """Generate a bar plot from metrics."""
    try:
        metrics = load_metrics()
        report = metrics["report"]
        
        # Extract class metrics (excluding averages)
        class_rows = {k: v for k, v in report.items() 
                     if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']}
        
        if not class_rows:
            return "No class-specific metrics found in the report"

        # Create DataFrame
        df = pd.DataFrame(class_rows).T
        metrics_to_plot = ["precision", "recall", "f1-score"]
        df = df[metrics_to_plot]
        
        # Create plot
        fig = go.Figure()
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
        
        for i, col in enumerate(df.columns):
            name = "F1-Score" if col == "f1-score" else col.title()
            fig.add_trace(go.Bar(
                name=name,
                x=df.index,
                y=df[col],
                text=[f"{v:.3f}" for v in df[col]],
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Per-class Performance Metrics",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Class",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            barmode='group',
            width=800,
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig
        
    except Exception as e:
        return f"Error generating plot: {str(e)}"

def show_report():
    """Generate the complete report."""
    try:
        metrics = load_metrics()
        
        content = create_metrics_summary(metrics)
        
        fig = create_plot()
        if isinstance(fig, str):  # Error message
            return content + f"<div style='color: red;'>{fig}</div>"
            
        return content + gr.Plot(value=fig, show_label=False)
        
    except Exception as e:
        return f"""
        <div style="color: red; padding: 20px; background: #fee; border-radius: 8px;">
            <h3>Error Generating Report</h3>
            <p>{str(e)}</p>
        </div>
        """

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # Bean Disease Classification Statistics
                
                This page shows the model's performance metrics and visualizations.
                Click the button below to load or refresh the report.
                """
            )
    
    # Initialize plot container
    plot_output = gr.Plot(label="Performance Metrics")
    metrics_html = gr.HTML()
    
    # Update button and status
    with gr.Row():
        update_btn = gr.Button("Show/Update Plot", variant="primary")
        status = gr.Markdown("")
    
    def update_display():
        try:
            metrics = load_metrics()
            html = create_metrics_summary(metrics)
            fig = create_plot()
            
            if isinstance(fig, str):  # Error message
                return html, None, f"Error: {fig}"
                
            return html, fig, "Report updated successfully"
            
        except Exception as e:
            return (
                f"<div style='color: red;'>Error loading metrics: {str(e)}</div>",
                None,
                f"Error: {str(e)}"
            )
    
    # Wire up update button and initialize display
    update_btn.click(
        fn=update_display,
        outputs=[metrics_html, plot_output, status]
    )
    
    # Initialize on page load
    metrics_html.value, plot_output.value, _ = update_display()

if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860, max_tries=100):
        for port in range(start_port, start_port + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        raise OSError(f"Could not find a free port in range {start_port}-{start_port + max_tries}")

    try:
        port = find_free_port(7860)
        print(f"Starting server on http://127.0.0.1:{port}")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")