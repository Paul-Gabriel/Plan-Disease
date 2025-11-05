import json
import plotly.graph_objects as go
import pandas as pd

# Load metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Extract class metrics
report = metrics["report"]
class_rows = {k: v for k, v in report.items() if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']}

# Create DataFrame
df = pd.DataFrame(class_rows).T
df = df[["precision", "recall", "f1-score"]]
df = df.rename(columns={"f1-score": "f1"})

# Create bar chart
fig = go.Figure()
colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

for i, col in enumerate(df.columns):
    fig.add_trace(go.Bar(
        name=col,
        x=df.index,
        y=df[col],
        text=[f"{v:.3f}" for v in df[col]],
        textposition='auto',
        marker_color=colors[i % len(colors)]
    ))

# Update layout
fig.update_layout(
    barmode="group",
    title={
        'text': "Per-class Metrics",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    yaxis_title="Score",
    yaxis_range=[0, 1],
    width=800,
    height=500,
    template="plotly_white",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Save as both standalone HTML and write the plot div
fig.write_html("test_bars.html", include_plotlyjs='cdn')
print("Plot HTML div:", fig.to_html(full_html=False, include_plotlyjs='cdn'))