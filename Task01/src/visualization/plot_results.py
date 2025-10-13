import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_results(feature_names, results_path='results/qc_prediction_results.json'):
    """Create publication-quality visualizations comparing model performance"""
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_context("talk")
    sns.set_palette("husl")
    
    # Metrics to plot
    metrics = ['accuracy', 'f1', 'f2', 'precision', 'recall']
    metric_names = ['Accuracy', 'F1 Score', 'F2 Score', 'Precision', 'Recall']
    
    # Create separate figure for each model and metric
    for model in results.keys():
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 15))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 2])
            
            # Main metric plot
            ax = fig.add_subplot(gs[0])
            
            # Prepare data for plotting
            plot_data = []
            for category in results[model].keys():
                for data_type in ['all', 'raw']:
                    mean_score = np.mean(
                        results[model][category][data_type][metric]
                    )
                    std_score = np.std(
                        results[model][category][data_type][metric]
                    )
                    plot_data.append({
                        'Category': category,
                        'Data Type': data_type.capitalize(),
                        'Score': mean_score,
                        'Std': std_score
                    })
            
            # Convert to DataFrame
            df = pd.DataFrame(plot_data)
            
            # Create grouped bar plot
            sns.barplot(
                data=df,
                x='Category',
                y='Score',
                hue='Data Type',
                ax=ax,
                capsize=0.1,
                errorbar='sd'
            )
            
            # Customize plot
            ax.set_title(f'{model.upper()} - {metric_name} Comparison', pad=20, fontsize=14)
            ax.set_xlabel('QC Category', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(
                title='Data Type',
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
            
            # Add value labels on top of bars with smaller font
            for container in ax.containers:
                ax.bar_label(
                    container,
                    fmt='%.3f',
                )
            
            # Feature importance plot (only for tree-based models)
            ax_imp = fig.add_subplot(gs[1])
            
            if model in ['rf', 'xgb']:
                # Get feature importance from the first category
                first_category = list(results[model].keys())[0]
                feature_importance = results[model][first_category]['all'][
                    'feature_importance'
                ]
                
                # Create feature importance plot
                importance_df = pd.DataFrame({
                    'Feature': [
                        f'{feature_names[i]}' for i in range(len(feature_importance))
                    ],
                    'Importance': feature_importance
                })
                importance_df = importance_df.sort_values(
                    'Importance',
                    ascending=False
                ).head(10)
                
                sns.barplot(
                    data=importance_df,
                    x='Importance',
                    y='Feature',
                    ax=ax_imp,
                    palette='viridis'
                )
                
                ax_imp.set_title(
                    f'Top 10 Feature Importance\n({model.upper()} - All Data)',
                    pad=20,
                    fontsize=14
                )
                ax_imp.set_xlabel('Importance Score', fontsize=12)
                ax_imp.set_ylabel('Feature', fontsize=12)
            else:
                # For non-tree models, remove the feature importance subplot
                ax_imp.remove()
                gs.update(hspace=0.3)  # Adjust spacing
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(
                f'results/{model}_comparison_{metric}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
    
    print("Visualizations saved to results/*_comparison_*.png")


if __name__ == "__main__":
    visualize_results() 