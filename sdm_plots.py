import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
})

CLASS_COLORS = ['#e0e0e0', '#a6d96a', '#fdae61', '#d7191c']

def get_class_labels(lang):
    if lang == 'vi': return ['Không thích hợp', 'Ít thích hợp', 'Thích hợp', 'Rất thích hợp']
    return ['Unsuitable', 'Poorly suitable', 'Moderately suitable', 'Highly suitable']

def plot_correlation(X, lang='en'):
    title = "Mối tương quan giữa các biến môi trường" if lang == 'vi' else "Environmental variables correlation matrix"
    label = "Hệ số tương quan Pearson" if lang == 'vi' else "Pearson correlation"
    
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmax=1, vmin=-1, 
                square=True, linewidths=0.5, linecolor='black', 
                cbar_kws={"shrink": .7, "label": label}, ax=ax)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
        
    ax.set_title(title, weight='bold', fontsize=14, pad=15)
    return fig

def plot_vif_selection(vif_df, threshold, lang='en'):
    title = 'Đánh giá đa cộng tuyến' if lang == 'vi' else 'Multicollinearity assessment'
    xlabel = 'Hệ số phóng đại phương sai (thang log)' if lang == 'vi' else 'Variance Inflation Factor (log scale)'
    retained_lbl = 'Biến giữ lại' if lang == 'vi' else 'Retained'
    dropped_lbl = 'Biến loại bỏ' if lang == 'vi' else 'Dropped'
    
    vif_df['Status_Lang'] = np.where(vif_df['Status'] == 'Retained', retained_lbl, dropped_lbl)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=vif_df, x='VIF', y='Feature', hue='Status_Lang', 
                palette={retained_lbl: '#2b83ba', dropped_lbl: '#d7191c'}, 
                edgecolor='black', linewidth=1, ax=ax)
    ax.set_xscale('log')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, label=f'Ngưỡng (VIF={threshold})' if lang=='vi' else f'Threshold (VIF={threshold})')
    ax.set(xlabel=xlabel, ylabel='', title=title)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    return fig

def plot_roc_curve(y_train, y_train_prob, y_test, y_test_prob, lang='en'):
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_prob)
    fpr_ts, tpr_ts, _ = roc_curve(y_test, y_test_prob)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    lbl_tr = f'Dữ liệu huấn luyện AUC = {auc(fpr_tr, tpr_tr):.3f}' if lang == 'vi' else f'Train AUC = {auc(fpr_tr, tpr_tr):.3f}'
    lbl_ts = f'Dữ liệu kiểm tra AUC = {auc(fpr_ts, tpr_ts):.3f}' if lang == 'vi' else f'Test AUC = {auc(fpr_ts, tpr_ts):.3f}'
    lbl_rd = 'Phân loại ngẫu nhiên' if lang == 'vi' else 'Random classifier'
    
    ax.plot(fpr_tr, tpr_tr, color='#d7191c', lw=2, label=lbl_tr)
    ax.plot(fpr_ts, tpr_ts, color='#2b83ba', lw=2, linestyle='--', label=lbl_ts)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label=lbl_rd)
    
    xlabel = '1 - Độ đặc hiệu (tỷ lệ dự đoán sai sự xuất hiện)' if lang == 'vi' else 'False Positive Rate'
    ylabel = 'Độ nhạy (tỷ lệ dự đoán đúng sự xuất hiện)' if lang == 'vi' else 'True Positive Rate'
    
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.05], xlabel=xlabel, ylabel=ylabel, title='Đường cong ROC' if lang=='vi' else 'ROC Curves')
    ax.legend(loc="lower right", frameon=True, edgecolor='black')
    return fig

def plot_feature_importance(importances, features, lang='en'):
    idx = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
    
    ax.barh(range(len(idx)), importances[idx], color='#2c7bb6', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    
    xlabel = 'Mức độ đóng góp tương đối' if lang == 'vi' else 'Relative contribution'
    title = 'Tầm quan trọng của biến' if lang == 'vi' else 'Feature importance'
    
    ax.set(xlabel=xlabel, title=title)
    return fig

def plot_violin_classes(df_plot, features, thresholds, lang='en'):
    t1, t2, t3 = thresholds
    conds = [(df_plot['Suit_Prob'] < t1), (df_plot['Suit_Prob'] >= t1) & (df_plot['Suit_Prob'] < t2), 
             (df_plot['Suit_Prob'] >= t2) & (df_plot['Suit_Prob'] < t3), (df_plot['Suit_Prob'] >= t3)]
    
    c_labels = get_class_labels(lang)
    df_plot['Class'] = pd.Categorical(np.select(conds, c_labels, default=c_labels[0]), categories=c_labels, ordered=True)
    palette = dict(zip(c_labels, CLASS_COLORS))
    
    cols = min(3, len(features))
    import math
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    num_feats = len(features)
    
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.violinplot(data=df_plot, x='Class', y=feat, ax=ax, palette=palette, inner="quartile", linewidth=1.2)
        means = df_plot.groupby('Class', observed=False)[feat].mean()
        ax.scatter(x=range(len(c_labels)), y=means, color='black', marker='D', s=40, zorder=10)
        
        ax.set_title(feat, weight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Chỉ hiển thị Tên trục X chéo 45 độ ở hàng dưới cùng
        if i + cols < num_feats:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks(range(len(c_labels)))
            ax.set_xticklabels(c_labels, rotation=45, ha='right')
        
    for j in range(num_feats, len(axes)): axes[j].set_visible(False)
        
    title = 'Phân bố giá trị biến môi trường theo cấp độ thích nghi' if lang == 'vi' else 'Environmental variable distribution across suitability classes'
    fig.suptitle(title, y=1.02, fontsize=15, weight='bold')
    plt.tight_layout()
    return fig

def plot_categorical_map(cat_map, extent, crs, world_shp_path, species_name, lang='en'):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap_cat = ListedColormap(CLASS_COLORS)
    
    ax.imshow(np.ma.masked_invalid(cat_map), cmap=cmap_cat, extent=extent, origin='upper')
    ax.set_ylim(-60, 90)  
    ax.set_xlim(-180, 180)
        
    ax.grid(True, linestyle=':', color='black', alpha=0.3, zorder=1)
    
    if world_shp_path:
        import geopandas as gpd
        gpd.read_file(world_shp_path).to_crs(crs).boundary.plot(ax=ax, linewidth=0.4, color='black', alpha=0.6, zorder=2)
        
    c_labels = get_class_labels(lang)
    
    # Hiển thị legend có khung vuông
    ax.legend(handles=[
        mpatches.Patch(color=CLASS_COLORS[0], label=c_labels[0]),
        mpatches.Patch(color=CLASS_COLORS[1], label=c_labels[1]),
        mpatches.Patch(color=CLASS_COLORS[2], label=c_labels[2]),
        mpatches.Patch(color=CLASS_COLORS[3], label=c_labels[3])
    ], loc='lower left', frameon=True, edgecolor='black', framealpha=1.0)
    
    title = f'Phân bố tiềm năng của {species_name}' if lang == 'vi' else f'Potential distribution of {species_name}'
    ax.set(xlabel='Kinh độ' if lang == 'vi' else 'Longitude', 
           ylabel='Vĩ độ' if lang == 'vi' else 'Latitude', 
           title=title)
    
    ax.set_facecolor('#ffffff')
    return fig
