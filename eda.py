import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    df = pd.read_csv('datasets/train_data.csv')
    print("Đọc dữ liệu thành công!")
    print(f"Tổng số dòng dữ liệu: {df.shape[0]}")
except FileNotFoundError:
    print("Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn.")
    exit()

aspects = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']

label_mapping = {-1: 'Không đề cập', 0: 'Tiêu cực', 1: 'Tích cực', 2: 'Trung tính'}

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, aspect in enumerate(aspects):
    counts = df[aspect].value_counts().rename(index=label_mapping)
    
    sns.barplot(x=counts.index, y=counts.values, ax=axes[i], palette='Set2')

    axes[i].set_title(f'Phân phối nhãn: {aspect}', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Số lượng câu', fontsize=12)
    axes[i].set_xlabel('')

    for p in axes[i].patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            axes[i].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=11, color='black', 
                             xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
os.mkdir('./images')
plt.savefig('images/phan_phoi_du_lieu_absa.png', dpi=300, bbox_inches='tight')
print("Đã lưu biểu đồ thành file 'phan_phoi_du_lieu_absa.png'.")
plt.show()

print("\n--- Kiểm tra dữ liệu bị thiếu (Null/NaN) ---")
print(df.isnull().sum())