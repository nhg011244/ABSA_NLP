import os
import pandas as pd
import re
from underthesea import word_tokenize

# 1. Khởi tạo từ điển chuẩn hóa Teencode
teen_code_dict = {
    "sp": "sản phẩm", "sz": "kích cỡ", "size": "kích cỡ", "đc": "được",
    "k": "không", "ko": "không", "kh": "không", "auth": "chính hãng",
    "rep": "hàng giả", "đẹp": "đẹp", "okela": "tốt", "oke": "tốt",
    "ok": "tốt", "tl": "trả lời", "ib": "nhắn tin", "shop": "cửa hàng",
    "nv": "nhân viên", "ship": "giao hàng", "shipper": "người giao hàng"
}

# 2. Hàm tiền xử lý văn bản
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    words = text.split()
    standardized_words = [teen_code_dict.get(word, word) for word in words]
    text = ' '.join(standardized_words)
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text, format="text")
    return text

# 3. Khai báo thư mục gốc và thư mục đích
input_dir = 'datasets'
output_dir = 'datasets_cleaned'

# Tạo thư mục datasets_cleaned nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục mới: {output_dir}/")

# Danh sách các file cần làm sạch (dựa trên cây thư mục của bạn)
files_to_clean = ['train_data.csv', 'val_data.csv', 'test_data.csv']

print("-" * 50)
print("BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH ĐỒNG LOẠT...")

# 4. Vòng lặp xử lý từng file
for file_name in files_to_clean:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    
    try:
        # Kiểm tra xem file có tồn tại trong thư mục datasets không
        if os.path.exists(input_path):
            print(f"Đang xử lý: {file_name} ...")
            
            # Đọc dữ liệu
            df = pd.read_csv(input_path)
            
            # Kiểm tra xem file có cột 'Review' không
            if 'Review' in df.columns:
                # Áp dụng hàm làm sạch và tạo cột Cleaned_Review
                df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
                
                # Lưu file sang thư mục mới
                df.to_csv(output_path, index=False)
                print(f" -> Xong! Đã lưu tại: {output_path}")
            else:
                print(f" -> CẢNH BÁO: File {file_name} không có cột 'Review'. Bỏ qua.")
        else:
            print(f" -> LỖI: Không tìm thấy file {input_path}")
            
    except Exception as e:
        print(f" -> LỖI khi xử lý {file_name}: {e}")

print("-" * 50)
print("HOÀN TẤT TOÀN BỘ!")