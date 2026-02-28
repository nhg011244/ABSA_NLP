import streamlit as st
import torch
import re
from underthesea import word_tokenize
from transformers import AutoTokenizer
import pandas as pd

from model.pho_bert import PhoBERT_ABSA

st.set_page_config(page_title="Demo ABSA - PhoBERT", page_icon="🛍️", layout="centered")

st.title("🛍️ Hệ thống Phân tích Đánh giá Đa khía cạnh")
st.markdown("""
Hệ thống sử dụng mô hình **PhoBERT** để tự động bóc tách cảm xúc khách hàng thành 8 khía cạnh độc lập. 
Dự án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên.
""")

teen_code_dict = {
    "sp": "sản phẩm", "sz": "kích cỡ", "size": "kích cỡ", "đc": "được",
    "k": "không", "ko": "không", "kh": "không", "auth": "chính hãng",
    "rep": "hàng giả", "đẹp": "đẹp", "okela": "tốt", "oke": "tốt",
    "ok": "tốt", "tl": "trả lời", "ib": "nhắn tin", "shop": "cửa hàng",
    "nv": "nhân viên", "ship": "giao hàng", "shipper": "người giao hàng",
    "bt": "bình thường", "vs": "với"
}

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    standardized_words = [teen_code_dict.get(word, word) for word in words]
    text = ' '.join(standardized_words)
    text = re.sub(r'\s+', ' ', text).strip()
    return word_tokenize(text, format="text")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhoBERT_ABSA().to(device)
    
    model.load_state_dict(torch.load('saved_models/phobert_absa_weights.pth', map_location=device))
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

user_input = st.text_area("💬 Nhập bình luận của khách hàng vào đây:", 
                          value="giày xịn form ôm chân đẹp, nhưng shop rep tin nhắn chậm quá", 
                          height=100)

if st.button("🚀 Phân Tích Ngay"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập bình luận!")
    else:
        with st.spinner('Mô hình đang suy nghĩ...'):
            cleaned_text = preprocess_text(user_input)
            
            encoding = tokenizer(
                cleaned_text, add_special_tokens=True, max_length=128,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            aspects = ['Price (Giá cả)', 'Shipping (Giao hàng)', 'Outlook (Hình thức)', 
                       'Quality (Chất lượng)', 'Size (Kích cỡ)', 'Shop_Service (Dịch vụ)', 
                       'General (Đánh giá chung)', 'Others (Khác)']
            
            label_map = {
                0: ('Tiêu cực', '🔴'), 
                1: ('Tích cực', '🟢'), 
                2: ('Trung tính', '🟡'), 
                3: ('Không đề cập', '⚪')
            }
            
            st.success("✅ Phân tích hoàn tất!")
            st.markdown(f"**Văn bản sau khi làm sạch (Dành cho kiểm tra):** `{cleaned_text}`")
            
            results_data = []
            found_aspects = False
            
            for i, aspect in enumerate(aspects):
                pred_label = torch.argmax(outputs[i], dim=1).item()
                if pred_label != 3:
                    found_aspects = True
                    text_label, icon = label_map[pred_label]
                    results_data.append({"Khía cạnh": aspect, "Đánh giá": f"{icon} {text_label}"})
            
            if found_aspects:
                df_results = pd.DataFrame(results_data)
                st.table(df_results)
            else:
                st.info("Mô hình không nhận diện được khía cạnh khen/chê nào rõ ràng trong câu này.")