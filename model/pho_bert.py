import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERT_ABSA(nn.Module):
    def __init__(self, num_aspects=8, num_classes=4):
        """
        Khởi tạo mô hình PhoBERT cho bài toán phân loại đa khía cạnh
        """
        super(PhoBERT_ABSA, self).__init__()
        
        # 1. Tải mô hình PhoBERT-base đã được pre-train từ Hugging Face
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        
        # Lấy kích thước vector đặc trưng đầu ra của PhoBERT (768 chiều)
        hidden_size = self.phobert.config.hidden_size 
        
        # 2. Lớp Dropout giúp giảm thiểu over-fitting trong quá trình huấn luyện
        self.dropout = nn.Dropout(0.3)
        
        # 3. Xây dựng 8 nhánh phân loại (Classification Heads) độc lập
        # Sử dụng nn.ModuleList để mạng nơ-ron có thể cập nhật gradient cho cả 8 nhánh
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        # Bước 1: Đưa văn bản vào phần thân PhoBERT
        outputs = self.phobert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Bước 2: Trích xuất vector đại diện của token [CLS] (pooler_output)
        pooled_output = outputs.pooler_output 
        pooled_output = self.dropout(pooled_output)
        
        # Bước 3: Phân loại song song cho 8 khía cạnh
        # Trả về một danh sách gồm 8 tensor logits, mỗi tensor có shape (batch_size, 4)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        
        return logits