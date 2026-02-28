import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttentionABSA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_aspects=8, num_classes=4):
        """
        Khởi tạo mô hình BiLSTM-Attention cho bài toán ABSA 8 khía cạnh.
        - num_aspects: 8 (Price, Shipping, Outlook, Quality, Size, Shop_Service, General, Others)
        - num_classes: 4 (-1: Không đề cập, 0: Tiêu cực, 1: Tích cực, 2: Trung tính)
        """
        super(BiLSTMAttentionABSA, self).__init__()
        
        # 1. Lớp Embedding biến đổi index của từ thành vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Lớp BiLSTM trích xuất ngữ cảnh 2 chiều
        # batch_first=True giúp đầu vào có dạng (batch_size, sequence_length, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # 3. Lớp tính điểm Attention
        # Kích thước đầu vào là hidden_dim * 2 vì BiLSTM ghép nối 2 chiều
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        
        # 4. Các lớp Phân loại (Classification Heads)
        # Sử dụng nn.ModuleList để tạo ra 8 lớp Linear độc lập cho 8 khía cạnh
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, x):
        # x: (batch_size, sequence_length)
        
        # Bước 1: Chuyển đổi từ thành vector
        embedded = self.embedding(x) 
        # Output: (batch_size, sequence_length, embedding_dim)
        
        # Bước 2: Đưa qua BiLSTM
        lstm_out, _ = self.lstm(embedded) 
        # Output: (batch_size, sequence_length, hidden_dim * 2)
        
        # Bước 3: Tính toán Attention
        attention_scores = self.attention_weights(lstm_out) 
        # Output: (batch_size, sequence_length, 1)
        
        # Sử dụng softmax để đưa điểm attention về khoảng (0, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Bước 4: Tạo Context Vector bằng cách nhân có trọng số đầu ra LSTM với Attention
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) 
        # Output: (batch_size, hidden_dim * 2)
        
        # Bước 5: Phân loại cho 8 khía cạnh
        # Truyền context_vector qua 8 lớp Linear độc lập
        outputs = [classifier(context_vector) for classifier in self.classifiers]
        
        # Trả về một list chứa 8 tensor, mỗi tensor có shape (batch_size, 4)
        return outputs