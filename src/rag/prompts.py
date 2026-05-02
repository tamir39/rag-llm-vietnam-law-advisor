"""Vietnamese QA prompt templates (with and without retrieved context)."""
from __future__ import annotations

SYSTEM_VI = (
    "Bạn là trợ lý pháp luật chuyên về Luật Thuế giá trị gia tăng của Việt Nam. "
    "Trả lời ngắn gọn, chính xác, dựa trên căn cứ pháp lý khi có."
)

QA_NO_RAG = """{system}

Câu hỏi: {question}
Trả lời:"""

QA_WITH_RAG = """{system}

Ngữ cảnh tham khảo:
{context}

Câu hỏi: {question}
Hãy trả lời dựa vào ngữ cảnh tham khảo. Nếu ngữ cảnh không đủ, hãy nói rõ.
Trả lời:"""
