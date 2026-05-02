"""Vietnamese QA prompt templates (with and without retrieved context).

Scope: 5 tax laws covered by the KB — Thuế GTGT (VAT), Thuế TNCN (PIT), Thuế
sử dụng đất phi nông nghiệp (Land), Thuế tiêu thụ đặc biệt (Excise), and Thuế
thu nhập doanh nghiệp (CIT).
"""
from __future__ import annotations

SYSTEM_VI = (
    "Bạn là trợ lý pháp luật chuyên về pháp luật thuế của Việt Nam, "
    "bao gồm: Thuế giá trị gia tăng (GTGT), Thuế thu nhập cá nhân (TNCN), "
    "Thuế sử dụng đất phi nông nghiệp, Thuế tiêu thụ đặc biệt (TTĐB), "
    "và Thuế thu nhập doanh nghiệp (TNDN). "
    "Trả lời ngắn gọn, chính xác, dựa trên căn cứ pháp lý khi có. "
    "Nếu không đủ căn cứ trong ngữ cảnh, hãy nói rõ là không đủ thông tin."
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


def build_no_rag_prompt(question: str, system: str = SYSTEM_VI) -> str:
    return QA_NO_RAG.format(system=system, question=question)


def build_rag_prompt(question: str, context: str, system: str = SYSTEM_VI) -> str:
    return QA_WITH_RAG.format(system=system, context=context, question=question)
