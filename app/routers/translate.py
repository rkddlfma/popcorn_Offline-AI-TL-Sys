from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/translate", tags=["translate"])


class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "ko"
    tgt_lang: str = "en"
    glossary: dict[str, str] | None = None


class TranslateResponse(BaseModel):
    translated: str
    src_lang: str
    tgt_lang: str


@router.post("", response_model=TranslateResponse)
async def translate(body: TranslateRequest, request: Request):
    """채팅/공지 번역 API. 짧은 문장 중심의 다국어 번역."""
    translation_service = request.app.state.translation_service
    result = await translation_service.translate(
        body.text, body.src_lang, body.tgt_lang, body.glossary
    )
    return TranslateResponse(translated=result, src_lang=body.src_lang, tgt_lang=body.tgt_lang)
