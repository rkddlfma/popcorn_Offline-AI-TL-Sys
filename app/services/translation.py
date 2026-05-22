from app.backends.base import TranslationBackend


class TranslationService:
    def __init__(self, backend: TranslationBackend):
        self._backend = backend

    async def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """텍스트를 번역합니다. glossary가 있으면 프롬프트에 삽입합니다."""
        if glossary:
            text = self._apply_glossary_hint(text, glossary)
        return await self._backend.translate(text, src_lang, tgt_lang)

    def _apply_glossary_hint(self, text: str, glossary: dict[str, str]) -> str:
        # 원문에 실제로 등장하는 용어만 필터링
        matched = {k: v for k, v in glossary.items() if k in text}
        if not matched:
            return text
        terms = ", ".join(f"{k}→{v}" for k, v in matched.items())
        return f"[용어집: {terms}]\n{text}"
