from typing import AsyncIterator

from app.backends.base import TranslationBackend


class TranslationService:
    def __init__(self, backend: TranslationBackend):
        self._backend = backend

    async def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        glossary: dict[str, str | dict[str, str]] | None = None,
        priority: str = "normal",
    ) -> str:
        """텍스트를 번역하고, 일치하는 용어는 목표 용어로 바꿉니다."""
        if glossary:
            text = self._apply_glossary_terms(text, glossary, tgt_lang)
        return await self._backend.translate(text, src_lang, tgt_lang, priority)

    async def translate_stream(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        glossary: dict[str, str | dict[str, str]] | None = None,
        priority: str = "normal",
    ) -> AsyncIterator[str]:
        """번역을 점진적으로(누적 문자열) yield합니다."""
        if glossary:
            text = self._apply_glossary_terms(text, glossary, tgt_lang)
        async for partial in self._backend.translate_stream(
            text, src_lang, tgt_lang, priority
        ):
            yield partial

    def _apply_glossary_terms(
        self,
        text: str,
        glossary: dict[str, str | dict[str, str]],
        tgt_lang: str,
    ) -> str:
        """Replace matched source terms in place; never add instructions to source text."""
        candidates = []
        for source, translations in glossary.items():
            if not isinstance(source, str) or not source:
                continue
            if isinstance(translations, dict):
                target = translations.get(tgt_lang)
            else:
                # 이전 형식의 단일 번역 용어집과 호환합니다.
                target = translations
            if not isinstance(target, str) or not target:
                continue

            start = 0
            while (start := text.find(source, start)) != -1:
                candidates.append((start, start + len(source), source, target))
                start += 1

        if not candidates:
            return text

        # 긴 용어를 먼저 선택해 겹치는 짧은 항목보다 우선하고,
        # 같은 용어가 원문에 여러 번 있으면 모두 치환합니다.
        selected = []
        for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
            start, end, _, _ = candidate
            if any(start < used_end and used_start < end
                   for used_start, used_end, _, _ in selected):
                continue
            selected.append(candidate)

        result = []
        cursor = 0
        for start, end, _, target in sorted(selected, key=lambda item: item[0]):
            result.extend((text[cursor:start], target))
            cursor = end
        result.append(text[cursor:])
        return "".join(result)
