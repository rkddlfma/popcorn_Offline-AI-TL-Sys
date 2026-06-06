from abc import ABC, abstractmethod
from typing import AsyncIterator


class TranslationBackend(ABC):
    """번역 백엔드 추상 인터페이스.
    VLLM 전환 시 이 클래스를 상속한 VLLMBackend로 교체만 하면 됩니다.
    """

    @abstractmethod
    async def translate(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> str:
        """텍스트를 번역하여 반환.

        priority="high"는 실시간 자막처럼 지연에 민감한 요청으로,
        배치 요청보다 GPU를 먼저 사용합니다.
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """모델 로드. 앱 시작 시 한 번 호출됩니다."""
        ...

    async def translate_stream(
        self, text: str, src_lang: str, tgt_lang: str, priority: str = "normal"
    ) -> AsyncIterator[str]:
        """번역을 점진적으로 yield (누적 문자열).

        기본 구현은 스트리밍 미지원 백엔드를 위해 전체 결과를 한 번에 yield합니다.
        """
        yield await self.translate(text, src_lang, tgt_lang, priority)
