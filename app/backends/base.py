from abc import ABC, abstractmethod


class TranslationBackend(ABC):
    """번역 백엔드 추상 인터페이스.
    VLLM 전환 시 이 클래스를 상속한 VLLMBackend로 교체만 하면 됩니다.
    """

    @abstractmethod
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """텍스트를 번역하여 반환."""
        ...

    @abstractmethod
    def load(self) -> None:
        """모델 로드. 앱 시작 시 한 번 호출됩니다."""
        ...
