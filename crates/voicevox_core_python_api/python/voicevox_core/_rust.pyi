from typing import Final, List, TypedDict

from voicevox_core import AudioQuery

METAS: Final[List["Meta"]]
SUPPORTED_DEVICES: Final["SupportedDevices"]

class Meta(TypedDict):
    name: str
    styles: List["Style"]
    speaker_uuid: str
    version: str

class Style(TypedDict):
    name: str
    id: int

class SupportedDevices(TypedDict):
    cpu: bool
    cuda: bool
    dml: bool

class VoicevoxCore:
    def __init__(
        self, use_gpu: bool, cpu_num_threads: int, load_all_models: bool
    ) -> None:
        pass
    def __repr__(self) -> str:
        pass
    def load_model(self, speaker_id: int) -> None:
        pass
    def is_model_loaded(self, speaker_id: int) -> bool:
        pass
    def load_openjtalk_dict(self, dict_path: str) -> None:
        pass
    def audio_query(self, text: str, speaker_id: int) -> "AudioQuery":
        pass
    def audio_query_from_kana(self, text: str, speaker_id: int) -> "AudioQuery":
        pass
    def synthesis(self, audio_query: "AudioQuery", speaker_id: int) -> bytes:
        pass
    def tts(self, text: str, speaker_id: int) -> bytes:
        pass
    def tts_from_kana(self, text: str, speaker_id: int) -> bytes:
        pass
