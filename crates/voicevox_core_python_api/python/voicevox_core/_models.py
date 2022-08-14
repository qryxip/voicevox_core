from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioQuery:
    accent_phrases: List["AccentPhrase"]
    speedScale: float
    pitchScale: float
    intonationScale: float
    volumeScale: float
    prePhonemeLength: float
    postPhonemeLength: float
    outputSamplingRate: int
    outputStereo: bool
    kana: Optional[str]


@dataclass
class AccentPhrase:
    moras: List["Mora"]
    accent: int
    pause_mora: Optional["Mora"]
    is_interrogative: bool


@dataclass
class Mora:
    text: str
    consonant: Optional[str]
    consonant_length: Optional[float]
    vowel: str
    vowel_length: float
    pitch: float
