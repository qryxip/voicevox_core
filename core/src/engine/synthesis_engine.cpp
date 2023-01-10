#include "synthesis_engine.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "../core.h"
#include "full_context_label.h"
#include "mora_list.h"

namespace voicevox::core::engine {
std::vector<MoraModel> to_flatten_moras(std::vector<AccentPhraseModel> accent_phrases) {
  std::vector<MoraModel> flatten_moras;

  for (AccentPhraseModel accent_phrase : accent_phrases) {
    std::vector<MoraModel> moras = accent_phrase.moras;
    for (MoraModel mora : moras) {
      flatten_moras.push_back(mora);
    }
    if (accent_phrase.pause_mora) {
      MoraModel pause_mora = accent_phrase.pause_mora.value();
      flatten_moras.push_back(pause_mora);
    }
  }

  return flatten_moras;
}

std::vector<int64_t> to_phoneme_id_list(std::vector<std::string> phoneme_str_list) {
  std::vector<OjtPhoneme> phoneme_data_list;
  std::vector<int64_t> phoneme_id_list;
  for (size_t i = 0; i < phoneme_str_list.size(); i++) {
    phoneme_data_list.push_back(OjtPhoneme(phoneme_str_list[i], (float)i, (float)i + 1.0f));
  }
  phoneme_data_list = OjtPhoneme::convert(phoneme_data_list);
  for (OjtPhoneme phoneme_data : phoneme_data_list) {
    phoneme_id_list.push_back(phoneme_data.phoneme_id());
  }
  return phoneme_id_list;
}

std::vector<int64_t> to_accent_id_list(std::vector<std::string> accent_str_list) {
  std::vector<int64_t> accent_id_list;
  for (std::string accent_str : accent_str_list) {
    accent_id_list.push_back(Accent(accent_str).accent_id());
  }
  return accent_id_list;
}

void split_mora(std::vector<OjtPhoneme> phoneme_list, std::vector<OjtPhoneme> &consonant_phoneme_list,
                std::vector<OjtPhoneme> &vowel_phoneme_list, std::vector<int64_t> &vowel_indexes) {
  for (size_t i = 0; i < phoneme_list.size(); i++) {
    std::vector<std::string>::iterator result =
        std::find(mora_phoneme_list.begin(), mora_phoneme_list.end(), phoneme_list[i].phoneme);
    if (result != mora_phoneme_list.end()) {
      vowel_indexes.push_back((long)i);
    }
  }
  for (int64_t index : vowel_indexes) {
    vowel_phoneme_list.push_back(phoneme_list[index]);
  }
  consonant_phoneme_list.push_back(OjtPhoneme());
  for (size_t i = 0; i < vowel_indexes.size() - 1; i++) {
    int64_t prev = vowel_indexes[i];
    int64_t next = vowel_indexes[1 + i];
    if (next - prev == 1) {
      consonant_phoneme_list.push_back(OjtPhoneme());
    } else {
      consonant_phoneme_list.push_back(phoneme_list[next - 1]);
    }
  }
}

std::vector<AccentPhraseModel> adjust_interrogative_accent_phrases(std::vector<AccentPhraseModel> accent_phrases) {
  std::vector<AccentPhraseModel> new_accent_phrases(accent_phrases.size());
  for (size_t i = 0; i < accent_phrases.size(); i++) {
    AccentPhraseModel accent_phrase = accent_phrases[i];
    AccentPhraseModel new_accent_phrase = {
        adjust_interrogative_moras(accent_phrase),
        accent_phrase.accent,
        accent_phrase.pause_mora,
        accent_phrase.is_interrogative,
    };
    new_accent_phrases[i] = new_accent_phrase;
  }
  return new_accent_phrases;
}

std::vector<MoraModel> adjust_interrogative_moras(AccentPhraseModel accent_phrase) {
  std::vector<MoraModel> moras = accent_phrase.moras;
  if (accent_phrase.is_interrogative) {
    if (!moras.empty()) {
      MoraModel last_mora = moras[moras.size() - 1];
      float last_mora_pitch = last_mora.pitch;
      if (last_mora_pitch != 0.0) {
        std::vector<MoraModel> new_moras(moras.size() + 1);
        std::copy(moras.begin(), moras.end(), new_moras.begin());
        MoraModel interrogative_mora = make_interrogative_mora(last_mora);
        new_moras[moras.size()] = interrogative_mora;
        return new_moras;
      }
    }
  }
  return moras;
}

MoraModel make_interrogative_mora(MoraModel last_mora) {
  float fix_vowel_length = 0.15f;
  float adjust_pitch = 0.3f;
  float max_pitch = 6.5f;

  float pitch = last_mora.pitch + adjust_pitch;
  if (pitch > max_pitch) {
    pitch = max_pitch;
  }
  MoraModel interrogative_mora = {
      mora2text(last_mora.vowel), std::nullopt, std::nullopt, last_mora.vowel, fix_vowel_length, pitch,
  };
  return interrogative_mora;
}

std::vector<AccentPhraseModel> SynthesisEngine::create_accent_phrases(std::string text, int64_t *speaker_id) {
  if (text.empty()) {
    return {};
  }

  Utterance utterance = extract_full_context_label(m_openjtalk, text);
  if (utterance.breath_groups.empty()) {
    return {};
  }

  size_t accent_phrases_size = 0;
  for (const auto &breath_group : utterance.breath_groups) accent_phrases_size += breath_group.accent_phrases.size();
  std::vector<AccentPhraseModel> accent_phrases(accent_phrases_size);

  int accent_phrases_count = 0;
  for (size_t i = 0; i < utterance.breath_groups.size(); i++) {
    const auto &breath_group = utterance.breath_groups[i];
    for (size_t j = 0; j < breath_group.accent_phrases.size(); j++) {
      const auto &accent_phrase = breath_group.accent_phrases[j];

      std::vector<MoraModel> moras(accent_phrase.moras.size());
      for (size_t k = 0; k < accent_phrase.moras.size(); k++) {
        auto &mora = accent_phrase.moras[k];
        std::string moras_text = "";
        for (auto &phoneme : mora.phonemes()) moras_text += phoneme.phoneme();
        std::transform(moras_text.begin(), moras_text.end(), moras_text.begin(), ::tolower);
        if (moras_text == "n") moras_text = "N";
        std::optional<std::string> consonant = std::nullopt;
        std::optional<float> consonant_length = std::nullopt;
        if (mora.consonant) {
          consonant = mora.consonant.value().phoneme();
          consonant_length = 0.0f;
        }
        MoraModel new_mora = {
            mora2text(moras_text), consonant, consonant_length, mora.vowel.phoneme(), 0.0f, 0.0f,
        };
        moras[k] = new_mora;
      }

      std::optional<MoraModel> pause_mora = std::nullopt;
      if (i != utterance.breath_groups.size() - 1 && j == breath_group.accent_phrases.size() - 1) {
        pause_mora = {
            "、", std::nullopt, std::nullopt, "pau", 0.0f, 0.0f,
        };
      }
      AccentPhraseModel new_accent_phrase = {
          moras,
          accent_phrase.accent,
          pause_mora,
          accent_phrase.is_interrogative,
      };

      accent_phrases[accent_phrases_count] = new_accent_phrase;
      accent_phrases_count++;
    }
  }

  accent_phrases = replace_mora_data(accent_phrases, speaker_id);

  return accent_phrases;
}

std::vector<AccentPhraseModel> SynthesisEngine::replace_mora_data(std::vector<AccentPhraseModel> accent_phrases,
                                                                  int64_t *speaker_id) {
  std::vector<float> pitches;
  std::vector<AccentPhraseModel> changed_accent_phrases = replace_phoneme_length(accent_phrases, speaker_id, pitches);
  return replace_mora_pitch(changed_accent_phrases, speaker_id, pitches.data());
}

std::vector<AccentPhraseModel> SynthesisEngine::replace_phoneme_length(std::vector<AccentPhraseModel> accent_phrases,
                                                                       int64_t *speaker_id,
                                                                       std::vector<float> &pitches) {
  std::vector<MoraModel> flatten_moras;
  std::vector<int64_t> phoneme_id_list;
  std::vector<int64_t> accent_id_list;
  initial_process(accent_phrases, flatten_moras, phoneme_id_list, accent_id_list);

  std::vector<float> phoneme_length(phoneme_id_list.size(), 0.0);
  pitches.resize(phoneme_id_list.size());
  bool success = variance_forward((int64_t)phoneme_id_list.size(), phoneme_id_list.data(), accent_id_list.data(),
                                  speaker_id, pitches.data(), phoneme_length.data());

  if (!success) {
    throw std::runtime_error(last_error_message());
  }

  int index = 0;
  for (size_t i = 0; i < accent_phrases.size(); i++) {
    AccentPhraseModel accent_phrase = accent_phrases[i];
    std::vector<MoraModel> moras = accent_phrase.moras;
    for (size_t j = 0; j < moras.size(); j++) {
      MoraModel mora = moras[j];
      if (mora.consonant) {
        mora.consonant_length = phoneme_length[index];
        index++;
      }
      mora.vowel_length = phoneme_length[index];
      index++;
      moras[j] = mora;
    }
    accent_phrase.moras = moras;
    if (accent_phrase.pause_mora) {
      std::optional<MoraModel> pause_mora = accent_phrase.pause_mora;
      pause_mora->vowel_length = phoneme_length[index];
      index++;
      accent_phrase.pause_mora = pause_mora;
    }
    accent_phrases[i] = accent_phrase;
  }

  return accent_phrases;
}

std::vector<AccentPhraseModel> SynthesisEngine::replace_mora_pitch(std::vector<AccentPhraseModel> accent_phrases,
                                                                   int64_t *speaker_id, float *before_pitches) {
  std::vector<MoraModel> flatten_moras;
  std::vector<int64_t> phoneme_id_list;
  std::vector<int64_t> accent_id_list;
  initial_process(accent_phrases, flatten_moras, phoneme_id_list, accent_id_list);

  int64_t length = phoneme_id_list.size();
  std::vector<float> pitches(length, 0);
  if (before_pitches == nullptr) {
    std::vector<float> durations(length, 0);
    bool success = variance_forward((int64_t)phoneme_id_list.size(), phoneme_id_list.data(), accent_id_list.data(),
                                    speaker_id, pitches.data(), durations.data());

    if (!success) {
      throw std::runtime_error(last_error_message());
    }
  } else {
    for (int64_t i = 0; i < length; i++) {
      pitches[i] = before_pitches[i];
    }
  }

  int index = 0;
  for (size_t i = 0; i < accent_phrases.size(); i++) {
    AccentPhraseModel accent_phrase = accent_phrases[i];
    std::vector<MoraModel> moras = accent_phrase.moras;
    for (size_t j = 0; j < moras.size(); j++) {
      MoraModel mora = moras[j];
      if (mora.consonant.has_value()) index++;
      std::vector<std::string>::iterator found_unvoice_mora =
          std::find(unvoiced_mora_phoneme_list.begin(), unvoiced_mora_phoneme_list.end(), mora.vowel);
      if (found_unvoice_mora != unvoiced_mora_phoneme_list.end()) {
        mora.pitch = 0.0f;
      } else {
        mora.pitch = pitches[index];
      }
      index++;
      moras[j] = mora;
    }
    accent_phrase.moras = moras;
    if (accent_phrase.pause_mora) {
      std::optional<MoraModel> pause_mora = accent_phrase.pause_mora;
      pause_mora->pitch = 0;
      index++;
      accent_phrase.pause_mora = pause_mora;
    }
    accent_phrases[i] = accent_phrase;
  }

  return accent_phrases;
}

std::vector<uint8_t> SynthesisEngine::synthesis_wave_format(AudioQueryModel query, int64_t *speaker_id,
                                                            int *binary_size, bool enable_interrogative_upspeak) {
  std::vector<float> wave = synthesis(query, speaker_id, enable_interrogative_upspeak);

  float volume_scale = query.volume_scale;
  bool output_stereo = query.output_stereo;
  // TODO: 44.1kHzなどの対応
  unsigned int output_sampling_rate = query.output_sampling_rate;

  char num_channels = output_stereo ? 2 : 1;
  char bit_depth = 16;
  uint32_t repeat_count = (output_sampling_rate / default_sampling_rate) * num_channels;
  char block_size = bit_depth * num_channels / 8;

  std::stringstream ss;
  ss.write("RIFF", 4);
  int bytes_size = (int)wave.size() * repeat_count * 2;
  const int wave_size = bytes_size + 44;
  int chunk_size = wave_size - 8;
  for (int i = 0; i < 4; i++) {
    ss.put((uint8_t)(chunk_size & 0xff));  // chunk size
    chunk_size >>= 8;
  }
  ss.write("WAVEfmt ", 8);

  ss.put((char)16);                                // fmt header length
  for (int i = 0; i < 3; i++) ss.put((uint8_t)0);  // fmt header length
  ss.put(1);                                       // linear PCM
  ss.put(0);                                       // linear PCM
  ss.put(num_channels);                            // channnel
  ss.put(0);                                       // channnel

  int sampling_rate = output_sampling_rate;
  for (int i = 0; i < 4; i++) {
    ss.put((char)(sampling_rate & 0xff));
    sampling_rate >>= 8;
  }
  int block_rate = output_sampling_rate * block_size;
  for (int i = 0; i < 4; i++) {
    ss.put((char)(block_rate & 0xff));
    block_rate >>= 8;
  }

  ss.put(block_size);
  ss.put(0);
  ss.put(bit_depth);
  ss.put(0);

  ss.write("data", 4);
  for (int i = 0; i < 4; i++) {
    ss.put((char)(bytes_size & 0xff));
    bytes_size >>= 8;
  }

  for (size_t i = 0; i < wave.size(); i++) {
    float v = wave[i] * volume_scale;
    // clip
    v = 1.0f < v ? 1.0f : v;
    v = -1.0f > v ? -1.0f : v;
    int16_t data = (int16_t)(v * (float)0x7fff);
    for (uint32_t j = 0; j < repeat_count; j++) {
      ss.put((char)(data & 0xff));
      ss.put((char)((data & 0xff00) >> 8));
    }
  }

  *binary_size = wave_size;
  ss.seekg(0, std::ios::beg);

  std::vector<uint8_t> result(*binary_size);
  for (int i = 0; i < *binary_size; i++) {
    result[i] = (uint8_t)ss.get();
  }
  return result;
}

std::vector<float> SynthesisEngine::synthesis(AudioQueryModel query, int64_t *speaker_id,
                                              bool enable_interrogative_upspeak) {
  std::vector<AccentPhraseModel> accent_phrases = query.accent_phrases;
  if (enable_interrogative_upspeak) {
    accent_phrases = adjust_interrogative_accent_phrases(accent_phrases);
  }
  std::vector<MoraModel> flatten_moras;
  std::vector<int64_t> phoneme_id_list;
  std::vector<int64_t> accent_id_list;
  initial_process(accent_phrases, flatten_moras, phoneme_id_list, accent_id_list);

  float pre_phoneme_length = query.pre_phoneme_length;
  float post_phoneme_length = query.post_phoneme_length;

  float pitch_scale = query.pitch_scale;
  float speed_scale = query.speed_scale;
  float intonation_scale = query.intonation_scale;

  std::vector<float> durations;
  std::vector<float> pitches;
  std::vector<bool> voiced;
  float mean_pitch = 0.0;
  int count = 0;

  // VOICEVOXと最終的な音の長さを合わせるために、丸め誤差などを考慮して若干特殊な処理をしている
  int64_t wave_size = 0;
  for (MoraModel mora : flatten_moras) {
    float pitch = mora.pitch * std::pow(2.0f, pitch_scale);
    pitches.push_back(pitch);
    bool big_than_zero = pitch > 0.0;
    voiced.push_back(big_than_zero);
    if (big_than_zero) {
      mean_pitch += pitch;
      count++;
    }
    if (mora.consonant) {
      float consonant_length = static_cast<float>(*mora.consonant_length);
      durations.push_back(consonant_length);
      wave_size += (int64_t)std::nearbyint(consonant_length * 93.75f); // 48000 / 512 = 93.75
      pitches.push_back(pitch);
      voiced.push_back(big_than_zero);
      if (big_than_zero) {
        mean_pitch += pitch;
        count++;
      }
    }
    float vowel_length = mora.vowel_length;
    durations.push_back(vowel_length);
    wave_size += (int64_t)(vowel_length * 93.75f); // 48000 / 512 = 93.75
  }
  mean_pitch /= (float)count;
  wave_size *= 512;

  if (!std::isnan(mean_pitch)) {
    for (size_t i = 0; i < pitches.size(); i++) {
      if (voiced[i]) {
        pitches[i] = (pitches[i] - mean_pitch) * intonation_scale + mean_pitch;
      }
    }
  }

  std::vector<float> wave(wave_size, 0.0);
  bool success = decode_forward((int64_t)phoneme_id_list.size(), phoneme_id_list.data(), pitches.data(),
                                durations.data(), speaker_id, wave.data());

  if (!success) {
    throw std::runtime_error(last_error_message());
  }

  return wave;
}

void SynthesisEngine::load_openjtalk_dict(const std::string &dict_path) { m_openjtalk.load(dict_path); }

void SynthesisEngine::initial_process(std::vector<AccentPhraseModel> &accent_phrases,
                                      std::vector<MoraModel> &flatten_moras, std::vector<int64_t> &phoneme_id_list,
                                      std::vector<int64_t> &accent_id_list) {
  flatten_moras = to_flatten_moras(accent_phrases);
  std::vector<std::string> phoneme_str_list;
  std::vector<std::string> accent_str_list;

  for (MoraModel mora : flatten_moras) {
    std::optional<std::string> consonant = mora.consonant;
    if (consonant) phoneme_str_list.push_back(consonant.value());
    phoneme_str_list.push_back(mora.vowel);
  }
  for (auto accent_phrase : accent_phrases) {
    for (int64_t i = 0; i < accent_phrase.moras.size(); i++) {
      MoraModel mora = accent_phrase.moras[i];
      if (i + 1 == accent_phrase.accent && accent_phrase.moras.size() != accent_phrase.accent) {
        if (mora.consonant.has_value()) {
          accent_str_list.push_back("_");
        }
        accent_str_list.push_back("]");
      } else {
        if (mora.consonant.has_value()) {
          accent_str_list.push_back("_");
        }
        if (i == 0) {
          accent_str_list.push_back("[");
        } else {
          accent_str_list.push_back("_");
        }
      }
    }
    if (accent_phrase.pause_mora.has_value()) {
      accent_str_list.push_back("_");
    }
    if (accent_phrase.is_interrogative) {
      accent_str_list[accent_str_list.size() - 1] = "?";
    } else {
      accent_str_list[accent_str_list.size() - 1] = "#";
    }
  }

  phoneme_id_list = to_phoneme_id_list(phoneme_str_list);
  accent_id_list = to_accent_id_list(accent_str_list);
}

void SynthesisEngine::create_one_accent_list(std::vector<int64_t> &accent_list, AccentPhraseModel accent_phrase,
                                             int point) {
  std::vector<MoraModel> moras = accent_phrase.moras;

  std::vector<int64_t> one_accent_list;
  for (size_t i = 0; i < moras.size(); i++) {
    MoraModel mora = moras[i];
    int64_t value;
    if ((int)i == point || (point < 0 && i == moras.size() + point))
      value = 1;
    else
      value = 0;
    one_accent_list.push_back(value);
    if (mora.consonant) {
      one_accent_list.push_back(value);
    }
  }
  if (accent_phrase.pause_mora) one_accent_list.push_back(0);
  std::copy(one_accent_list.begin(), one_accent_list.end(), std::back_inserter(accent_list));
}
}  // namespace voicevox::core::engine
