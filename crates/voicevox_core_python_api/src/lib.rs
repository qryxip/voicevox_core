// FIXME: <https://github.com/PyO3/pyo3/pull/2503>を含むPyO3のリリースが出たら不要
#![allow(clippy::borrow_deref_ref)]

use std::fmt::Display;

use easy_ext::ext;
use log::debug;
use pyo3::{
    create_exception,
    exceptions::PyException,
    pyclass, pymethods, pymodule,
    types::{PyBytes, PyDict, PyModule},
    Py, PyAny, PyResult, Python, ToPyObject,
};
use voicevox_core::{AccentPhraseModel, AudioQueryModel, MoraModel};

macro_rules! dict {
    ($py:expr, $(($key:expr, $value:expr $(,)?)),* $(,)?) => {{
        let d = PyDict::new($py);
        $(
            d.set_item($key, $value)?;
        )*
        d
    }}
}

#[ext]
impl AudioQueryModel {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        return py.import("voicevox_core")?.getattr("AudioQuery")?.call(
            (),
            Some(dict!(
                py,
                ("accent_phrases", self.accent_phrases_to_py(py)?),
                ("speedScale", self.speed_scale()),
                ("pitchScale", self.pitch_scale()),
                ("intonationScale", self.intonation_scale()),
                ("volumeScale", self.volume_scale()),
                ("prePhonemeLength", self.pre_phoneme_length()),
                ("postPhonemeLength", self.post_phoneme_length()),
                ("outputSamplingRate", self.output_sampling_rate()),
                ("outputStereo", self.output_stereo()),
                ("kana", self.kana()),
            )),
        );

        #[ext]
        impl AudioQueryModel {
            fn accent_phrases_to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let accent_phrases = self
                    .accent_phrases()
                    .iter()
                    .map(|accent_phrase| accent_phrase.to_py(py))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(accent_phrases.to_object(py))
            }
        }
    }
}

#[ext]
impl AccentPhraseModel {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        return py.import("voicevox_core")?.getattr("AccentPhrase")?.call(
            (),
            Some(dict!(
                py,
                ("moras", self.moras_to_py(py)?),
                ("accent", self.accent()),
                ("pause_mora", self.pause_mora_to_py(py)?),
                ("is_interrogative", self.is_interrogative()),
            )),
        );

        #[ext]
        impl AccentPhraseModel {
            fn moras_to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let moras = self
                    .moras()
                    .iter()
                    .map(|mora| mora.to_py(py))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(moras.to_object(py))
            }

            fn pause_mora_to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let pause_mora = self
                    .pause_mora()
                    .as_ref()
                    .map(|pause_mora| pause_mora.to_py(py))
                    .transpose()?;
                Ok(pause_mora.to_object(py))
            }
        }
    }
}

#[ext]
impl MoraModel {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        py.import("voicevox_core")?.getattr("Mora")?.call(
            (),
            Some(dict!(
                py,
                ("text", self.text()),
                ("consonant", self.consonant()),
                ("consonant_length", self.consonant_length()),
                ("vowel", self.vowel()),
                ("vowel_length", self.vowel_length()),
                ("pitch", self.pitch()),
            )),
        )
    }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn rust(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    // `importlib.import_module("json").loads`
    let json_loads = py.import("json")?.getattr("loads")?;

    // `json_loads(METAS)`
    let metas = json_loads.call1((voicevox_core::METAS,))?;

    // `json_loads(SUPPORTED_DEVICES)`
    let supported_devices = json_loads.call1((voicevox_core::SUPPORTED_DEVICES.to_json(),))?;

    module.add("METAS", metas)?;
    module.add("SUPPORTED_DEVICES", supported_devices)?;
    module.add_class::<VoicevoxCore>()
}

create_exception!(
    voicevox_core,
    VoicevoxError,
    PyException,
    "voicevox_core Error."
);

#[pyclass]
struct VoicevoxCore {
    inner: voicevox_core::VoicevoxCore,
}

#[pymethods]
impl VoicevoxCore {
    #[new]
    fn new(use_gpu: bool, cpu_num_threads: usize, load_all_models: bool) -> PyResult<Self> {
        let inner = voicevox_core::VoicevoxCore::new_with_initialize(
            use_gpu,
            cpu_num_threads,
            load_all_models,
        )
        .into_py_result()?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn a(py: Python<'_>) -> PyResult<&PyAny> {
        let o = py
            .import("voicevox_core_python_api")?
            .getattr("_models")?
            .getattr("AudioQuery")?;
        Ok(o)
    }

    fn load_model(&mut self, speaker_id: usize) -> PyResult<()> {
        self.inner.load_model(speaker_id).into_py_result()
    }

    fn is_model_loaded(&self, speaker_id: usize) -> bool {
        self.inner.is_model_loaded(speaker_id)
    }

    fn load_openjtalk_dict(&mut self, dict_path: &str) -> PyResult<()> {
        self.inner
            .voicevox_load_openjtalk_dict(dict_path)
            .into_py_result()
    }

    fn audio_query<'py>(
        &mut self,
        text: &str,
        speaker_id: usize,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        self.inner
            .voicevox_audio_query(text, speaker_id)
            .into_py_result()?
            .to_py(py)
    }

    fn audio_query_from_kana<'py>(
        &mut self,
        text: &str,
        speaker_id: usize,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        self.inner
            .voicevox_audio_query_from_kana(text, speaker_id)
            .into_py_result()?
            .to_py(py)
    }

    fn synthesis<'py>(
        &mut self,
        #[pyo3(from_py_with = "as_audio_query")] audio_query: AudioQueryModel,
        speaker_id: usize,
        py: Python<'py>,
    ) -> PyResult<&'py PyBytes> {
        let wav = &self
            .inner
            .voicevox_synthesis(&audio_query, speaker_id)
            .into_py_result()?;
        Ok(PyBytes::new(py, wav))
    }

    fn tts<'py>(
        &mut self,
        text: &str,
        speaker_id: usize,
        py: Python<'py>,
    ) -> PyResult<&'py PyBytes> {
        let wav = &self.inner.voicevox_tts(text, speaker_id).into_py_result()?;
        Ok(PyBytes::new(py, wav))
    }

    fn tts_from_kana<'py>(
        &mut self,
        text: &str,
        speaker_id: usize,
        py: Python<'py>,
    ) -> PyResult<&'py PyBytes> {
        let wav = &self
            .inner
            .voicevox_tts_from_kana(text, speaker_id)
            .into_py_result()?;
        Ok(PyBytes::new(py, wav))
    }

    fn __repr__(&self) -> &'static str {
        "VoicevoxCore { .. }"
    }
}

fn as_audio_query(obj: &PyAny) -> PyResult<AudioQueryModel> {
    let py = obj.py();

    let audio_query = py.import("dataclasses")?.getattr("asdict")?.call1((obj,))?;
    let audio_query = &py
        .import("json")?
        .call_method1("dumps", (audio_query,))?
        .extract::<String>()?;
    serde_json::from_str(audio_query).into_py_result()
}

impl Drop for VoicevoxCore {
    fn drop(&mut self) {
        debug!("Destructing a VoicevoxCore");
        self.inner.finalize();
    }
}

#[ext]
impl<T, E: Display> Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| VoicevoxError::new_err(e.to_string()))
    }
}
