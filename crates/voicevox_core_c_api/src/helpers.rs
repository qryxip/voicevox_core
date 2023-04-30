use std::alloc::Layout;
use std::collections::BTreeMap;
use std::fmt::Debug;

use thiserror::Error;

use super::*;

pub(crate) fn into_result_code_with_error(result: CApiResult<()>) -> VoicevoxResultCode {
    if let Err(err) = &result {
        display_error(err);
    }
    return into_result_code(result);

    fn display_error(err: &CApiError) {
        eprintln!("Error(Display): {err}");
        eprintln!("Error(Debug): {err:#?}");
    }

    fn into_result_code(result: CApiResult<()>) -> VoicevoxResultCode {
        use voicevox_core::{result_code::VoicevoxResultCode::*, Error::*};
        use CApiError::*;

        match result {
            Ok(()) => VOICEVOX_RESULT_OK,
            Err(RustApi(NotLoadedOpenjtalkDict)) => VOICEVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR,
            Err(RustApi(GpuSupport)) => VOICEVOX_RESULT_GPU_SUPPORT_ERROR,
            Err(RustApi(LoadModel { .. })) => VOICEVOX_RESULT_LOAD_MODEL_ERROR,
            Err(RustApi(LoadMetas(_))) => VOICEVOX_RESULT_LOAD_METAS_ERROR,
            Err(RustApi(GetSupportedDevices(_))) => VOICEVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR,
            Err(RustApi(UninitializedStatus)) => VOICEVOX_RESULT_UNINITIALIZED_STATUS_ERROR,
            Err(RustApi(InvalidSpeakerId { .. })) => VOICEVOX_RESULT_INVALID_SPEAKER_ID_ERROR,
            Err(RustApi(InvalidModelIndex { .. })) => VOICEVOX_RESULT_INVALID_MODEL_INDEX_ERROR,
            Err(RustApi(InferenceFailed)) => VOICEVOX_RESULT_INFERENCE_ERROR,
            Err(RustApi(ExtractFullContextLabel(_))) => {
                VOICEVOX_RESULT_EXTRACT_FULL_CONTEXT_LABEL_ERROR
            }
            Err(RustApi(ParseKana(_))) => VOICEVOX_RESULT_PARSE_KANA_ERROR,
            Err(InvalidUtf8Input) => VOICEVOX_RESULT_INVALID_UTF8_INPUT_ERROR,
            Err(InvalidAudioQuery(_)) => VOICEVOX_RESULT_INVALID_AUDIO_QUERY_ERROR,
        }
    }
}

type CApiResult<T> = std::result::Result<T, CApiError>;

#[derive(Error, Debug)]
pub(crate) enum CApiError {
    #[error("{0}")]
    RustApi(#[from] voicevox_core::Error),
    #[error("UTF-8として不正な入力です")]
    InvalidUtf8Input,
    #[error("無効なAudioQueryです: {0}")]
    InvalidAudioQuery(serde_json::Error),
}

pub(crate) fn create_audio_query(
    japanese_or_kana: &CStr,
    speaker_id: u32,
    method: fn(
        &mut Internal,
        &str,
        u32,
        voicevox_core::AudioQueryOptions,
    ) -> Result<AudioQueryModel>,
    options: VoicevoxAudioQueryOptions,
) -> CApiResult<CString> {
    let japanese_or_kana = ensure_utf8(japanese_or_kana)?;

    let audio_query = method(
        &mut lock_internal(),
        japanese_or_kana,
        speaker_id,
        options.into(),
    )?;
    Ok(CString::new(audio_query_model_to_json(&audio_query)).expect("should not contain '\\0'"))
}

fn audio_query_model_to_json(audio_query_model: &AudioQueryModel) -> String {
    serde_json::to_string(audio_query_model).expect("should be always valid")
}

pub(crate) fn ensure_utf8(s: &CStr) -> CApiResult<&str> {
    s.to_str().map_err(|_| CApiError::InvalidUtf8Input)
}

impl From<voicevox_core::AudioQueryOptions> for VoicevoxAudioQueryOptions {
    fn from(options: voicevox_core::AudioQueryOptions) -> Self {
        Self { kana: options.kana }
    }
}
impl From<VoicevoxAudioQueryOptions> for voicevox_core::AudioQueryOptions {
    fn from(options: VoicevoxAudioQueryOptions) -> Self {
        Self { kana: options.kana }
    }
}

impl From<VoicevoxSynthesisOptions> for voicevox_core::SynthesisOptions {
    fn from(options: VoicevoxSynthesisOptions) -> Self {
        Self {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl From<voicevox_core::AccelerationMode> for VoicevoxAccelerationMode {
    fn from(mode: voicevox_core::AccelerationMode) -> Self {
        use voicevox_core::AccelerationMode::*;
        match mode {
            Auto => Self::VOICEVOX_ACCELERATION_MODE_AUTO,
            Cpu => Self::VOICEVOX_ACCELERATION_MODE_CPU,
            Gpu => Self::VOICEVOX_ACCELERATION_MODE_GPU,
        }
    }
}

impl From<VoicevoxAccelerationMode> for voicevox_core::AccelerationMode {
    fn from(mode: VoicevoxAccelerationMode) -> Self {
        use VoicevoxAccelerationMode::*;
        match mode {
            VOICEVOX_ACCELERATION_MODE_AUTO => Self::Auto,
            VOICEVOX_ACCELERATION_MODE_CPU => Self::Cpu,
            VOICEVOX_ACCELERATION_MODE_GPU => Self::Gpu,
        }
    }
}

impl Default for VoicevoxInitializeOptions {
    fn default() -> Self {
        let options = voicevox_core::InitializeOptions::default();
        Self {
            acceleration_mode: options.acceleration_mode.into(),
            cpu_num_threads: options.cpu_num_threads,
            load_all_models: options.load_all_models,
            open_jtalk_dict_dir: null(),
        }
    }
}

impl VoicevoxInitializeOptions {
    pub(crate) unsafe fn try_into_options(self) -> CApiResult<voicevox_core::InitializeOptions> {
        let open_jtalk_dict_dir = (!self.open_jtalk_dict_dir.is_null())
            .then(|| ensure_utf8(CStr::from_ptr(self.open_jtalk_dict_dir)).map(Into::into))
            .transpose()?;
        Ok(voicevox_core::InitializeOptions {
            acceleration_mode: self.acceleration_mode.into(),
            cpu_num_threads: self.cpu_num_threads,
            load_all_models: self.load_all_models,
            open_jtalk_dict_dir,
        })
    }
}

impl From<voicevox_core::TtsOptions> for VoicevoxTtsOptions {
    fn from(options: voicevox_core::TtsOptions) -> Self {
        Self {
            kana: options.kana,
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl From<VoicevoxTtsOptions> for voicevox_core::TtsOptions {
    fn from(options: VoicevoxTtsOptions) -> Self {
        Self {
            kana: options.kana,
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

impl Default for VoicevoxSynthesisOptions {
    fn default() -> Self {
        let options = voicevox_core::TtsOptions::default();
        Self {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
        }
    }
}

pub(crate) struct BufferManager {
    address_to_layout_table: BTreeMap<usize, Layout>,
}

impl BufferManager {
    pub const fn new() -> Self {
        Self {
            address_to_layout_table: BTreeMap::new(),
        }
    }

    pub fn vec_into_raw<T: Copy>(&mut self, vec: Vec<T>) -> (*mut T, usize) {
        let slice = Box::leak(vec.into_boxed_slice());
        let layout = Layout::for_value(slice);
        let len = slice.len();
        let ptr = slice.as_mut_ptr();
        let addr = ptr as usize;

        let not_occupied = self.address_to_layout_table.insert(addr, layout).is_none();

        assert!(not_occupied, "すでに値が入っている状態はおかしい");

        (ptr, len)
    }

    /// `vec_into_raw`でC API利用側に貸し出したポインタに対し、デアロケートする。
    ///
    /// # Safety
    ///
    /// - `buffer_ptr`は`vec_into_raw`で取得したものであること。
    pub unsafe fn dealloc_slice<T: Copy>(&mut self, buffer_ptr: *const T) {
        let addr = buffer_ptr as usize;
        let layout = self.address_to_layout_table.remove(&addr).expect(
            "解放しようとしたポインタはvoicevox_coreの管理下にありません。\
             誤ったポインタであるか、二重解放になっていることが考えられます",
        );

        if layout.size() > 0 {
            // `T: Copy`より、`T: !Drop`であるため`drop_in_place`は不要

            // SAFETY:
            // - `addr`と`layout`は対応したものである
            // - `layout.size() > 0`より、`addr`はダングリングではない有効なポインタである
            std::alloc::dealloc(addr as *mut u8, layout);
        }
    }

    pub fn c_string_into_raw(&mut self, s: CString) -> *const c_char {
        s.into_raw() as *const c_char
    }

    /// `c_string_into_raw`でリークしたポインタをCStringに戻す。
    ///
    /// # Safety
    ///
    /// - `s` は`c_string_into_raw`で取得したものであること。
    pub unsafe fn dealloc_c_string(&mut self, s: *const c_char) {
        // SAFETY:
        // - `s`は`CString::into_raw`で得たものである
        drop(CString::from_raw(s as *mut c_char));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_manager_works() {
        let mut buffer_manager = BufferManager::new();

        unsafe {
            let (ptr, len) = buffer_manager.vec_into_raw(vec![()]);
            assert_eq!(1, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let (ptr, len) = buffer_manager.vec_into_raw(Vec::<u8>::new());
            assert_eq!(0, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let (ptr, len) = buffer_manager.vec_into_raw(vec![0u8]);
            assert_eq!(1, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let mut vec = Vec::with_capacity(2);
            vec.push(0u8);
            let (ptr, len) = buffer_manager.vec_into_raw(vec);
            assert_eq!(1, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let (ptr, len) = buffer_manager.vec_into_raw(Vec::<f32>::new());
            assert_eq!(0, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let (ptr, len) = buffer_manager.vec_into_raw(vec![0f32]);
            assert_eq!(1, len);
            buffer_manager.dealloc_slice(ptr);
        }

        unsafe {
            let mut vec = Vec::with_capacity(2);
            vec.push(0f32);
            let (ptr, len) = buffer_manager.vec_into_raw(vec);
            assert_eq!(1, len);
            buffer_manager.dealloc_slice(ptr);
        }
    }

    #[test]
    #[should_panic(
        expected = "解放しようとしたポインタはvoicevox_coreの管理下にありません。誤ったポインタであるか、二重解放になっていることが考えられます"
    )]
    fn buffer_manager_denies_unknown_ptr() {
        let mut buffer_manager = BufferManager::new();
        unsafe {
            let x = 42;
            buffer_manager.dealloc_slice(&x as *const i32);
        }
    }
}
