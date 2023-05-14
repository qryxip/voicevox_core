use std::{ffi::CStr, mem::MaybeUninit};

use assert_cmd::assert::AssertResult;
use libloading::Library;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use voicevox_core::result_code::VoicevoxResultCode;

use crate::{
    assert_cdylib::{self, case, Utf8Output},
    snapshots,
    symbols::Symbols,
};

macro_rules! cstr {
    ($s:literal $(,)?) => {
        CStr::from_bytes_with_nul(concat!($s, '\0').as_ref()).unwrap()
    };
}

case!(TestCase);

#[derive(Serialize, Deserialize)]
struct TestCase;

#[typetag::serde(name = "voice_model_infos")]
impl assert_cdylib::TestCase for TestCase {
    unsafe fn exec(&self, lib: &Library) -> anyhow::Result<()> {
        let Symbols {
            voicevox_voice_model_new_from_path,
            voicevox_voice_model_id,
            voicevox_voice_model_get_metas_json,
            voicevox_voice_model_delete,
            ..
        } = Symbols::new(lib)?;

        let expected_model = RUNTIME.block_on(voicevox_core::VoiceModel::from_path(
            "../../model/sample.vvm",
        ))?;

        let actual_model = {
            let mut model = MaybeUninit::uninit();
            assert_ok(voicevox_voice_model_new_from_path(
                cstr!("../../model/sample.vvm").as_ptr(),
                model.as_mut_ptr(),
            ));
            model.assume_init()
        };

        let actual_id = voicevox_voice_model_id(actual_model);
        let actual_metas_json = voicevox_voice_model_get_metas_json(actual_model);

        {
            static NANOID: Lazy<Regex> = Lazy::new(|| r"\A[A-Za-z0-9_-]{21}\z".parse().unwrap());

            let actual_id = CStr::from_ptr(actual_id).to_str()?;
            assert!(NANOID.is_match(actual_id));
        }

        {
            let expected_metas_json = expected_model.metas();
            let actual_metas_json = CStr::from_ptr(actual_metas_json).to_str()?;
            let actual_metas_json =
                &serde_json::from_str::<Vec<voicevox_core::SpeakerMeta>>(actual_metas_json)?;
            std::assert_eq!(expected_metas_json, actual_metas_json);
        }

        voicevox_voice_model_delete(actual_model);

        return Ok(());

        static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().unwrap());

        fn assert_ok(result_code: VoicevoxResultCode) {
            std::assert_eq!(VoicevoxResultCode::VOICEVOX_RESULT_OK, result_code);
        }
    }

    fn assert_output(&self, output: Utf8Output) -> AssertResult {
        output
            .mask_timestamps()
            .mask_windows_video_cards()
            .assert()
            .try_success()?
            .try_stdout("")?
            .try_stderr(&*SNAPSHOTS.stderr)
    }
}

static SNAPSHOTS: Lazy<Snapshots> = snapshots::section!(voice_model_infos);

#[derive(Deserialize)]
struct Snapshots {
    #[serde(deserialize_with = "snapshots::deserialize_platform_specific_snapshot")]
    stderr: String,
}
