use std::ffi::CStr;

use assert_cmd::assert::AssertResult;
use libloading::Library;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::{
    assert_cdylib::{self, case, Utf8Output},
    snapshots,
    symbols::Symbols,
};

case!(TestCase);

#[derive(Serialize, Deserialize)]
struct TestCase;

#[typetag::serde(name = "global_infos")]
impl assert_cdylib::TestCase for TestCase {
    unsafe fn exec(&self, lib: &Library) -> anyhow::Result<()> {
        let Symbols {
            voicevox_get_version,
            voicevox_get_supported_devices_json,
            ..
        } = Symbols::new(lib)?;

        let expected = voicevox_core::get_version();
        let actual = CStr::from_ptr(voicevox_get_version()).to_str()?;
        std::assert_eq!(expected, actual);

        let expected = voicevox_core::SupportedDevices::get_supported_devices()?;
        let actual = CStr::from_ptr(voicevox_get_supported_devices_json()).to_str()?;
        let actual = serde_json::from_str::<voicevox_core::SupportedDevices>(actual)?;
        std::assert_eq!(expected, actual);

        Ok(())
    }

    fn assert_output(&self, output: Utf8Output) -> AssertResult {
        output
            .assert()
            .try_success()?
            .try_stdout("")?
            .try_stderr(&*SNAPSHOTS.stderr)
    }
}

static SNAPSHOTS: Lazy<Snapshots> = snapshots::section!(global_infos);

#[derive(Deserialize)]
struct Snapshots {
    stderr: String,
}
