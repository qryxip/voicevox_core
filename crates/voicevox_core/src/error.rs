use self::engine::{FullContextLabelError, KanaParseError};
use self::result_code::VoicevoxResultCode::{self, *};
use super::*;
//use engine::
use std::path::PathBuf;
use thiserror::Error;

/*
 * 新しいエラーを定義したら、必ずresult_code.rsにあるVoicevoxResultCodeに対応するコードを定義し、
 * internal.rsにある変換関数に変換処理を加えること
 */

#[derive(Error, Debug)]
pub enum Error {
    /*
     * エラーメッセージのベースとなる文字列は必ずbase_error_message関数を使用してVoicevoxResultCodeのエラー出力の内容と対応するようにすること
     */
    #[error(
        "{}",
        base_error_message(VOICEVOX_RESULT_NOT_LOADED_OPENJTALK_DICT_ERROR)
    )]
    NotLoadedOpenjtalkDict,

    #[error("{}", base_error_message(VOICEVOX_RESULT_GPU_SUPPORT_ERROR))]
    GpuSupport,

    #[error("{} ({}): {source}", base_error_message(VOICEVOX_RESULT_LOAD_MODEL_ERROR), path.display())]
    LoadModel {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
    #[error("{} ({})", base_error_message(VOICEVOX_ALREADY_LOADED_MODEL_ERROR), path.display())]
    AlreadyLoadedModel { path: PathBuf },

    #[error("{} ({model_id:?})", base_error_message(VOICEVOX_UNLOADED_MODEL_ERROR))]
    UnloadedModel { model_id: VoiceModelId },

    #[error("{}({path}):{source}", base_error_message(VOICEVOX_OPEN_FILE_ERROR))]
    OpenFile {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },

    #[error("{},{filename}", base_error_message(VOICEVOX_VVM_MODEL_READ_ERROR))]
    VvmRead { filename: String },

    #[error("{},{0}", base_error_message(VOICEVOX_RESULT_LOAD_METAS_ERROR))]
    LoadMetas(#[source] anyhow::Error),

    #[error(
        "{},{0}",
        base_error_message(VOICEVOX_RESULT_GET_SUPPORTED_DEVICES_ERROR)
    )]
    GetSupportedDevices(#[source] anyhow::Error),

    #[error(
        "{}: {style_id:?}",
        base_error_message(VOICEVOX_RESULT_INVALID_STYLE_ID_ERROR)
    )]
    InvalidStyleId { style_id: StyleId },

    #[error(
        "{}: {model_index}",
        base_error_message(VOICEVOX_RESULT_INVALID_MODEL_INDEX_ERROR)
    )]
    InvalidModelIndex { model_index: usize },

    #[error("{}", base_error_message(VOICEVOX_RESULT_INFERENCE_ERROR))]
    InferenceFailed,

    #[error(
        "{},{0}",
        base_error_message(VOICEVOX_RESULT_EXTRACT_FULL_CONTEXT_LABEL_ERROR)
    )]
    ExtractFullContextLabel(#[from] FullContextLabelError),

    #[error("{},{0}", base_error_message(VOICEVOX_RESULT_PARSE_KANA_ERROR))]
    ParseKana(#[from] KanaParseError),
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NotLoadedOpenjtalkDict, Self::NotLoadedOpenjtalkDict)
            | (Self::GpuSupport, Self::GpuSupport)
            | (Self::InferenceFailed, Self::InferenceFailed) => true,
            (
                Self::LoadModel {
                    path: path1,
                    source: source1,
                },
                Self::LoadModel {
                    path: path2,
                    source: source2,
                },
            ) => (path1, source1.to_string()) == (path2, source2.to_string()),
            (Self::LoadMetas(e1), Self::LoadMetas(e2))
            | (Self::GetSupportedDevices(e1), Self::GetSupportedDevices(e2)) => {
                e1.to_string() == e2.to_string()
            }
            (
                Self::InvalidStyleId {
                    style_id: speaker_id1,
                },
                Self::InvalidStyleId {
                    style_id: speaker_id2,
                },
            ) => speaker_id1 == speaker_id2,
            (
                Self::InvalidModelIndex {
                    model_index: model_index1,
                },
                Self::InvalidModelIndex {
                    model_index: model_index2,
                },
            ) => model_index1 == model_index2,
            (Self::ExtractFullContextLabel(e1), Self::ExtractFullContextLabel(e2)) => e1 == e2,
            (Self::ParseKana(e1), Self::ParseKana(e2)) => e1 == e2,
            (
                Self::VvmRead {
                    filename: filename1,
                },
                Self::VvmRead {
                    filename: filename2,
                },
            ) => filename1 == filename2,
            (
                Self::OpenFile {
                    path: path1,
                    source: source1,
                },
                Self::OpenFile {
                    path: path2,
                    source: source2,
                },
            ) => path1 == path2 && source1.to_string() == source2.to_string(),
            _ => false,
        }
    }
}

fn base_error_message(result_code: VoicevoxResultCode) -> &'static str {
    let c_message: &'static str = crate::result_code::error_result_to_message(result_code);
    &c_message[..(c_message.len() - 1)]
}
