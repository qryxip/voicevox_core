pub(super) fn decrypt(content: &[u8]) -> std::result::Result<Vec<u8>, DecryptError> {
    Ok(content.to_owned())
}

pub(super) const SPEAKER_ID_MAP: &[(u32, (usize, u32))] = &[(0, (0, 0)), (1, (0, 1))];

pub(super) const MODEL_FILE_NAMES: &[ModelFileNames] = &[ModelFileNames {
    predict_duration_model: "predict_duration.onnx",
    predict_intonation_model: "predict_intonation.onnx",
    decode_model: "decode.onnx",
}];

pub(super) struct ModelFileNames {
    pub(super) predict_duration_model: &'static str,
    pub(super) predict_intonation_model: &'static str,
    pub(super) decode_model: &'static str,
}

#[derive(thiserror::Error, Debug)]
#[error("不正なモデルファイルです")]
pub(super) struct DecryptError;
