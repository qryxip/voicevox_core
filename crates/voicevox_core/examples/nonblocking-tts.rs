use test_util::{ONNXRUNTIME_DYLIB_PATH, OPEN_JTALK_DIC_DIR, SAMPLE_VOICE_MODEL_FILE_PATH};
use voicevox_core::{
    nonblocking::{Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile},
    StyleId,
};

const STYLE_ID: StyleId = StyleId(0);
const TEXT: &str = "この音声は、ボイスボックスを使用して、出力されています。";

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    let synth = {
        let ort = Onnxruntime::load_once()
            .filename(ONNXRUNTIME_DYLIB_PATH)
            .perform()
            .await?;
        let ojt = OpenJtalk::new(OPEN_JTALK_DIC_DIR).await?;
        Synthesizer::builder(ort).text_analyzer(ojt).build()?
    };
    synth
        .load_voice_model(&VoiceModelFile::open(SAMPLE_VOICE_MODEL_FILE_PATH).await?)
        .await?;
    synth.tts(TEXT, STYLE_ID).perform().await?;
    Ok(())
}
