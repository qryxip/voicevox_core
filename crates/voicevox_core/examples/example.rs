use voicevox_core::{
    blocking::{Onnxruntime, Synthesizer, VoiceModel},
    AccelerationMode, InitializeOptions,
};

fn main() -> anyhow::Result<()> {
    let _synth = Synthesizer::new(
        Onnxruntime::load_once()
            .filename("example/python/libvoicevox_onnxruntime.so.1.17.3")
            .exec()?,
        (),
        &InitializeOptions {
            acceleration_mode: AccelerationMode::Gpu,
            ..Default::default()
        },
    )?;
    let synth = Synthesizer::new(
        Onnxruntime::load_once()
            .filename("./libonnxruntime.so.1.17.3")
            .exec()?,
        (),
        &InitializeOptions {
            acceleration_mode: AccelerationMode::Gpu,
            ..Default::default()
        },
    )?;
    synth.load_voice_model(&VoiceModel::from_path(
        "crates/test_util/data/model/sample.vvm",
    )?)?;
    Ok(())
}
