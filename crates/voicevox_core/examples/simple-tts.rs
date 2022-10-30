use std::path::PathBuf;

use clap::Parser as _;
use voicevox_core::{InitializeOptions, VoicevoxCore};

#[derive(clap::Parser)]
struct Args {
    openjtalk: String,
    text: String,
    output: PathBuf,
}

const SPEAKER_ID: u32 = 0;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut vv = VoicevoxCore::new_with_initialize(InitializeOptions {
        open_jtalk_dict_dir: Some(args.openjtalk.into()),
        ..Default::default()
    })?;

    vv.load_model(SPEAKER_ID)?;
    let wav = vv.tts(&args.text, SPEAKER_ID, Default::default())?;
    fs_err::write(&args.output, wav)?;
    eprintln!("Wrote {}", args.output.display());
    Ok(())
}
