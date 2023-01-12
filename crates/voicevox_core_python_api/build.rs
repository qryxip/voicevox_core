use fs_extra::dir::CopyOptions;

fn main() -> anyhow::Result<()> {
    fs_extra::dir::copy(
        "../../model",
        "./python/sharevox_core/",
        &CopyOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;
    Ok(())
}
