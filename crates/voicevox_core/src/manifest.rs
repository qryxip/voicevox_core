use std::{
    collections::BTreeMap,
    fmt::{self, Display},
    sync::Arc,
};

use derive_getters::Getters;
use derive_more::Deref;
use derive_new::new;
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_with::{serde_as, DisplayFromStr};

use crate::{StyleId, VoiceModelId};

pub(crate) use self::model_filename::ModelFilename;

#[derive(Clone)]
struct FormatVersionV1;

impl<'de> Deserialize<'de> for FormatVersionV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        return deserializer.deserialize_any(Visitor);

        struct Visitor;

        impl<'de> de::Visitor<'de> for Visitor {
            type Value = FormatVersionV1;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an unsigned integer")
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match v {
                    1 => Ok(FormatVersionV1),
                    v => Err(E::custom(format!(
                        "未知の形式です（`vvm_format_version={v}`）。新しいバージョンのVOICEVOX \
                         COREであれば対応しているかもしれません",
                    ))),
                }
            }
        }
    }
}

/// モデル内IDの実体
pub type RawInnerVoiceId = u32;
/// モデル内ID
#[derive(PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Deserialize, Serialize, new, Debug)]
pub struct InnerVoiceId(RawInnerVoiceId);

impl InnerVoiceId {
    pub fn raw_id(self) -> RawInnerVoiceId {
        self.0
    }
}

impl Display for InnerVoiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw_id())
    }
}

#[derive(Deserialize, Getters, Clone)]
pub struct Manifest {
    #[allow(dead_code)]
    vvm_format_version: FormatVersionV1,
    pub(crate) id: VoiceModelId,
    metas_filename: String,
    #[serde(flatten)]
    domains: ManifestDomains,
}

#[derive(Deserialize, Clone)]
pub(crate) struct ManifestDomains {
    pub(crate) talk: Option<TalkManifest>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct TalkManifest {
    pub(crate) predict_duration_filename: ModelFilename,
    pub(crate) predict_intonation_filename: ModelFilename,
    pub(crate) decode_filename: ModelFilename,
    #[serde(default)]
    pub(crate) style_id_to_inner_voice_id: StyleIdToInnerVoiceId,
}

#[derive(Clone, Copy, strum::AsRefStr)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum ModelFileKind {
    Onnx,
    Bin,
}

#[serde_as]
#[derive(Default, Clone, Deref, Deserialize)]
#[deref(forward)]
pub(crate) struct StyleIdToInnerVoiceId(
    #[serde_as(as = "Arc<BTreeMap<DisplayFromStr, _>>")] Arc<BTreeMap<StyleId, InnerVoiceId>>,
);

mod model_filename {
    use camino::Utf8Path;
    use serde::{de::Error as _, Deserialize, Deserializer};

    use super::ModelFileKind;

    #[derive(Clone)]
    pub(crate) struct ModelFilename(String);

    impl ModelFilename {
        #[cfg(test)]
        pub(crate) fn new(stem: &str, kind: ModelFileKind) -> Self {
            ModelFilename(Utf8Path::new(stem).with_extension(kind).into())
        }

        pub(crate) fn get(&self) -> &str {
            &self.0
        }

        pub(crate) fn kind(&self) -> ModelFileKind {
            match Utf8Path::new(&self.0).extension() {
                Some("onnx") => ModelFileKind::Onnx,
                Some("bin") => ModelFileKind::Bin,
                _ => unreachable!(
                    "unexpected extension. this should have been checked: {:?}",
                    self.0,
                ),
            }
        }
    }

    impl<'de> Deserialize<'de> for ModelFilename {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let s = String::deserialize(deserializer)?;
            if ![Some("onnx"), Some("bin")].contains(&Utf8Path::new(&s).extension()) {
                return Err(D::Error::custom(
                    "model filename must ends with .onnx or .bin",
                ));
            }
            Ok(Self(s))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use rstest::rstest;
    use serde::Deserialize;

    use super::FormatVersionV1;

    #[rstest]
    #[case("{\"vvm_format_version\":1}", Ok(()))]
    #[case(
        "{\"vvm_format_version\":2}",
        Err(
            "未知の形式です（`vvm_format_version=2`）。新しいバージョンのVOICEVOX COREであれば対応\
             しているかもしれません at line 1 column 23",
        )
    )]
    fn vvm_format_version_works(
        #[case] input: &str,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let actual = serde_json::from_str::<ManifestPart>(input).map_err(|e| e.to_string());
        let actual = actual.as_ref().map(|_| ()).map_err(Deref::deref);
        assert_eq!(expected, actual);
        return Ok(());

        #[derive(Deserialize)]
        struct ManifestPart {
            #[allow(dead_code)]
            vvm_format_version: FormatVersionV1,
        }
    }
}
