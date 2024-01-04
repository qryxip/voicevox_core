use std::{collections::BTreeMap, iter};

use easy_ext::ext;
use ndarray::{Array, Array2};
use num_traits::Zero;
use world::{
    signal_analyzer::{AnalyzeResult, SignalAnalyzerBuilder},
    spectrogram_like::SpectrogramLike,
};

use crate::{
    error::ErrorRepr, synthesizer::DEFAULT_SAMPLING_RATE, AudioQueryModel, MorphableTargetInfo,
    SpeakerMeta, StyleId, StyleMeta,
};

use self::permit::MorphableStyles;

impl<O> crate::blocking::Synthesizer<O> {
    pub(crate) fn morphable_targets_(
        &self,
        style_id: StyleId,
    ) -> crate::Result<BTreeMap<StyleId, MorphableTargetInfo>> {
        let metas = &self.metas();

        metas
            .iter()
            .flat_map(SpeakerMeta::styles)
            .map(StyleMeta::id)
            .map(|&target| {
                let style_ids = MorphingPair {
                    base: style_id,
                    target,
                };
                let is_morphable = self.is_synthesis_morphing_permitted(style_ids, metas)?;
                Ok((target, MorphableTargetInfo { is_morphable }))
            })
            .collect()
    }

    fn is_synthesis_morphing_permitted(
        &self,
        style_ids: MorphingPair<StyleId>,
        metas: &[SpeakerMeta],
    ) -> crate::Result<bool> {
        let pair = style_ids.lookup_speakers(metas)?;
        Ok(MorphableStyles::permit(pair).is_ok())
    }

    pub(crate) fn synthesis_morphing_(
        &self,
        audio_query: &AudioQueryModel,
        base_style_id: StyleId,
        target_style_id: StyleId,
        morph_rate: f64,
    ) -> crate::Result<Vec<u8>> {
        let metas = &self.metas();

        let pair = MorphingPair {
            base: base_style_id,
            target: target_style_id,
        }
        .lookup_speakers(metas)?;

        MorphableStyles::permit(pair)?.synthesis_morphing(self, audio_query, morph_rate)
    }
}

impl<'metas> MorphableStyles<'metas> {
    fn synthesis_morphing(
        self,
        synthesizer: &crate::blocking::Synthesizer<impl Sized>,
        audio_query: &AudioQueryModel,
        morph_rate: f64,
    ) -> crate::Result<Vec<u8>> {
        let waves = &self.get().try_map(|style_id| {
            synthesizer.synthesis_wave(audio_query, style_id, &Default::default())
        })?;

        let MorphingParameter {
            base_f0,
            base_aperiodicity,
            base_spectrogram,
            target_spectrogram,
        } = &MorphingParameter::new(waves);

        let morph_spectrogram =
            &(base_spectrogram * (1. - morph_rate) + target_spectrogram * morph_rate).into();

        let wave = &world::synthesis::synthesis(
            base_f0,
            morph_spectrogram,
            base_aperiodicity,
            None,
            FRAME_PERIOD,
            DEFAULT_SAMPLING_RATE,
        )
        .unwrap_or_else(|_| {
            // FIXME: ここをどうするか考える。ただしここのエラーはspectrogramが巨大すぎる
            // (`world::synthesis::SynthesisError::TooLargeValue`)ときに限るはず
            todo!()
        });

        return Ok(super::to_wav(wave, audio_query));

        const FRAME_PERIOD: f64 = 1.;

        struct MorphingParameter {
            base_f0: Box<[f64]>,
            base_aperiodicity: SpectrogramLike<f64>,
            base_spectrogram: Array2<f64>,
            target_spectrogram: Array2<f64>,
        }

        impl MorphingParameter {
            fn new(wave: &MorphingPair<Vec<f32>>) -> Self {
                let (base_f0, base_spectrogram, base_aperiodicity) = analyze(&wave.base);
                let (_, target_spectrogram, _) = analyze(&wave.target);

                let base_spectrogram = Array::from(base_spectrogram);
                let target_spectrogram =
                    Array::from(target_spectrogram).resize(base_spectrogram.dim());

                Self {
                    base_f0,
                    base_aperiodicity,
                    base_spectrogram,
                    target_spectrogram,
                }
            }
        }

        fn analyze(wave: &[f32]) -> (Box<[f64]>, SpectrogramLike<f64>, SpectrogramLike<f64>) {
            let analyzer = {
                let mut analyzer = SignalAnalyzerBuilder::new(DEFAULT_SAMPLING_RATE);
                analyzer.harvest_option_mut().set_frame_period(FRAME_PERIOD);
                analyzer.build(wave.iter().copied().map(Into::into).collect())
            };

            analyzer.calc_all();

            let AnalyzeResult {
                f0,
                spectrogram,
                aperiodicity,
                ..
            } = analyzer.into_result();

            let f0 = f0.expect("should be present");
            let spectrogram = spectrogram.expect("should be present");
            let aperiodicity = aperiodicity.expect("should be present");

            (f0, spectrogram, aperiodicity)
        }
    }
}

#[derive(Clone, Copy)]
struct MorphingPair<T> {
    base: T,
    target: T,
}

impl<T> MorphingPair<T> {
    fn map<S>(self, mut f: impl FnMut(T) -> S) -> MorphingPair<S> {
        let base = f(self.base);
        let target = f(self.target);
        MorphingPair { base, target }
    }

    fn try_map<S, E>(
        self,
        mut f: impl FnMut(T) -> std::result::Result<S, E>,
    ) -> std::result::Result<MorphingPair<S>, E> {
        let base = f(self.base)?;
        let target = f(self.target)?;
        Ok(MorphingPair { base, target })
    }
}

impl MorphingPair<StyleId> {
    fn lookup_speakers(
        self,
        metas: &[SpeakerMeta],
    ) -> crate::Result<MorphingPair<(StyleId, &SpeakerMeta)>> {
        self.try_map(|style_id| {
            let speaker = metas
                .iter()
                .find(|m| m.styles().iter().any(|m| *m.id() == style_id))
                .ok_or(ErrorRepr::StyleNotFound { style_id })?;
            Ok((style_id, speaker))
        })
    }
}

#[ext(Array2Ext)]
impl<T: Zero + Copy> Array2<T> {
    fn resize(self, (nrows, ncols): (usize, usize)) -> Self {
        if self.dim() == (nrows, ncols) {
            return self;
        }

        let mut ret = Array2::zeros((nrows, ncols));
        for (ret, this) in iter::zip(ret.rows_mut(), self.rows()) {
            for (ret, this) in iter::zip(ret, this) {
                *ret = *this;
            }
        }
        ret
    }
}

mod permit {
    use std::marker::PhantomData;

    use crate::{
        error::{SpeakerFeatureError, SpeakerFeatureErrorKind},
        metas::PermittedSynthesisMorphing,
        SpeakerMeta, StyleId,
    };

    use super::MorphingPair;

    pub(super) struct MorphableStyles<'metas> {
        inner: MorphingPair<StyleId>,
        marker: PhantomData<&'metas ()>,
    }

    impl<'metas> MorphableStyles<'metas> {
        pub(super) fn permit(
            pair: MorphingPair<(StyleId, &'metas SpeakerMeta)>,
        ) -> std::result::Result<Self, SpeakerFeatureError> {
            match pair.map(|(_, speaker)| {
                (
                    speaker.supported_features().permitted_synthesis_morphing,
                    speaker,
                )
            }) {
                MorphingPair {
                    base: (PermittedSynthesisMorphing::All, _),
                    target: (PermittedSynthesisMorphing::All, _),
                } => {}

                MorphingPair {
                    base: (PermittedSynthesisMorphing::SelfOnly, base),
                    target: (PermittedSynthesisMorphing::SelfOnly, target),
                } if base.speaker_uuid() == target.speaker_uuid() => {}

                MorphingPair {
                    base: (_, base),
                    target: (_, target),
                } => {
                    return Err(SpeakerFeatureError {
                        speaker_name: base.name().clone(),
                        speaker_uuid: base.speaker_uuid().clone(),
                        context: SpeakerFeatureErrorKind::Morph {
                            target_speaker_name: target.name().clone(),
                            target_speaker_uuid: target.speaker_uuid().clone(),
                        },
                    })
                }
            }

            Ok(Self {
                inner: pair.map(|(style_id, _)| style_id),
                marker: PhantomData,
            })
        }

        pub(super) fn get(&self) -> MorphingPair<StyleId> {
            self.inner
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use rstest::rstest;

    use super::Array2Ext as _;

    #[rstest]
    #[case(array![[1]], (2, 2), array![[1, 0], [0, 0]])]
    #[case(array![[1, 1], [1, 1]], (1, 1), array![[1]])]
    fn resize_works(
        #[case] arr: Array2<i32>,
        #[case] dim: (usize, usize),
        #[case] expected: Array2<i32>,
    ) {
        pretty_assertions::assert_eq!(expected, arr.resize(dim));
    }
}
