pub(crate) mod domains;
mod model_file;
pub(crate) mod runtimes;
pub(crate) mod status;

use std::{borrow::Cow, collections::BTreeSet, fmt::Debug};

use derive_new::new;
use duplicate::duplicate_item;
use enum_map::{Enum, EnumMap};
use ndarray::{Array, ArrayD, Dimension, ShapeError};
use thiserror::Error;

use crate::{StyleType, SupportedDevices};

pub(crate) trait InferenceRuntime: 'static {
    type Session: Sized + Send + 'static;
    type RunContext<'a>: From<&'a mut Self::Session> + PushInputTensor;

    fn supported_devices() -> crate::Result<SupportedDevices>;

    #[allow(clippy::type_complexity)]
    fn new_session(
        model: impl FnOnce() -> std::result::Result<Vec<u8>, DecryptModelError>,
        options: InferenceSessionOptions,
    ) -> anyhow::Result<(
        Self::Session,
        Vec<ParamInfo<InputScalarKind>>,
        Vec<ParamInfo<OutputScalarKind>>,
    )>;

    fn run(ctx: Self::RunContext<'_>) -> anyhow::Result<Vec<OutputTensor>>;
}

pub(crate) trait InferenceDomainGroup: Sized {
    type Map<V: InferenceDomainMapValueProjection>: InferenceDomainMap<
        Group = Self,
        ValueProjection = V,
    >;
}

pub(crate) trait InferenceDomainMap {
    type Group: InferenceDomainGroup;
    type ValueProjection: InferenceDomainMapValueProjection;

    fn any(
        &self,
        p: impl InferenceDomainMapValuePredicate<InputProjection = Self::ValueProjection>,
    ) -> bool;

    fn ref_map<
        F: InferenceDomainMapValueFunction<
            Group = Self::Group,
            InputProjection = Self::ValueProjection,
        >,
    >(
        &self,
        f: F,
    ) -> <Self::Group as InferenceDomainGroup>::Map<F::OutputProjection>;

    fn try_ref_map<
        F: InferenceDomainMapValueTryFunction<
            Group = Self::Group,
            InputProjection = Self::ValueProjection,
        >,
    >(
        &self,
        f: F,
    ) -> Result<<Self::Group as InferenceDomainGroup>::Map<F::OutputProjection>, F::Error>;
}

pub(crate) trait InferenceDomainMapValuePredicate {
    type InputProjection: InferenceDomainMapValueProjection;

    fn test<D: InferenceDomain>(
        &self,
        x: &<Self::InputProjection as InferenceDomainMapValueProjection>::Target<D>,
    ) -> bool;
}

pub(crate) trait InferenceDomainMapValueFunction {
    type Group: InferenceDomainGroup;
    type InputProjection: InferenceDomainMapValueProjection;
    type OutputProjection: InferenceDomainMapValueProjection;

    fn apply<D: InferenceDomain<Group = Self::Group>>(
        &self,
        x: &<Self::InputProjection as InferenceDomainMapValueProjection>::Target<D>,
    ) -> <Self::OutputProjection as InferenceDomainMapValueProjection>::Target<D>;
}

pub(crate) trait InferenceDomainMapValueTryFunction {
    type Group: InferenceDomainGroup;
    type InputProjection: InferenceDomainMapValueProjection;
    type OutputProjection: InferenceDomainMapValueProjection;
    type Error;

    fn apply<D: InferenceDomain<Group = Self::Group>>(
        &self,
        x: &<Self::InputProjection as InferenceDomainMapValueProjection>::Target<D>,
    ) -> Result<<Self::OutputProjection as InferenceDomainMapValueProjection>::Target<D>, Self::Error>;
}

pub(crate) trait InferenceDomainMapValueProjection {
    type Target<D: InferenceDomain>;
}

impl<V1: InferenceDomainMapValueProjection, V2: InferenceDomainMapValueProjection>
    InferenceDomainMapValueProjection for (V1, V2)
{
    type Target<D: InferenceDomain> = (V1::Target<D>, V2::Target<D>);
}

impl<V: InferenceDomainMapValueProjection> InferenceDomainMapValueProjection for Option<V> {
    type Target<D: InferenceDomain> = Option<V::Target<D>>;
}

pub(crate) struct ForAllInferenceDomain<T>(pub(crate) T);

impl<T> InferenceDomainMapValueProjection for ForAllInferenceDomain<T> {
    type Target<D: InferenceDomain> = T;
}

/// ある`VoiceModel`が提供する推論操作の集合を示す。
pub(crate) trait InferenceDomain: Sized {
    type Group: InferenceDomainGroup;
    type Operation: InferenceOperation;

    fn style_types() -> &'static BTreeSet<StyleType>;

    fn visit<V: InferenceDomainMapValueProjection>(
        map: &<Self::Group as InferenceDomainGroup>::Map<V>,
    ) -> &V::Target<Self>;
}

/// `InferenceDomain`の推論操作を表す列挙型。
///
/// それぞれのバリアントには、対応する`InferenceSignature`が存在するべきである。
///
/// `::macros::InferenceOperation`により導出される。
pub(crate) trait InferenceOperation: Copy + Enum {
    /// `{InferenceInputSignature,InferenceOutputSignature}::PARAM_INFOS`を集めたもの。
    #[allow(clippy::type_complexity)]
    const PARAM_INFOS: EnumMap<
        Self,
        (
            &'static [ParamInfo<InputScalarKind>],
            &'static [ParamInfo<OutputScalarKind>],
        ),
    >;
}

/// `InferenceDomain`の推論操作を表す列挙型。
///
/// `::macros::InferenceOperation`により、具体型ごと生成される。
pub(crate) trait InferenceSignature: Sized + Send + 'static {
    type Domain: InferenceDomain;
    type Input: InferenceInputSignature<Signature = Self>;
    type Output: InferenceOutputSignature;
    const OPERATION: <Self::Domain as InferenceDomain>::Operation;
}

/// 推論操作の入力シグネチャ。
///
/// `::macros::InferenceInputSignature`により導出される。
pub(crate) trait InferenceInputSignature: Send + 'static {
    type Signature: InferenceSignature<Input = Self>;
    const PARAM_INFOS: &'static [ParamInfo<InputScalarKind>];
    fn make_run_context<R: InferenceRuntime>(self, sess: &mut R::Session) -> R::RunContext<'_>;
}

pub(crate) trait InputScalar: Sized {
    const KIND: InputScalarKind;

    fn push_tensor_to_ctx(
        tensor: Array<Self, impl Dimension + 'static>,
        visitor: &mut impl PushInputTensor,
    );
}

#[duplicate_item(
    T       KIND_VAL                     push;
    [ i64 ] [ InputScalarKind::Int64 ]   [ push_int64 ];
    [ f32 ] [ InputScalarKind::Float32 ] [ push_float32 ];
)]
impl InputScalar for T {
    const KIND: InputScalarKind = KIND_VAL;

    fn push_tensor_to_ctx(
        tensor: Array<Self, impl Dimension + 'static>,
        ctx: &mut impl PushInputTensor,
    ) {
        ctx.push(tensor);
    }
}

#[derive(Clone, Copy, PartialEq, derive_more::Display)]
pub(crate) enum InputScalarKind {
    #[display(fmt = "int64_t")]
    Int64,

    #[display(fmt = "float")]
    Float32,
}

pub(crate) trait PushInputTensor {
    fn push_int64(&mut self, tensor: Array<i64, impl Dimension + 'static>);
    fn push_float32(&mut self, tensor: Array<f32, impl Dimension + 'static>);
}

/// 推論操作の出力シグネチャ。
///
/// `::macros::InferenceOutputSignature`により、`TryFrom<OutputTensor>`も含めて導出される。
pub(crate) trait InferenceOutputSignature:
    TryFrom<Vec<OutputTensor>, Error = anyhow::Error> + Send
{
    const PARAM_INFOS: &'static [ParamInfo<OutputScalarKind>];
}

pub(crate) trait OutputScalar: Sized {
    const KIND: OutputScalarKind;
    fn extract(tensor: OutputTensor) -> std::result::Result<ArrayD<Self>, ExtractError>;
}

impl OutputScalar for f32 {
    const KIND: OutputScalarKind = OutputScalarKind::Float32;

    fn extract(tensor: OutputTensor) -> std::result::Result<ArrayD<Self>, ExtractError> {
        match tensor {
            OutputTensor::Float32(tensor) => Ok(tensor),
        }
    }
}

#[derive(Clone, Copy, PartialEq, derive_more::Display)]
pub(crate) enum OutputScalarKind {
    #[display(fmt = "float")]
    Float32,
}

pub(crate) enum OutputTensor {
    Float32(ArrayD<f32>),
}

impl<A: OutputScalar, D: Dimension> TryFrom<OutputTensor> for Array<A, D> {
    type Error = ExtractError;

    fn try_from(tensor: OutputTensor) -> Result<Self, Self::Error> {
        let this = A::extract(tensor)?.into_dimensionality()?;
        Ok(this)
    }
}

pub(crate) struct ParamInfo<D> {
    name: Cow<'static, str>,
    dt: D,
    ndim: Option<usize>,
}

impl<D: PartialEq> ParamInfo<D> {
    fn accepts(&self, other: &Self) -> bool {
        self.name == other.name
            && self.dt == other.dt
            && (self.ndim.is_none() || self.ndim == other.ndim)
    }
}

#[derive(new, Clone, Copy, PartialEq, Debug)]
pub(crate) struct InferenceSessionOptions {
    pub(crate) cpu_num_threads: u16,
    pub(crate) use_gpu: bool,
}

#[derive(Error, Debug)]
pub(crate) enum ExtractError {
    #[error(transparent)]
    Shape(#[from] ShapeError),
}

#[derive(Error, Debug)]
#[error("不正なモデルファイルです")]
pub(crate) struct DecryptModelError;
