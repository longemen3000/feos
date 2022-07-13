#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dft")]
mod dft;
mod eos;
pub(crate) mod parameters;

#[cfg(feature = "dft")]
pub use dft::PcSaftFunctional;
pub use eos::{DQVariants, PcSaft, PcSaftOptions};
pub use parameters::{PcSaftParameters, PcSaftRecord};

#[cfg(feature = "python")]
pub mod python;
