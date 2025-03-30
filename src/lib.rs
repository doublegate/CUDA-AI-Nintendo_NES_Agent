/**
 * CUDA-AI-Nintendo_NES_Agent - Library Module
 * 
 * This is the main library file for the CUDA-AI-Nintendo_NES_Agent project.
 * It exposes the public API for the library and re-exports the key components
 * needed by applications that use this library.
 * 
 * The library provides functionality for NES emulation with CUDA-accelerated
 * AI agents that can learn to play NES games.
 * 
 * Author: doublegate
 * License: MIT
 */

// Re-export main modules
pub mod emulator;
pub mod ai_agent;
pub mod cuda_interface;
pub mod utils;

// Export key components for library users
pub use emulator::{NesEmulator, ControllerInput};
pub use ai_agent::{Agent, TrainingConfig};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the library version
pub fn version() -> &'static str {
    VERSION
}

/// Initializes the library
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA-AI-Nintendo_NES_Agent library v{}", VERSION);
    
    // Initialize CUDA
    cuda_interface::initialize_cuda()?;
    
    Ok(())
}
