/**
 * CUDA-AI-Nintendo_NES_Agent - Utilities Module
 * 
 * This module provides various utility functions for the CUDA-AI-Nintendo_NES_Agent
 * project, including frame processing, data conversion, logging, and performance
 * measurement.
 * 
 * These utilities help with common tasks used throughout the application.
 * 
 * Author: doublegate
 * License: MIT
 */

use std::time::{Duration, Instant};
use std::error::Error;

/// Frame processing utilities for NES screen data
pub mod frame {
    /// Converts RGB frame data to grayscale
    pub fn to_grayscale(rgb_data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut grayscale = Vec::with_capacity(width * height);
        
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3; // RGB format
                let r = rgb_data[offset] as f32;
                let g = rgb_data[offset + 1] as f32;
                let b = rgb_data[offset + 2] as f32;
                
                // Standard grayscale conversion weights
                let gray = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                grayscale.push(gray);
            }
        }
        
        grayscale
    }
    
    /// Downscales a frame to the given dimensions
    pub fn downscale(data: &[u8], src_width: usize, src_height: usize, 
                    dst_width: usize, dst_height: usize) -> Vec<u8> {
        let channels = data.len() / (src_width * src_height);
        let mut result = vec![0u8; dst_width * dst_height * channels];
        
        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;
        
        for y in 0..dst_height {
            for x in 0..dst_width {
                let px = (x as f32 * x_ratio) as usize;
                let py = (y as f32 * y_ratio) as usize;
                
                let src_offset = (py * src_width + px) * channels;
                let dst_offset = (y * dst_width + x) * channels;
                
                for c in 0..channels {
                    result[dst_offset + c] = data[src_offset + c];
                }
            }
        }
        
        result
    }
    
    /// Normalizes pixel values to the range [0.0, 1.0]
    pub fn normalize(data: &[u8]) -> Vec<f32> {
        data.iter().map(|&x| x as f32 / 255.0).collect()
    }
}

/// Performance measurement utilities
pub struct Timer {
    start: Instant,
    splits: Vec<(String, Duration)>,
}

impl Timer {
    /// Creates a new timer and starts it
    pub fn new() -> Self {
        Timer {
            start: Instant::now(),
            splits: Vec::new(),
        }
    }
    
    /// Records a split with the given name
    pub fn split(&mut self, name: &str) {
        let elapsed = self.start.elapsed();
        self.splits.push((name.to_string(), elapsed));
    }
    
    /// Returns the total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    /// Prints a summary of all splits
    pub fn print_summary(&self) {
        println!("Performance summary:");
        
        let mut prev_duration = Duration::from_secs(0);
        
        for (i, (name, duration)) in self.splits.iter().enumerate() {
            let split_duration = if i == 0 {
                *duration
            } else {
                *duration - prev_duration
            };
            
            println!("  {}: {:.2}ms", name, split_duration.as_secs_f64() * 1000.0);
            prev_duration = *duration;
        }
        
        println!("Total: {:.2}ms", self.elapsed().as_secs_f64() * 1000.0);
    }
}

/// Logging utilities with different verbosity levels
pub mod log {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static LOG_LEVEL: AtomicUsize = AtomicUsize::new(1); // 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE
    
    /// Sets the logging verbosity level
    pub fn set_level(level: usize) {
        LOG_LEVEL.store(level, Ordering::SeqCst);
    }
    
    /// Gets the current logging verbosity level
    pub fn get_level() -> usize {
        LOG_LEVEL.load(Ordering::SeqCst)
    }
    
    /// Logs an error message
    pub fn error(msg: &str) {
        if get_level() >= 1 {
            eprintln!("[ERROR] {}", msg);
        }
    }
    
    /// Logs a warning message
    pub fn warn(msg: &str) {
        if get_level() >= 2 {
            println!("[WARN] {}", msg);
        }
    }
    
    /// Logs an informational message
    pub fn info(msg: &str) {
        if get_level() >= 3 {
            println!("[INFO] {}", msg);
        }
    }
    
    /// Logs a debug message
    pub fn debug(msg: &str) {
        if get_level() >= 4 {
            println!("[DEBUG] {}", msg);
        }
    }
    
    /// Logs a trace message
    pub fn trace(msg: &str) {
        if get_level() >= 5 {
            println!("[TRACE] {}", msg);
        }
    }
}
