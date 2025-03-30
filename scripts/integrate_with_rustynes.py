#!/usr/bin/env python3
"""
CUDA-AI-Nintendo_NES_Agent - rustyNES Integration Script

This script automates the integration between the CUDA-AI agent and the
rustyNES emulator. It copies necessary files from the rustyNES repository,
modifies them for integration, and sets up the required interfaces.

Author: doublegate
License: MIT
"""

import argparse
import os
import shutil
import sys
import re
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Integrate CUDA-AI agent with rustyNES")
    parser.add_argument("--rustynes-path", type=str, default="../../Downloads/rustyNES",
                       help="Path to the rustyNES repository")
    parser.add_argument("--output-dir", type=str, default="../lib/rustynes",
                       help="Directory to store the integrated rustyNES files")
    parser.add_argument("--force", action="store_true", 
                       help="Force overwrite of existing files")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()

def ensure_dirs(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def copy_files(src_dir, dst_dir, verbose=False):
    """Copy files from source to destination."""
    if verbose:
        print(f"Copying files from {src_dir} to {dst_dir}")
    
    # Create destination directory if it doesn't exist
    ensure_dirs(dst_dir)
    
    # Get list of files in source directory
    src_files = os.listdir(src_dir)
    
    # Copy each file or directory
    for file_name in src_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        
        # Skip .git directory and specified files
        if file_name in ['.git', '.github', '.gitignore', 'target', 'LICENSE', 'README.md']:
            continue
            
        if os.path.isdir(src_path):
            # Recursively copy directories
            if verbose:
                print(f"Copying directory: {file_name}")
            copy_files(src_path, dst_path, verbose)
        else:
            # Copy files
            if verbose:
                print(f"Copying file: {file_name}")
            shutil.copy2(src_path, dst_path)

def modify_cargo_toml(cargo_path, verbose=False):
    """Modify Cargo.toml for integration."""
    if verbose:
        print(f"Modifying {cargo_path}")
    
    with open(cargo_path, 'r') as f:
        content = f.read()
    
    # Change package name
    content = re.sub(r'name = "rustynes"', 'name = "rustynes_lib"', content)
    
    # Add library config
    if '[lib]' not in content:
        content += """
[lib]
name = "rustynes_lib"
crate-type = ["rlib", "staticlib"]
"""
    
    # Write modified content back
    with open(cargo_path, 'w') as f:
        f.write(content)

def create_interface_file(output_dir, verbose=False):
    """Create interface file for the rustyNES emulator."""
    interface_path = os.path.join(output_dir, "src", "interface.rs")
    
    if verbose:
        print(f"Creating interface file: {interface_path}")
    
    interface_content = """//! Interface module for rustyNES emulation
//!
//! This module provides a clean API for interacting with the rustyNES emulator
//! from external code, specifically designed for the CUDA-AI-Nintendo_NES_Agent.
//!
//! It wraps the core emulator functionality in a way that's suitable for
//! reinforcement learning and AI agent integration.

use crate::nes::Nes;
use crate::rom::Rom;
use crate::joypad::{Joypad, JoypadButton};
use std::path::Path;
use std::io::Result;

/// Struct that provides an interface to the rustyNES emulator
pub struct EmulatorInterface {
    nes: Nes,
    frame_buffer: Vec<u8>,
    width: usize,
    height: usize,
}

impl EmulatorInterface {
    /// Create a new emulator interface and load a ROM
    pub fn new(rom_path: &Path) -> Result<Self> {
        let rom = Rom::load(rom_path)?;
        let mut nes = Nes::new(rom);
        
        // Initialize frame buffer
        let width = 256;
        let height = 240;
        let frame_buffer = vec![0; width * height * 4]; // RGBA buffer
        
        Ok(EmulatorInterface {
            nes,
            frame_buffer,
            width,
            height,
        })
    }
    
    /// Run the emulator for one frame
    pub fn step_frame(&mut self) {
        // Run until a new frame is rendered
        let mut frame_completed = false;
        while !frame_completed {
            frame_completed = self.nes.step();
        }
        
        // Copy the rendered frame to our buffer
        let ppu_frame = self.nes.frame_buffer();
        
        // Convert from RGB to RGBA
        for y in 0..self.height {
            for x in 0..self.width {
                let src_idx = (y * self.width + x) * 3; // RGB format
                let dst_idx = (y * self.width + x) * 4; // RGBA format
                
                self.frame_buffer[dst_idx] = ppu_frame[src_idx];     // R
                self.frame_buffer[dst_idx + 1] = ppu_frame[src_idx + 1]; // G
                self.frame_buffer[dst_idx + 2] = ppu_frame[src_idx + 2]; // B
                self.frame_buffer[dst_idx + 3] = 255;  // A (fully opaque)
            }
        }
    }
    
    /// Get the current frame buffer
    pub fn frame_buffer(&self) -> &[u8] {
        &self.frame_buffer
    }
    
    /// Set a button state on the joypad
    pub fn set_button(&mut self, button: JoypadButton, pressed: bool) {
        self.nes.set_button_pressed_player1(button, pressed);
    }
    
    /// Set all buttons at once using a bitmask
    pub fn set_controller_state(&mut self, state: u8) {
        // Set each button based on the bitmask
        self.set_button(JoypadButton::A, state & 0x01 != 0);
        self.set_button(JoypadButton::B, state & 0x02 != 0);
        self.set_button(JoypadButton::Select, state & 0x04 != 0);
        self.set_button(JoypadButton::Start, state & 0x08 != 0);
        self.set_button(JoypadButton::Up, state & 0x10 != 0);
        self.set_button(JoypadButton::Down, state & 0x20 != 0);
        self.set_button(JoypadButton::Left, state & 0x40 != 0);
        self.set_button(JoypadButton::Right, state & 0x80 != 0);
    }
    
    /// Save the current state to a file
    pub fn save_state(&self, path: &Path) -> Result<()> {
        // TODO: Implement state saving
        Ok(())
    }
    
    /// Load a state from a file
    pub fn load_state(&mut self, path: &Path) -> Result<()> {
        // TODO: Implement state loading
        Ok(())
    }
    
    /// Get the width of the frame buffer
    pub fn width(&self) -> usize {
        self.width
    }
    
    /// Get the height of the frame buffer
    pub fn height(&self) -> usize {
        self.height
    }
    
    /// Read a memory address from the NES
    pub fn read_memory(&self, address: u16) -> u8 {
        self.nes.read_memory(address)
    }
    
    /// Write to a memory address in the NES
    pub fn write_memory(&mut self, address: u16, value: u8) {
        self.nes.write_memory(address, value);
    }
    
    /// Reset the emulator
    pub fn reset(&mut self) {
        self.nes.reset();
    }
}

/// Converts an EmulatorInterface JoypadButton to a controller bitmask
pub fn button_to_bitmask(button: JoypadButton) -> u8 {
    match button {
        JoypadButton::A => 0x01,
        JoypadButton::B => 0x02,
        JoypadButton::Select => 0x04,
        JoypadButton::Start => 0x08,
        JoypadButton::Up => 0x10,
        JoypadButton::Down => 0x20,
        JoypadButton::Left => 0x40,
        JoypadButton::Right => 0x80,
    }
}

/// FFI interface for use with C/C++/CUDA code
#[no_mangle]
pub extern "C" fn rustynes_create(rom_path_ptr: *const i8) -> *mut EmulatorInterface {
    use std::ffi::CStr;
    
    if rom_path_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let rom_path_c_str = unsafe { CStr::from_ptr(rom_path_ptr) };
    let rom_path_str = match rom_path_c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let rom_path = Path::new(rom_path_str);
    
    match EmulatorInterface::new(rom_path) {
        Ok(interface) => Box::into_raw(Box::new(interface)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rustynes_destroy(interface_ptr: *mut EmulatorInterface) {
    if !interface_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(interface_ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn rustynes_step_frame(interface_ptr: *mut EmulatorInterface) {
    if !interface_ptr.is_null() {
        unsafe {
            (*interface_ptr).step_frame();
        }
    }
}

#[no_mangle]
pub extern "C" fn rustynes_get_frame_buffer(interface_ptr: *const EmulatorInterface) -> *const u8 {
    if interface_ptr.is_null() {
        return std::ptr::null();
    }
    
    unsafe {
        (*interface_ptr).frame_buffer().as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn rustynes_set_controller_state(interface_ptr: *mut EmulatorInterface, state: u8) {
    if !interface_ptr.is_null() {
        unsafe {
            (*interface_ptr).set_controller_state(state);
        }
    }
}

#[no_mangle]
pub extern "C" fn rustynes_get_width(interface_ptr: *const EmulatorInterface) -> i32 {
    if interface_ptr.is_null() {
        return 0;
    }
    
    unsafe {
        (*interface_ptr).width() as i32
    }
}

#[no_mangle]
pub extern "C" fn rustynes_get_height(interface_ptr: *const EmulatorInterface) -> i32 {
    if interface_ptr.is_null() {
        return 0;
    }
    
    unsafe {
        (*interface_ptr).height() as i32
    }
}

#[no_mangle]
pub extern "C" fn rustynes_read_memory(interface_ptr: *const EmulatorInterface, address: u16) -> u8 {
    if interface_ptr.is_null() {
        return 0;
    }
    
    unsafe {
        (*interface_ptr).read_memory(address)
    }
}

#[no_mangle]
pub extern "C" fn rustynes_write_memory(interface_ptr: *mut EmulatorInterface, address: u16, value: u8) {
    if !interface_ptr.is_null() {
        unsafe {
            (*interface_ptr).write_memory(address, value);
        }
    }
}

#[no_mangle]
pub extern "C" fn rustynes_reset(interface_ptr: *mut EmulatorInterface) {
    if !interface_ptr.is_null() {
        unsafe {
            (*interface_ptr).reset();
        }
    }
}
"""
    
    # Ensure the directory exists
    ensure_dirs(os.path.dirname(interface_path))
    
    # Write the interface file
    with open(interface_path, 'w') as f:
        f.write(interface_content)

def modify_lib_rs(lib_path, verbose=False):
    """Modify lib.rs to export the interface module."""
    if verbose:
        print(f"Modifying {lib_path}")
    
    # Check if file exists
    if not os.path.exists(lib_path):
        if verbose:
            print(f"Warning: {lib_path} not found, creating new file")
        
        with open(lib_path, 'w') as f:
            f.write("""//! rustyNES library for integration with CUDA-AI-Nintendo_NES_Agent
//!
//! This library provides access to the rustyNES emulator functionality
//! through a clean interface designed for AI agent integration.

// Export core modules
pub mod apu;
pub mod bus;
pub mod cartridge;
pub mod cpu;
pub mod joypad;
pub mod nes;
pub mod ppu;
pub mod rom;

// Export interface module
pub mod interface;

// Re-export key types for convenience
pub use interface::EmulatorInterface;
pub use joypad::JoypadButton;
""")
        return
    
    # Read existing content
    with open(lib_path, 'r') as f:
        content = f.read()
    
    # Check if interface module is already included
    if "pub mod interface;" not in content:
        # Add interface module
        content += "\n// Export interface module\npub mod interface;\n"
        
        # Add re-exports
        content += "\n// Re-export key types for convenience\npub use interface::EmulatorInterface;\npub use joypad::JoypadButton;\n"
    
    # Write modified content back
    with open(lib_path, 'w') as f:
        f.write(content)

def create_c_header(output_dir, verbose=False):
    """Create C header file for the rustyNES emulator interface."""
    header_path = os.path.join(output_dir, "include", "rustynes_interface.h")
    
    if verbose:
        print(f"Creating C header file: {header_path}")
    
    header_content = """/**
 * rustyNES Emulator C Interface
 * 
 * This header provides a C interface to the rustyNES emulator for use with
 * the CUDA-AI-Nintendo_NES_Agent project.
 * 
 * Author: doublegate
 * License: MIT
 */

#ifndef RUSTYNES_INTERFACE_H
#define RUSTYNES_INTERFACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to the emulator interface */
typedef struct EmulatorInterface EmulatorInterface;

/**
 * Create a new emulator interface and load a ROM
 * 
 * @param rom_path Path to the ROM file
 * @return Handle to the emulator interface, or NULL on error
 */
EmulatorInterface* rustynes_create(const char* rom_path);

/**
 * Destroy an emulator interface
 * 
 * @param interface Handle to the emulator interface
 */
void rustynes_destroy(EmulatorInterface* interface);

/**
 * Run the emulator for one frame
 * 
 * @param interface Handle to the emulator interface
 */
void rustynes_step_frame(EmulatorInterface* interface);

/**
 * Get the current frame buffer (RGBA format)
 * 
 * @param interface Handle to the emulator interface
 * @return Pointer to the frame buffer, or NULL on error
 */
const uint8_t* rustynes_get_frame_buffer(const EmulatorInterface* interface);

/**
 * Set all controller buttons at once using a bitmask
 * 
 * Bitmask format:
 * - Bit 0: A
 * - Bit 1: B
 * - Bit 2: Select
 * - Bit 3: Start
 * - Bit 4: Up
 * - Bit 5: Down
 * - Bit 6: Left
 * - Bit 7: Right
 * 
 * @param interface Handle to the emulator interface
 * @param state Controller state bitmask
 */
void rustynes_set_controller_state(EmulatorInterface* interface, uint8_t state);

/**
 * Get the width of the frame buffer
 * 
 * @param interface Handle to the emulator interface
 * @return Width of the frame buffer in pixels
 */
int32_t rustynes_get_width(const EmulatorInterface* interface);

/**
 * Get the height of the frame buffer
 * 
 * @param interface Handle to the emulator interface
 * @return Height of the frame buffer in pixels
 */
int32_t rustynes_get_height(const EmulatorInterface* interface);

/**
 * Read a memory address from the NES
 * 
 * @param interface Handle to the emulator interface
 * @param address Memory address to read
 * @return Value at the memory address
 */
uint8_t rustynes_read_memory(const EmulatorInterface* interface, uint16_t address);

/**
 * Write to a memory address in the NES
 * 
 * @param interface Handle to the emulator interface
 * @param address Memory address to write to
 * @param value Value to write
 */
void rustynes_write_memory(EmulatorInterface* interface, uint16_t address, uint8_t value);

/**
 * Reset the emulator
 * 
 * @param interface Handle to the emulator interface
 */
void rustynes_reset(EmulatorInterface* interface);

/**
 * Button constants for player 1 controller
 */
typedef enum {
    RUSTYNES_BUTTON_A      = 0x01,
    RUSTYNES_BUTTON_B      = 0x02,
    RUSTYNES_BUTTON_SELECT = 0x04,
    RUSTYNES_BUTTON_START  = 0x08,
    RUSTYNES_BUTTON_UP     = 0x10,
    RUSTYNES_BUTTON_DOWN   = 0x20,
    RUSTYNES_BUTTON_LEFT   = 0x40,
    RUSTYNES_BUTTON_RIGHT  = 0x80
} RustyNesButton;

#ifdef __cplusplus
}
#endif

#endif // RUSTYNES_INTERFACE_H
"""
    
    # Ensure the directory exists
    ensure_dirs(os.path.dirname(header_path))
    
    # Write the header file
    with open(header_path, 'w') as f:
        f.write(header_content)

def create_integration_readme(output_dir, verbose=False):
    """Create README file for the rustyNES integration."""
    readme_path = os.path.join(output_dir, "README_INTEGRATION.md")
    
    if verbose:
        print(f"Creating integration README: {readme_path}")
    
    readme_content = """# rustyNES Integration for CUDA-AI-Nintendo_NES_Agent

This directory contains an integrated version of the rustyNES emulator, specifically modified for use with the CUDA-AI-Nintendo_NES_Agent project.

## Modifications

The original rustyNES emulator has been modified in the following ways:

1. Added an `interface.rs` module that provides a clean API for external code
2. Added C-compatible FFI functions for integration with C++/CUDA code
3. Modified the build configuration to produce a static library
4. Added a C header file for easy inclusion in C/C++ code

## Usage

### From Rust

```rust
use rustynes_lib::EmulatorInterface;
use std::path::Path;

// Create a new emulator instance and load a ROM
let mut emu = EmulatorInterface::new(Path::new("path/to/rom.nes")).unwrap();

// Run for one frame
emu.step_frame();

// Get the frame buffer (RGBA format)
let frame_buffer = emu.frame_buffer();

// Set controller input
emu.set_controller_state(0x80); // Press the Right button
```

### From C/C++

```c
#include "rustynes_interface.h"

// Create a new emulator instance and load a ROM
EmulatorInterface* emu = rustynes_create("path/to/rom.nes");

// Run for one frame
rustynes_step_frame(emu);

// Get the frame buffer (RGBA format)
const uint8_t* frame_buffer = rustynes_get_frame_buffer(emu);

// Set controller input
rustynes_set_controller_state(emu, RUSTYNES_BUTTON_RIGHT); // Press the Right button

// Clean up when done
rustynes_destroy(emu);
```

## Memory Map

For reference, here is a brief overview of the NES memory map:

- `0x0000-0x07FF`: 2KB of internal RAM
- `0x0800-0x1FFF`: Mirrors of internal RAM
- `0x2000-0x2007`: PPU registers
- `0x2008-0x3FFF`: Mirrors of PPU registers
- `0x4000-0x4017`: APU and I/O registers
- `0x4018-0x401F`: APU and I/O functionality that is normally disabled
- `0x4020-0xFFFF`: Cartridge space: PRG ROM, PRG RAM, and mapper registers

## Game-Specific Memory Locations

### Super Mario Bros.

- `0x075A`: Player X position (screen)
- `0x075F`: Player X position (world)
- `0x0776`: Player Y position
- `0x0747`: Score (BCD)
- `0x075C`: Powerup state (0: small, 1: big, 2: fire flower)
- `0x0770`: Player is dying flag
- `0x000E`: World number
- `0x000F`: Level number

## Building

To build the library:

```bash
cd lib/rustynes
cargo build --release
```

This will produce a static library in `target/release/librustynes_lib.a` (or `rustynes_lib.lib` on Windows).

## Credits

This integration is based on the original [rustyNES](https://github.com/doublegate/rustyNES) emulator, with modifications for AI agent integration.
"""
    
    # Write the README file
    with open(readme_path, 'w') as f:
        f.write(readme_content)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).resolve().parent
    rustynes_path = Path(args.rustynes_path).resolve()
    output_dir = (script_dir.parent / args.output_dir).resolve()
    
    if args.verbose:
        print(f"rustyNES path: {rustynes_path}")
        print(f"Output directory: {output_dir}")
    
    # Check if rustyNES path exists
    if not rustynes_path.exists():
        print(f"Error: rustyNES path does not exist: {rustynes_path}")
        return 1
    
    # Check if output directory exists and is not empty
    if output_dir.exists() and os.listdir(output_dir) and not args.force:
        print(f"Error: Output directory exists and is not empty: {output_dir}")
        print("Use --force to overwrite existing files")
        return 1
    
    # Create output directory
    ensure_dirs(output_dir)
    
    # Copy files
    copy_files(rustynes_path, output_dir, args.verbose)
    
    # Modify Cargo.toml
    cargo_path = output_dir / "Cargo.toml"
    modify_cargo_toml(cargo_path, args.verbose)
    
    # Create interface file
    create_interface_file(output_dir, args.verbose)
    
    # Modify lib.rs
    lib_path = output_dir / "src" / "lib.rs"
    modify_lib_rs(lib_path, args.verbose)
    
    # Create include directory
    ensure_dirs(output_dir / "include")
    
    # Create C header
    create_c_header(output_dir, args.verbose)
    
    # Create integration README
    create_integration_readme(output_dir, args.verbose)
    
    print("rustyNES integration completed successfully!")
    print(f"The integrated library is available at: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
