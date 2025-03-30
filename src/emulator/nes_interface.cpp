/**
 * CUDA-AI-Nintendo_NES_Agent - NES Emulator Interface Implementation
 * 
 * This file implements the C++ wrapper around the rustyNES emulator's
 * C interface, connecting the AI agent with the emulator.
 * 
 * Author: doublegate
 * License: MIT
 */

#include "emulator/nes_interface.h"
#include "lib/rustynes/include/rustynes_interface.h"

#include <cstring>
#include <iostream>

namespace emulator {

NesEmulator::NesEmulator(const std::string& romPath) : m_interface(nullptr) {
    m_interface = rustynes_create(romPath.c_str());
    
    if (!m_interface) {
        throw NesEmulatorException("Failed to create NES emulator. Check if the ROM file exists and is a valid NES ROM.");
    }
}

NesEmulator::~NesEmulator() {
    if (m_interface) {
        rustynes_destroy(m_interface);
        m_interface = nullptr;
    }
}

NesEmulator::NesEmulator(NesEmulator&& other) noexcept : m_interface(other.m_interface) {
    other.m_interface = nullptr;
}

NesEmulator& NesEmulator::operator=(NesEmulator&& other) noexcept {
    if (this != &other) {
        if (m_interface) {
            rustynes_destroy(m_interface);
        }
        
        m_interface = other.m_interface;
        other.m_interface = nullptr;
    }
    
    return *this;
}

void NesEmulator::stepFrame() {
    validateEmulator();
    rustynes_step_frame(m_interface);
}

FrameBuffer NesEmulator::getFrameBuffer() const {
    validateEmulator();
    
    int width = getWidth();
    int height = getHeight();
    FrameBuffer buffer(width, height);
    
    const uint8_t* frameData = rustynes_get_frame_buffer(m_interface);
    std::memcpy(buffer.data.data(), frameData, width * height * 4);
    
    return buffer;
}

int NesEmulator::getWidth() const {
    validateEmulator();
    return rustynes_get_width(m_interface);
}

int NesEmulator::getHeight() const {
    validateEmulator();
    return rustynes_get_height(m_interface);
}

void NesEmulator::setControllerState(const ControllerState& state) {
    validateEmulator();
    rustynes_set_controller_state(m_interface, state.toBitmask());
}

void NesEmulator::setControllerState(uint8_t bitmask) {
    validateEmulator();
    rustynes_set_controller_state(m_interface, bitmask);
}

void NesEmulator::setButton(Button button, bool pressed) {
    validateEmulator();
    
    // Get current controller state
    uint8_t currentState = 0; // We don't have a direct way to get this from the C API
    // If button is pressed, set the bit; otherwise, clear it
    if (pressed) {
        currentState |= static_cast<uint8_t>(button);
    } else {
        currentState &= ~static_cast<uint8_t>(button);
    }
    
    rustynes_set_controller_state(m_interface, currentState);
}

uint8_t NesEmulator::readMemory(uint16_t address) const {
    validateEmulator();
    return rustynes_read_memory(m_interface, address);
}

void NesEmulator::writeMemory(uint16_t address, uint8_t value) {
    validateEmulator();
    rustynes_write_memory(m_interface, address, value);
}

void NesEmulator::reset() {
    validateEmulator();
    rustynes_reset(m_interface);
}

GameInfo NesEmulator::extractGameInfo() const {
    validateEmulator();
    
    GameInfo info;
    info.mario.extract(*this);
    
    return info;
}

void NesEmulator::saveState(const std::string& path) const {
    validateEmulator();
    // TODO: Implement when rustyNES supports state saving
    std::cerr << "Warning: State saving not yet implemented in rustyNES" << std::endl;
}

void NesEmulator::loadState(const std::string& path) {
    validateEmulator();
    // TODO: Implement when rustyNES supports state loading
    std::cerr << "Warning: State loading not yet implemented in rustyNES" << std::endl;
}

bool NesEmulator::isValid() const {
    return m_interface != nullptr;
}

EmulatorInterface* NesEmulator::getNativeHandle() {
    return m_interface;
}

const EmulatorInterface* NesEmulator::getNativeHandle() const {
    return m_interface;
}

void NesEmulator::validateEmulator() const {
    if (!isValid()) {
        throw NesEmulatorException("Invalid NES emulator instance");
    }
}

} // namespace emulator
