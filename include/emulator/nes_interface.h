/**
 * CUDA-AI-Nintendo_NES_Agent - NES Emulator Interface
 * 
 * This header file provides a C++ wrapper around the rustyNES emulator's
 * C interface, integrating it with the CUDA-AI agent.
 * 
 * It handles the communication between the AI agent and the emulator,
 * providing methods for frame processing, controller input, and
 * game state extraction.
 * 
 * Author: doublegate
 * License: MIT
 */

#ifndef NES_INTERFACE_H
#define NES_INTERFACE_H

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <functional>

// Forward declaration of the rustyNES emulator interface
struct EmulatorInterface;

namespace emulator {

/**
 * Button constants for NES controller
 */
enum class Button {
    A      = 0x01,
    B      = 0x02,
    SELECT = 0x04,
    START  = 0x08,
    UP     = 0x10,
    DOWN   = 0x20,
    LEFT   = 0x40,
    RIGHT  = 0x80
};

/**
 * Controller state representation
 */
struct ControllerState {
    bool a;
    bool b;
    bool select;
    bool start;
    bool up;
    bool down;
    bool left;
    bool right;
    
    /**
     * Convert to bitmask format used by the emulator
     */
    uint8_t toBitmask() const {
        uint8_t mask = 0;
        if (a)      mask |= static_cast<uint8_t>(Button::A);
        if (b)      mask |= static_cast<uint8_t>(Button::B);
        if (select) mask |= static_cast<uint8_t>(Button::SELECT);
        if (start)  mask |= static_cast<uint8_t>(Button::START);
        if (up)     mask |= static_cast<uint8_t>(Button::UP);
        if (down)   mask |= static_cast<uint8_t>(Button::DOWN);
        if (left)   mask |= static_cast<uint8_t>(Button::LEFT);
        if (right)  mask |= static_cast<uint8_t>(Button::RIGHT);
        return mask;
    }
    
    /**
     * Create from bitmask
     */
    static ControllerState fromBitmask(uint8_t mask) {
        ControllerState state;
        state.a      = (mask & static_cast<uint8_t>(Button::A)) != 0;
        state.b      = (mask & static_cast<uint8_t>(Button::B)) != 0;
        state.select = (mask & static_cast<uint8_t>(Button::SELECT)) != 0;
        state.start  = (mask & static_cast<uint8_t>(Button::START)) != 0;
        state.up     = (mask & static_cast<uint8_t>(Button::UP)) != 0;
        state.down   = (mask & static_cast<uint8_t>(Button::DOWN)) != 0;
        state.left   = (mask & static_cast<uint8_t>(Button::LEFT)) != 0;
        state.right  = (mask & static_cast<uint8_t>(Button::RIGHT)) != 0;
        return state;
    }
};

/**
 * NES frame buffer representation
 */
struct FrameBuffer {
    int width;
    int height;
    std::vector<uint8_t> data; // RGBA format
    
    /**
     * Default constructor
     */
    FrameBuffer() : width(0), height(0) {}
    
    /**
     * Constructor with dimensions
     */
    FrameBuffer(int w, int h) : width(w), height(h), data(w * h * 4, 0) {}
    
    /**
     * Check if frame buffer is valid
     */
    bool isValid() const {
        return width > 0 && height > 0 && data.size() == width * height * 4;
    }
    
    /**
     * Get pixel at specified position
     * @return RGBA pixel value (r, g, b, a)
     */
    std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> getPixel(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return std::make_tuple(0, 0, 0, 0);
        }
        
        size_t offset = (y * width + x) * 4;
        return std::make_tuple(
            data[offset],     // R
            data[offset + 1], // G
            data[offset + 2], // B
            data[offset + 3]  // A
        );
    }
    
    /**
     * Convert to grayscale
     */
    std::vector<float> toGrayscale() const {
        std::vector<float> grayscale(width * height, 0.0f);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto [r, g, b, a] = getPixel(x, y);
                
                // Use standard grayscale conversion formula
                float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                grayscale[y * width + x] = gray;
            }
        }
        
        return grayscale;
    }
    
    /**
     * Downscale to specified dimensions
     */
    FrameBuffer downscale(int targetWidth, int targetHeight) const {
        FrameBuffer result(targetWidth, targetHeight);
        
        float xRatio = static_cast<float>(width) / targetWidth;
        float yRatio = static_cast<float>(height) / targetHeight;
        
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                int srcX = static_cast<int>(x * xRatio);
                int srcY = static_cast<int>(y * yRatio);
                
                auto [r, g, b, a] = getPixel(srcX, srcY);
                
                size_t targetOffset = (y * targetWidth + x) * 4;
                result.data[targetOffset] = r;
                result.data[targetOffset + 1] = g;
                result.data[targetOffset + 2] = b;
                result.data[targetOffset + 3] = a;
            }
        }
        
        return result;
    }
};

/**
 * Game-specific information for reward calculation
 */
struct GameInfo {
    // Super Mario Bros. specific
    struct SuperMarioBros {
        uint8_t playerX;         // Screen position
        uint8_t playerXWorld;    // World position
        uint8_t playerY;
        uint8_t powerupState;    // 0: small, 1: big, 2: fire flower
        bool isDying;
        uint8_t world;
        uint8_t level;
        uint32_t score;
        
        /**
         * Extract game info from emulator memory
         */
        void extract(const class NesEmulator& emulator);
    };
    
    // Add more game types here in the future
    
    // Currently active game
    SuperMarioBros mario;
};

/**
 * Exception thrown by the NES emulator
 */
class NesEmulatorException : public std::runtime_error {
public:
    explicit NesEmulatorException(const std::string& message) 
        : std::runtime_error(message) {}
};

/**
 * C++ wrapper around the rustyNES emulator
 */
class NesEmulator {
public:
    /**
     * Create a new emulator and load a ROM
     * @param romPath Path to the ROM file
     * @throws NesEmulatorException if the emulator could not be created
     */
    explicit NesEmulator(const std::string& romPath);
    
    /**
     * Destructor
     */
    ~NesEmulator();
    
    /**
     * Copy constructor (deleted)
     */
    NesEmulator(const NesEmulator&) = delete;
    
    /**
     * Copy assignment operator (deleted)
     */
    NesEmulator& operator=(const NesEmulator&) = delete;
    
    /**
     * Move constructor
     */
    NesEmulator(NesEmulator&& other) noexcept;
    
    /**
     * Move assignment operator
     */
    NesEmulator& operator=(NesEmulator&& other) noexcept;
    
    /**
     * Run emulation for one frame
     */
    void stepFrame();
    
    /**
     * Get current frame buffer
     */
    FrameBuffer getFrameBuffer() const;
    
    /**
     * Get frame width
     */
    int getWidth() const;
    
    /**
     * Get frame height
     */
    int getHeight() const;
    
    /**
     * Set controller state
     */
    void setControllerState(const ControllerState& state);
    
    /**
     * Set controller state using a bitmask
     */
    void setControllerState(uint8_t bitmask);
    
    /**
     * Set a single button state
     */
    void setButton(Button button, bool pressed);
    
    /**
     * Read a memory address
     */
    uint8_t readMemory(uint16_t address) const;
    
    /**
     * Write to a memory address
     */
    void writeMemory(uint16_t address, uint8_t value);
    
    /**
     * Reset the emulator
     */
    void reset();
    
    /**
     * Extract game-specific information
     */
    GameInfo extractGameInfo() const;
    
    /**
     * Save state to a file
     */
    void saveState(const std::string& path) const;
    
    /**
     * Load state from a file
     */
    void loadState(const std::string& path);
    
    /**
     * Check if the emulator is valid
     */
    bool isValid() const;
    
    /**
     * Get a reference to the underlying C emulator interface
     * For advanced usage only
     */
    EmulatorInterface* getNativeHandle();
    
    /**
     * Get a const reference to the underlying C emulator interface
     * For advanced usage only
     */
    const EmulatorInterface* getNativeHandle() const;
    
private:
    EmulatorInterface* m_interface; // Pointer to the rustyNES emulator interface
    
    // Helper method to check if the emulator is valid and throw if not
    void validateEmulator() const;
};

/**
 * Implementation of game-specific information extraction
 */
inline void GameInfo::SuperMarioBros::extract(const NesEmulator& emulator) {
    playerX = emulator.readMemory(0x075A);
    playerXWorld = emulator.readMemory(0x075F);
    playerY = emulator.readMemory(0x0776);
    powerupState = emulator.readMemory(0x075C);
    isDying = emulator.readMemory(0x0770) != 0;
    world = emulator.readMemory(0x000E) + 1; // Add 1 to be 1-based
    level = emulator.readMemory(0x000F) + 1; // Add 1 to be 1-based
    
    // Score is BCD formatted across multiple addresses
    uint8_t score1 = emulator.readMemory(0x0747);
    uint8_t score2 = emulator.readMemory(0x0748);
    uint8_t score3 = emulator.readMemory(0x0749);
    uint8_t score4 = emulator.readMemory(0x074A);
    uint8_t score5 = emulator.readMemory(0x074B);
    uint8_t score6 = emulator.readMemory(0x074C);
    
    // Convert BCD to decimal
    score = 0;
    score += score1 % 16;
    score += (score1 / 16) * 10;
    score += (score2 % 16) * 100;
    score += (score2 / 16) * 1000;
    score += (score3 % 16) * 10000;
    score += (score3 / 16) * 100000;
}

} // namespace emulator

#endif // NES_INTERFACE_H
