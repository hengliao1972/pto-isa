/**
 * Binary Loader - ELF .o File Parser
 *
 * This module provides utilities for loading AICore kernel binaries from
 * compiled .o files (ELF format) and extracting the executable .text section.
 *
 * This implements the binary loading step from the production flow:
 * 1. Read .o file (ELF format)
 * 2. Parse ELF header
 * 3. Extract .text section (executable code)
 * 4. Return raw binary data
 *
 * Based on production code: src/interface/cache/function_cache.cpp:277-320
 */

#ifndef RUNTIME_BINARY_LOADER_H
#define RUNTIME_BINARY_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

/**
 * Get file size in bytes
 *
 * @param filePath  Path to the file
 * @return File size in bytes, or 0 on error
 */
uint32_t GetFileSize(const std::string& filePath);

/**
 * Load and extract .text section from ELF .o file
 *
 * This function:
 * 1. Reads the entire .o file into memory
 * 2. Parses the ELF64 header
 * 3. Locates the .text section (executable code)
 * 4. Extracts and returns the .text section binary data
 *
 * The returned binary data can be copied to device GM memory
 * and executed via function pointer casting.
 *
 * @param binPath  Path to the .o file (ELF format)
 * @return Vector of bytes containing .text section, empty on error
 */
std::vector<uint8_t> LoadBinData(const std::string& binPath);

#endif  // RUNTIME_BINARY_LOADER_H
