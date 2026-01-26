/**
 * Binary Loader Implementation
 *
 * Implements ELF parsing and .text section extraction for AICore kernel binaries.
 * Based on production code from src/interface/cache/function_cache.cpp
 */

#include "binary_loader.h"
#include <elf.h>
#include <fstream>
#include <iostream>
#include <cstring>

uint32_t GetFileSize(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filePath << '\n';
        return 0;
    }

    std::streamsize size = file.tellg();
    file.close();

    if (size < 0) {
        std::cerr << "Error: Invalid file size for: " << filePath << '\n';
        return 0;
    }

    return static_cast<uint32_t>(size);
}

std::vector<uint8_t> LoadBinData(const std::string& binPath) {
    std::vector<uint8_t> text;

    // Step 1: Read entire file
    uint32_t fileSize = GetFileSize(binPath);
    if (fileSize == 0) {
        std::cerr << "Error: File is empty or cannot be read: " << binPath << '\n';
        return text;
    }

    std::vector<char> buf(fileSize);
    std::ifstream file(binPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << binPath << '\n';
        return text;
    }

    file.read(buf.data(), fileSize);
    if (!file) {
        std::cerr << "Error: Failed to read file: " << binPath << '\n';
        return text;
    }
    file.close();

    // Step 2: Parse ELF header
    auto elfHeader = reinterpret_cast<Elf64_Ehdr*>(buf.data());

    // Verify ELF magic number (0x7F 'E' 'L' 'F')
    if (elfHeader->e_ident[EI_MAG0] != ELFMAG0 ||
        elfHeader->e_ident[EI_MAG1] != ELFMAG1 ||
        elfHeader->e_ident[EI_MAG2] != ELFMAG2 ||
        elfHeader->e_ident[EI_MAG3] != ELFMAG3) {
        std::cerr << "Error: Not a valid ELF file: " << binPath << '\n';
        return text;
    }

    // Step 3: Get section headers
    auto sectionHeaders = reinterpret_cast<Elf64_Shdr*>(
        reinterpret_cast<uint64_t>(elfHeader) + elfHeader->e_shoff);

    // Get string table for section names
    auto shstrHeader = &sectionHeaders[elfHeader->e_shstrndx];
    auto strtbl = buf.data() + shstrHeader->sh_offset;

    // Step 4: Find and extract .text section
    for (int i = 0; i < elfHeader->e_shnum; i++) {
        auto section = &sectionHeaders[i];
        auto sectionName = strtbl + section->sh_name;

        if (std::strcmp(sectionName, ".text") == 0) {
            // Extract .text section binary data
            text.resize(section->sh_size);
            std::memcpy(text.data(), buf.data() + section->sh_offset, section->sh_size);

            std::cout << "Loaded .text section from " << binPath
                      << " (size: " << section->sh_size << " bytes)\n";
            break;
        }
    }

    if (text.empty()) {
        std::cerr << "Error: .text section not found in: " << binPath << '\n';
    }

    return text;
}
