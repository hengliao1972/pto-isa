/**
 * Runtime Kernel Compiler Implementation
 */

#include "kernel_compiler.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <array>
#include <memory>
#include <vector>

int KernelCompiler::CompileKernel(const std::string& sourcePath,
                                  const std::string& ptoIsaRoot,
                                  int coreType,
                                  std::string& outputPath,
                                  std::string& errorMsg) {
    // Step 1: Get compiler path
    std::string compilerPath;
    if (GetCompilerPath(compilerPath) != 0) {
        errorMsg = "Failed to locate CCE compiler. Ensure ASCEND_HOME_PATH is set and the CANN toolchain is available.";
        return -1;
    }

    // Step 2: Validate source file exists
    std::ifstream sourceFile(sourcePath);
    if (!sourceFile.good()) {
        errorMsg = "Source file not found: " + sourcePath;
        return -1;
    }
    sourceFile.close();

    // Step 3: Validate PTO-ISA headers
    if (!ValidatePtoIsaHeaders(ptoIsaRoot)) {
        errorMsg = "PTO-ISA headers not found at: " + ptoIsaRoot;
        return -1;
    }

    // Step 4: Generate output path
    outputPath = GenerateOutputPath();

    // Step 5: Build compilation command
    std::string command = BuildCompileCommand(compilerPath, sourcePath, outputPath, ptoIsaRoot, coreType);

    // Step 6: Execute compilation
    const char* coreTypeName = (coreType == 1) ? "AIV" : "AIC";
    std::cout << "Compiling kernel (" << coreTypeName << "): " << sourcePath << std::endl;
    std::cout << "Command: " << command << std::endl;

    // Redirect stderr to capture compiler output
    std::string redirectedCommand = command + " 2>&1";

    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(redirectedCommand.c_str(), "r"), pclose);

    if (!pipe) {
        errorMsg = "Failed to execute compiler command";
        return -1;
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    int exitCode = pclose(pipe.release());

    if (exitCode != 0) {
        errorMsg = "Compilation failed with exit code " + std::to_string(exitCode) + ":\n" + result;
        std::cerr << errorMsg << std::endl;
        return -1;
    }

    // Step 7: Verify output file exists
    std::ifstream outputFile(outputPath);
    if (!outputFile.good()) {
        errorMsg = "Compilation succeeded but output file not found: " + outputPath;
        return -1;
    }
    outputFile.close();

    std::cout << "Compilation successful: " << outputPath << std::endl;
    return 0;
}

static bool PathExists(const std::string& p) {
    struct stat buffer;
    return stat(p.c_str(), &buffer) == 0;
}

static std::string JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

static bool FindExecutableInPath(const std::string& exe, std::string& outPath) {
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) {
        return false;
    }
    std::string pathStr(pathEnv);
    size_t start = 0;
    while (start <= pathStr.size()) {
        size_t end = pathStr.find(':', start);
        if (end == std::string::npos) {
            end = pathStr.size();
        }
        std::string dir = pathStr.substr(start, end - start);
        if (!dir.empty()) {
            std::string cand = JoinPath(dir, exe);
            if (PathExists(cand)) {
                outPath = cand;
                return true;
            }
        }
        if (end == pathStr.size()) {
            break;
        }
        start = end + 1;
    }
    return false;
}

int KernelCompiler::GetCompilerPath(std::string& compilerPath) {
    const char* overrideCompiler = std::getenv("PTO_CCE_COMPILER");
    if (overrideCompiler != nullptr && std::strlen(overrideCompiler) > 0) {
        std::string p(overrideCompiler);
        if (PathExists(p)) {
            compilerPath = p;
            return 0;
        }
        std::cerr << "Error: PTO_CCE_COMPILER is set but not found at " << p << std::endl;
        return -1;
    }

    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    if (ascendHome == nullptr) {
        std::cerr << "Error: ASCEND_HOME_PATH environment variable not set" << std::endl;
        return -1;
    }

    // CANN toolchains may expose the compiler in different locations:
    // - legacy: ${ASCEND_HOME_PATH}/bin/ccec
    // - current: ${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/ccec
    std::vector<std::string> candidates = {
        std::string(ascendHome) + "/bin/ccec",
        std::string(ascendHome) + "/compiler/ccec_compiler/bin/ccec",
        std::string(ascendHome) + "/bin/bisheng",
        std::string(ascendHome) + "/compiler/ccec_compiler/bin/bisheng",
    };

    for (const auto& p : candidates) {
        if (PathExists(p)) {
            compilerPath = p;
            return 0;
        }
    }

    // Fall back to PATH lookup (useful when the environment is sourced via `setenv.bash`).
    if (FindExecutableInPath("ccec", compilerPath)) {
        return 0;
    }
    if (FindExecutableInPath("bisheng", compilerPath)) {
        return 0;
    }

    std::cerr << "Error: CCE compiler not found (tried ASCEND_HOME_PATH locations and PATH)" << std::endl;
    return -1;
}

bool KernelCompiler::ValidatePtoIsaHeaders(const std::string& ptoIsaRoot) {
    // Check for critical header files
    std::string includePath = ptoIsaRoot + "/include";
    std::string ptoIncludePath = ptoIsaRoot + "/include/pto";

    struct stat buffer;
    if (stat(includePath.c_str(), &buffer) != 0) {
        std::cerr << "Error: PTO-ISA include directory not found: " << includePath << std::endl;
        return false;
    }

    if (stat(ptoIncludePath.c_str(), &buffer) != 0) {
        std::cerr << "Error: PTO-ISA pto include directory not found: " << ptoIncludePath << std::endl;
        return false;
    }

    return true;
}

std::string KernelCompiler::GenerateOutputPath() {
    // Generate unique filename using timestamp and PID
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    pid_t pid = getpid();

    std::ostringstream oss;
    oss << "/tmp/kernel_" << timestamp << "_" << pid << ".o";
    return oss.str();
}

std::string KernelCompiler::BuildCompileCommand(const std::string& compilerPath,
                                               const std::string& sourcePath,
                                               const std::string& outputPath,
                                               const std::string& ptoIsaRoot,
                                               int coreType) {
    std::ostringstream cmd;
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");

    // Compiler executable
    cmd << compilerPath;

    // Core flags
    cmd << " -c -O3 -g -x cce";
    cmd << " -Wall -std=c++17";

    // AICore-specific flags based on core type
    cmd << " --cce-aicore-only";
    if (coreType == 1) {  // AIV
        cmd << " --cce-aicore-arch=dav-c220-vec";
        cmd << " -D__AIV__";
    } else {  // AIC
        cmd << " --cce-aicore-arch=dav-c220-cube";
        cmd << " -D__AIC__";
    }

    // Stack and memory flags
    cmd << " -mllvm -cce-aicore-stack-size=0x8000";
    cmd << " -mllvm -cce-aicore-function-stack-size=0x8000";
    cmd << " -mllvm -cce-aicore-record-overflow=false";
    cmd << " -mllvm -cce-aicore-addr-transform";
    cmd << " -mllvm -cce-aicore-dcci-insert-for-scalar=false";
    cmd << " -DMEMORY_BASE";

    // Enable device-side debug prints (e.g., TPRINT) when requested.
    // `include/pto/npu/*/TPrint.hpp` gates printing on `_DEBUG`.
    const char* cceDebug = std::getenv("PTO_CCE_DEBUG");
    if (cceDebug != nullptr && std::strlen(cceDebug) > 0 && std::string(cceDebug) != "0") {
        cmd << " -D_DEBUG";
        // Enable CCE debug tunnel printing (`cce::printf`).
        // Without these, the toolchain doesn't provide `cce::printf` symbols.
        cmd << " -D__CCE_ENABLE_PRINT__ -D__CCE_ENABLE_PRINT_AICORE__";
    }

    // Include paths
    // PTO ISA headers (this repo).
    cmd << " -I" << ptoIsaRoot << "/include";
    cmd << " -I" << ptoIsaRoot << "/include/pto";

    // Ascend CANN/AscendC headers (needed for `kernel_operator.h`).
    if (ascendHome != nullptr && std::strlen(ascendHome) > 0) {
        std::string ah(ascendHome);
        std::vector<std::string> includes = {
            // AscendC (CANN 8.x) primary include roots.
            ah + "/compiler/ascendc/include/basic_api",
            ah + "/compiler/ascendc/include/basic_api/interface",
            ah + "/compiler/ascendc/include/basic_api/impl",
            // Some toolchains still use the older `asc` layout.
            ah + "/compiler/asc/include/basic_api",
            ah + "/compiler/asc/include/interface",
            ah + "/compiler/asc",
            // High-level AscendC API (tiling, etc.).
            ah + "/include/ascendc/highlevel_api",
            ah + "/include/ascendc",
            // General CANN includes.
            ah + "/include",
            ah + "/runtime/include",
        };
        for (const auto& inc : includes) {
            if (PathExists(inc)) {
                cmd << " -I" << inc;
            }
        }
    }

    // Output file
    cmd << " -o " << outputPath;

    // Input source file
    cmd << " " << sourcePath;

    return cmd.str();
}
