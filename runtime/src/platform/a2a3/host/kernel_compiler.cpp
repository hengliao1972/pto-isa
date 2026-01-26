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

int KernelCompiler::CompileKernel(const std::string& sourcePath,
                                  const std::string& ptoIsaRoot,
                                  int coreType,
                                  std::string& outputPath,
                                  std::string& errorMsg) {
    // Step 1: Get compiler path
    std::string compilerPath;
    if (GetCompilerPath(compilerPath) != 0) {
        errorMsg = "Failed to locate ccec compiler. Ensure ASCEND_HOME_PATH is set.";
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

int KernelCompiler::GetCompilerPath(std::string& compilerPath) {
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    if (ascendHome == nullptr) {
        std::cerr << "Error: ASCEND_HOME_PATH environment variable not set" << std::endl;
        return -1;
    }

    compilerPath = std::string(ascendHome) + "/bin/ccec";

    // Check if compiler exists
    struct stat buffer;
    if (stat(compilerPath.c_str(), &buffer) != 0) {
        std::cerr << "Error: ccec compiler not found at " << compilerPath << std::endl;
        return -1;
    }

    return 0;
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

    // Include paths
    cmd << " -I" << ptoIsaRoot << "/include";
    cmd << " -I" << ptoIsaRoot << "/include/pto";

    // Get parent directory of runtime/host for relative includes
    // Assuming source is in runtime/aicore/kernels, we need runtime/ as include
    // Extract runtime directory from source path
    size_t runtimePos = sourcePath.find("/runtime/");
    if (runtimePos != std::string::npos) {
        std::string runtimeDir = sourcePath.substr(0, runtimePos + 9); // Include "/runtime/"
        cmd << " -I" << runtimeDir;
    }

    // Output file
    cmd << " -o " << outputPath;

    // Input source file
    cmd << " " << sourcePath;

    return cmd.str();
}
