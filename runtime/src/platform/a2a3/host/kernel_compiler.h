/**
 * Runtime Kernel Compiler
 *
 * This module provides runtime compilation of AICore kernel source files (.cpp)
 * to ELF object files (.o) using the ccec compiler.
 *
 * Compilation is performed on-demand without caching - each call to CompileKernel
 * invokes the ccec compiler and generates a new .o file in /tmp.
 *
 * Requirements:
 * - ASCEND_HOME_PATH environment variable must be set
 * - PTO-ISA headers must be available (fetched at build time)
 * - ccec compiler must be available at ${ASCEND_HOME_PATH}/bin/ccec
 */

#ifndef RUNTIME_KERNEL_COMPILER_H
#define RUNTIME_KERNEL_COMPILER_H

#include <string>

/**
 * Runtime kernel compiler class
 *
 * Encapsulates the logic for invoking the ccec compiler with proper flags
 * to compile AICore kernel source code.
 */
class KernelCompiler {
public:
    /**
     * Compile a kernel source file to an object file
     *
     * This function:
     * 1. Validates ASCEND_HOME_PATH and compiler availability
     * 2. Validates PTO-ISA header location
     * 3. Generates unique output path in /tmp
     * 4. Builds ccec command with all required flags
     * 5. Invokes compiler and captures output
     * 6. Returns path to compiled .o file
     *
     * Compilation flags used:
     * - -c -O3 -g -x cce
     * - --cce-aicore-only
     * - --cce-aicore-arch=dav-c220-cube (AIC) or dav-c220-vec (AIV)
     * - -D__AIC__ (for AIC) or -D__AIV__ (for AIV)
     * - Stack size and overflow flags
     * - Include paths for PTO-ISA headers
     *
     * @param sourcePath  Path to kernel source file (.cpp)
     * @param ptoIsaRoot  Path to PTO-ISA root directory (headers location)
     * @param coreType    Core type: 0=AIC, 1=AIV (determines arch and define flags)
     * @param outputPath  Output parameter - path to compiled .o file
     * @param errorMsg    Output parameter - error message if compilation fails
     * @return 0 on success, -1 on error
     *
     * Example:
     *   std::string outputPath, errorMsg;
     *   int rc = KernelCompiler::CompileKernel(
     *       "./aicore/kernels/kernel_mul.cpp",
     *       "/path/to/pto-isa",
     *       1,  // AIV
     *       outputPath,
     *       errorMsg
     *   );
     *   if (rc != 0) {
     *       std::cerr << "Compilation failed: " << errorMsg << std::endl;
     *   }
     */
    static int CompileKernel(const std::string& sourcePath,
                            const std::string& ptoIsaRoot,
                            int coreType,
                            std::string& outputPath,
                            std::string& errorMsg);

private:
    /**
     * Get compiler path from ASCEND_HOME_PATH
     * @param compilerPath  Output parameter - path to ccec compiler
     * @return 0 on success, -1 if not found
     */
    static int GetCompilerPath(std::string& compilerPath);

    /**
     * Validate PTO-ISA headers exist
     * @param ptoIsaRoot  Path to PTO-ISA root directory
     * @return true if headers exist, false otherwise
     */
    static bool ValidatePtoIsaHeaders(const std::string& ptoIsaRoot);

    /**
     * Generate unique output filename in /tmp
     * @return Path to output .o file
     */
    static std::string GenerateOutputPath();

    /**
     * Build ccec compilation command
     * @param compilerPath  Path to ccec compiler
     * @param sourcePath    Path to source file
     * @param outputPath    Path to output .o file
     * @param ptoIsaRoot    Path to PTO-ISA headers
     * @param coreType      Core type: 0=AIC, 1=AIV
     * @return Complete compilation command
     */
    static std::string BuildCompileCommand(const std::string& compilerPath,
                                          const std::string& sourcePath,
                                          const std::string& outputPath,
                                          const std::string& ptoIsaRoot,
                                          int coreType);
};

#endif  // RUNTIME_KERNEL_COMPILER_H
