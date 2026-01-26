/**
 * Device logging implementation for AICPU kernel
 */

#include "device_log.h"

bool g_isLogEnableDebug = false;
bool g_isLogEnableInfo = false;
bool g_isLogEnableWarn = false;
bool g_isLogEnableError = false;

void InitLogSwitch() {
    g_isLogEnableDebug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_isLogEnableInfo = CheckLogLevel(AICPU, DLOG_INFO);
    g_isLogEnableWarn = CheckLogLevel(AICPU, DLOG_WARN);
    g_isLogEnableError = CheckLogLevel(AICPU, DLOG_ERROR);
}
