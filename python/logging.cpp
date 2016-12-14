#include "logging.h"

namespace skeletopyze {

logger::LogChannel pylog("pylog", "[skeletopyze] ");

logger::LogLevel getLogLevel() {
	return logger::LogManager::getGlobalLogLevel();
}

void setLogLevel(logger::LogLevel logLevel) {
	logger::LogManager::setGlobalLogLevel(logLevel);
}

} // namespace skeletopyze
