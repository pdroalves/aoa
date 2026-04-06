#include <AOADGT/tool/log.h>

Logger *Logger::instance = 0;

void Logger::log_debug(const char* s){
    print(s, DEBUG);
}
void Logger::log_info(const char* s){
    print(s, INFO);
}
void Logger::log_warning(const char* s){
    print(s, WARNING);
}
void Logger::log_error(const char* s){
    print(s, ERROR);
}

void __inline__ Logger::print(const char* s, const int logname){
    if(logname <= mode)
        std::cout << s << std::endl;
}