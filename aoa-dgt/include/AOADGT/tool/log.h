#ifndef LOG_H
#define LOG_H
#include <iostream>
#include <string>

enum log_mode {QUIET, INFO, WARNING, DEBUG, ERROR, VERBOSE};

/**
 * \brief      Offers a simple mechanism for logging using std::cout.
 * 
 * The Logger class follows the Singleton design pattern.
 * Only those messages tagged with a level lower or equal to
 * the selected mode will be print to log. The supported levels are defined in #log_mode
 * 
 * - QUIET: Nothing shall be written to log.
 * - INFO: Level 1
 * - WARNING: Level 2
 * - DEBUG: Level 3
 * - ERROR: Level 4
 * - VERBOSE: Level 5
 */
class Logger{
    private:
        /* Here will be the instance stored. */
        static Logger* instance;
        int mode;

        /* Private constructor to prevent instancing. */
        Logger(){
            mode = QUIET;
        };

        void __inline__ print(const char *s, int logname);

    public:
        
        /**
         * @brief      This is the way to obtain an instance of this Singleton
         *
         * @return     The instance.
         */
        static Logger* getInstance(){
            if (!instance)
              instance = new Logger;
            return instance;
        }

        void set_mode(int m){
            mode = m;
        }

        void log_debug(const char* s);
        void log_info(const char *s);
        void log_warning(const char* s);
        void log_error(const char* s);
};
#endif